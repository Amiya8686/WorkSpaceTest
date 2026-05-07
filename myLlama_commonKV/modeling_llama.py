# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn

# from ...activations import ACT2FN
# from ...cache_utils import Cache, DynamicCache
# from ...generation import GenerationMixin
# from ...integrations import use_kernel_forward_from_hub
# from ...masking_utils import create_causal_mask
# from ...modeling_layers import (
#     GenericForQuestionAnswering,
#     GenericForSequenceClassification,
#     GenericForTokenClassification,
#     GradientCheckpointingLayer,
# )
# from ...modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
# )
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from ...processing_utils import Unpack
# from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
# from ...utils.deprecation import deprecate_kwarg
# from ...utils.generic import check_model_inputs
# from .configuration_llama import LlamaConfig

# --- 修改开始 ---
# 将所有 ... 改为 transformers.，这样它会去你安装的库里找通用工具
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig 


import torch





class AdaptiveRoPECache:
    """ROPE参数缓存类:若空间不足,一开始容量成倍增长,后面线性增长"""
    def __init__(self, head_dim, device, dtype, initial_capacity=512, threshold=4096, linear_step=2048):
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # 扩容策略参数
        self.capacity = initial_capacity  # 当前预分配的总容量
        self.threshold = threshold        # 切换到线性增长的阈值
        self.linear_step = linear_step    # 线性增长的步长
        
        # 预分配 Buffer
        self.cos_buffer = torch.zeros((1, self.capacity, self.head_dim), device=device, dtype=dtype)
        self.sin_buffer = torch.zeros((1, self.capacity, self.head_dim), device=device, dtype=dtype)
        
        self.seen_tokens = 0

    def grow(self, min_capacity):
        """执行扩容逻辑"""
        # 计算新容量
        old_capacity = self.capacity
        new_capacity = old_capacity
        while new_capacity < min_capacity:
            if new_capacity < self.threshold:
                new_capacity = new_capacity * 2 # 两倍增长
            else:               
                new_capacity = new_capacity + self.linear_step # 线性增长

        # 申请新空间并拷贝数据
        new_cos = torch.zeros((1, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
        new_sin = torch.zeros((1, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
        new_cos[:, :old_capacity, :] = self.cos_buffer
        new_sin[:, :old_capacity, :] = self.sin_buffer
        self.cos_buffer = new_cos
        self.sin_buffer = new_sin
        self.capacity = new_capacity

    def update(self, new_cos, new_sin,cache_position):
        """
        传入新生成的 cos, sin,返回从 0 到当前位置的全量视图
        new_cos shape: [batch, length, head_dim]
        """
        cache_position = cache_position.reshape(-1).to(torch.long)
        required_capacity = int(cache_position.max().item()) + 1
        
        # 如果当前容量不够，触发扩容
        if required_capacity > self.capacity:
            self.grow(required_capacity)
        
        # 原地索引赋值 (In-place)
        self.cos_buffer[:, cache_position, :] = new_cos.detach()
        self.sin_buffer[:, cache_position, :] = new_sin.detach()
        
        self.seen_tokens = max(self.seen_tokens, required_capacity)
        
        # 返回已填充部分的视图 (View)，零拷贝
        return self.cos_buffer[:, :self.seen_tokens, :], self.sin_buffer[:, :self.seen_tokens, :]

    def get_len(self):
        """获取当前已存储的 token 数量"""
        return self.seen_tokens

    def reset(self):
        """重置游标，但不释放内存（方便复用）"""
        self.seen_tokens = 0


class LatentKVMerge_MiniCaceh_X:
    """MiniCaceh_X合并类"""
    def __init__(
            self,
            layer_num,
            device,
            dtype,
            batch_size,
            hidden_dim,
            layer_buffers,
            layer_seen_tokens,
            config: LlamaConfig,
            merge_args = None,
            ):
        self.config = config

        # 记录张量基础信息
        self.layer_num = layer_num
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # 合并所需参数
        self.merge_args = merge_args

        size = [self.batch_size,0,self.hidden_dim]
        mask_size = [self.batch_size,0]
        outliner_size = [self.batch_size,0,self.hidden_dim]
        modulus_size = [self.batch_size,0]

        # 层缓存(concat追加)
        self.layer_buffers = layer_buffers
        self.layer_seen_tokens = layer_seen_tokens

        # 组缓存(concat追加)
        self.group_dv = torch.zeros(size=size,dtype=self.dtype, device=self.device)
        self.group_modulus = [torch.zeros(size=modulus_size,dtype=self.dtype, device=self.device) for _ in range(layer_num)]
        self.group_mask = torch.zeros(size=mask_size, dtype=torch.bool, device=self.device)
        self.outliners = [torch.zeros(size=outliner_size,dtype=self.dtype, device=self.device) for _ in range(layer_num)]
        self.group_seen_tokens = 0

    def merge(self):
        """合并层缓存,追加到组缓存"""
        # 从layer_buffers中拿取merge_step长度的latentKV（用一个while循环比较好）
        eps = 1e-6

        while min(self.layer_seen_tokens) >= self.merge_args["MiniCache_X_merge_step"]:
            # 取出当前合并步的窗口
            layer_chunks = []
            layer_modulus_chunks = []
            layer_unit_chunks = []

            for layer_idx in range(self.layer_num):
                chunk = self.layer_buffers[layer_idx][:, :self.merge_args["MiniCache_X_merge_step"], :]
                modulus = torch.linalg.norm(chunk, dim=-1).clamp_min(eps)
                unit = chunk / modulus.unsqueeze(-1)

                layer_chunks.append(chunk)
                layer_modulus_chunks.append(modulus)
                layer_unit_chunks.append(unit)

            # 计算latentKV的层间单位方向向量均值
            stacked_units = torch.stack(layer_unit_chunks, dim=0)
            group_dv = stacked_units.sum(dim=0)
            group_dv = group_dv / torch.linalg.norm(group_dv, dim=-1, keepdim=True).clamp_min(eps)

            # 计算每一层相对单位方向向量均值的余弦相似度
            group_dv_expand = group_dv.unsqueeze(0)
            cosine_sim = torch.nn.functional.cosine_similarity(stacked_units, group_dv_expand, dim=-1)

            # 每一层的余弦相似度取均值
            mean_cosine_sim = cosine_sim.mean(dim=0)

            # 选取层间余弦相似度均值最小的token作为离群值
            outlier_pos = mean_cosine_sim.argmin(dim=1)

            # 追加离群值矩阵，保存离群值
            batch_index = torch.arange(self.batch_size, device=self.device)
            gather_index = outlier_pos.view(self.batch_size, 1, 1).expand(-1, 1, self.hidden_dim)

            outlier_mask = torch.zeros((self.batch_size, self.merge_args["MiniCache_X_merge_step"]), dtype=torch.bool, device=self.device)
            outlier_mask[batch_index, outlier_pos] = True
            self.group_mask = torch.concat([self.group_mask, outlier_mask], dim=1)

            # 追加单位方向向量矩阵 和 模长矩阵
            self.group_dv = torch.concat([self.group_dv, group_dv.to(self.dtype)], dim=1)
            for layer_idx in range(self.layer_num):
                self.group_modulus[layer_idx] = torch.concat(
                    [self.group_modulus[layer_idx], layer_modulus_chunks[layer_idx].to(self.dtype)],
                    dim=1,
                )

                outlier_value = layer_chunks[layer_idx].gather(1, gather_index)
                self.outliners[layer_idx] = torch.concat([self.outliners[layer_idx], outlier_value.to(self.dtype)], dim=1)

                # 已合并的前缀从层缓存中移除，剩余部分继续等待下一次merge
                self.layer_buffers[layer_idx] = self.layer_buffers[layer_idx][:, self.merge_args["MiniCache_X_merge_step"]:, :]
                self.layer_seen_tokens[layer_idx] = self.layer_buffers[layer_idx].shape[1]

            self.group_seen_tokens = self.group_dv.shape[1]

        # 释放原来的张量 (避免存储一个过大的张量)
        for layer_idx in range(self.layer_num):
            self.layer_buffers[layer_idx] = self.layer_buffers[layer_idx].clone()
    
    def unmerge(self,idx):
        """解压组缓存"""
        if self.group_seen_tokens == 0:
            return torch.zeros((self.batch_size, 0, self.hidden_dim), dtype=self.dtype, device=self.device)

        # 先把所有已合并token的单位方向向量乘回模长，得到基础重建值
        group_dv = self.group_dv[:, :self.group_seen_tokens, :]
        group_modulus = self.group_modulus[idx][:, :self.group_seen_tokens]
        group_buffer = group_dv * group_modulus.unsqueeze(-1)

        # 按同样的 batch/step 顺序一次性回填所有离群值
        if self.outliners[idx].shape[1] > 0:
            group_buffer[self.group_mask] = self.outliners[idx].reshape(-1, self.hidden_dim)

        return group_buffer
    
    def get_group_len(self):
        return self.group_seen_tokens



class LatentKVMerge_Mean:
    """均值合并算法类"""
    def init():...

    def merge():...

    def ummerge():...

    def get_group_len():...

class LatentKVGroup:
    """存储单组的LatenKV [B,HN,S,HD]"""
    def __init__(
            self,
            layer_num,
            device,
            dtype,
            batch_size,
            hidden_dim,
            config: LlamaConfig,
            init_capacity = 512,
            threshold=4096,
            linear_step=1024,
            merge_args = None
            ):
        # 记录张量基础信息
        self.layer_num = layer_num
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # 记录增长信息
        self.threshold = threshold
        self.linear_step = linear_step

        # 合并所需参数
        self.merge_args = merge_args


        size = [self.batch_size,0,self.hidden_dim]
        mask_size = [self.batch_size,0]
        outliner_size = [self.batch_size,0,self.hidden_dim]
        modulus_size = [self.batch_size,0]

        # 层缓存(concat追加)
        self.layer_buffers = [torch.zeros(size=size,dtype=self.dtype, device=self.device) for _ in range(layer_num)]
        self.layer_seen_tokens = [0 for _ in range(layer_num)]

        # 合并算法类
        if self.merge_args["merge_algorithm"] == "MiniCache_X":
            self.Merger = LatentKVMerge_MiniCaceh_X(
                layer_num = self.layer_num,
                device = self.device,
                dtype = self.device,
                batch_size = self.batch_size,
                hidden_dim = self.hidden_dim,
                layer_buffers = self.layer_buffers,
                layer_seen_tokens = self.layer_seen_tokens,
                config = self.config,
                merge_args = self.merge_args
            )
        elif self.merge_args["merge_algorithm"] == "Mean"
            self.Merger = LatentKVMerge_Mean(
                layer_num = self.layer_num,
                device = self.device,
                dtype = self.device,
                batch_size = self.batch_size,
                hidden_dim = self.hidden_dim,
                layer_buffers = self.layer_buffers,
                layer_seen_tokens = self.layer_seen_tokens,
                config = self.config,
                merge_args = self.merge_args
            )
        
 
    def update(self, idx, latentKV):
        """更新层缓存,适时合并组缓存"""
        # 记录层缓存
        self.layer_buffers[idx] = torch.concat([self.layer_buffers[idx], latentKV], dim=1)
        self.layer_seen_tokens[idx] = self.layer_buffers[idx].shape[1]

        # 判断是否合并组缓存
        if(self.merge_args["is_merge"] and idx == self.layer_num-1):
            self.merge()

        # 返回完整的latentKV:
        return self.get_latent_kv(idx)

    def merge(self):
        """合并层缓存,追加到组缓存"""
        self.Merger.merge()

    def unmerge(self, idx):
        """解压组缓存"""
        return self.Merger.unmerge(idx)

    def get_layer_len(self, idx):
        return self.layer_seen_tokens[idx]

    def get_group_len(self):
        return self.Merger.get_group_len()

    def get_latent_kv(self, idx):
        """返回该层的lantentKV"""
        if self.merge_args["is_merge"]:
            return torch.concat([self.unmerge(idx), self.layer_buffers[idx]], dim=1)
        else:
            return self.layer_buffers[idx]
    
class LatentKVCache:
    """存储潜在KV的类"""
    def __init__(
        self,
        commonKV_meta_data,
        device,
        dtype,
        batch_size,
        hidden_dim,
        config: LlamaConfig,
        init_capacity = 512,
        threshold=4096,
        linear_step=1024,
        merge_args = None
        ):
        
        # 存储元数据
        self.device = device
        self.dtype = dtype
        self.merge_args = merge_args

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.config = config
        self.commonKV_meta_data = commonKV_meta_data
        self.init_capacity = init_capacity
        self.threshold = threshold
        self.linear_step = linear_step

        # 维护每一组的LatentKVGroup
        # 创建一个 层序号 -> 组号，组内序号 的映射表表
        self.kvGroups = []
        self.mapTable = {}
        for group_idx in  range(len(commonKV_meta_data["group_lists"])):
            group_list = commonKV_meta_data["group_lists"][group_idx]
            layer_num = len(group_list)
            self.kvGroups.append(LatentKVGroup(
                layer_num = layer_num,
                device = self.device,
                dtype = self.dtype,
                batch_size = self.batch_size,
                hidden_dim = commonKV_meta_data["group_r"][group_idx],
                config = config,
                init_capacity = init_capacity,
                threshold = threshold,
                linear_step = linear_step,
                merge_args=merge_args
                ))
            # 建立映射表
            for group_in_idx in range(len(group_list)):
                self.mapTable[group_list[group_in_idx]] = {"g":group_idx,"i":group_in_idx}


    def update(self,latentKV:torch.tensor, layer_idx): 
        # 调用对应LatentKVGroup的update
        g,i = self.mapTable[layer_idx]["g"],self.mapTable[layer_idx]["i"]
        return self.kvGroups[g].update(i,latentKV)

    def get_latent_kv(self,layer_idx):
        g,i = self.mapTable[layer_idx]["g"],self.mapTable[layer_idx]["i"]
        return self.kvGroups[g].get_latent_kv(i)

    def get_seq_length(self):
        if not self.mapTable:
            return 0

        max_seq_length = 0
        for group in self.kvGroups:
            group_len = group.get_group_len()
            for layer_idx in range(group.layer_num):
                max_seq_length = max(max_seq_length, group.get_layer_len(layer_idx) + group_len)
        return max_seq_length

    def get_commonKV_meta_data(self):
        return self.commonKV_meta_data


# __name__ 调用路径（比如直接执行就是__main__,引用执行就是from后面的路径值）
# logger可以理解为一个带上身份标签的print，在终端输出时附上身份标识
logger = logging.get_logger(__name__)


# 类装饰器：本质上是个函数
# 不带参数:"@decorate",接受接受类，返回修饰后的类
# 带参数:"@decorate(arg1,arg2)",要求返回接受类的函数
# 优先使用transformers的RMSNorm计算算子（优化更彻底，计算更快）
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    # hidden_size就是token的维度
    # Parameter是一个注册的张量，我们可以在配置文件中修改它（一般训练得到）
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    # 计算RMS,然后除以RMS的平方根,最后重参数化（self.weight就是gamma向量）
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    #print这个类时候的输出
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# llama的旋转位置编码，在每一层对Q和K使用（可能会影响Kcache的层间相似性，之后研究）
# 所有头共用一组ROPE参数，且最终输出的cos和sin前半部分和后半部分相同
# 对于一次填入多个token的情况，会算出每个token的ROPE参数
class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # cos和sin的维度 [batch_size,seq_len,head_dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# 这两个函数是实现Rope的辅助函数
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    #[B,S,HD] -> [B,1,S,HD]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_commonKV(q, k, cos, sin, cache_position ,position_ids=None, unsqueeze_dim=1):
    #[B,S,HD] -> [B,1,S,HD]
    # 获取切片
    cache_position = cache_position.reshape(-1).to(torch.long)
    q_cos = cos[:, cache_position, :]
    q_sin = sin[:, cache_position, :]

    k_history_len = int(cache_position[-1].item()) + 1
    k_cos = cos[:, :k_history_len, :]
    k_sin = sin[:, :k_history_len, :]

    q_cos = q_cos.unsqueeze(unsqueeze_dim)
    q_sin = q_sin.unsqueeze(unsqueeze_dim)
    k_cos = k_cos.unsqueeze(unsqueeze_dim)
    k_sin = k_sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


# 全链接层（前馈神经网络）
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# 使用K和V的头数比Q少，所有进行运算时，要先复制K和V
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 获得经过因果注意力计算的token序列
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # attension_mask的最后两个维度是个上三角矩阵，数值是一个很小的负数（softmax之后就是0了）
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #训练时进行dropOut，随机将注意力置为0
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    #本来没有头维度的，头维度拆分之初是在倒数第二维的，但为了方便计算因果注意力，和序列长度维度交换了
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # 得到QKV的线性层（可能会出现升维后降维的情况，所以有o_proj）
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        #CommonKV使用到的参数
        self.A_proj = None
        self.B_k_proj = None
        self.B_v_proj = None
        self.use_commonKV = False
        self.use_model_commonKV = False #让没有启动commonKV的层也能够知道模型启动了commonKV

        #let变换使用到的参数
        self.use_let = False
        self.register_parameter("let_factor", None)

        #let训练需要用到的参数
        self.save_lkv = False
        self.lkv = None

        self.save_lkv_after_let = False
        self.lkv_after_let = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # commonKV分支
        if self.use_commonKV:
            batch_size, query_len, _ = hidden_states.shape

            # A投影
            latent_kv = self.A_proj(hidden_states).contiguous()

            # 存储let之前的lkv
            if self.save_lkv:
                self.lkv = latent_kv.clone()

            # 存储let之后的latentKV
            if self.save_lkv_after_let:
                let_factor = self.let_factor.to(device=latent_kv.device, dtype=latent_kv.dtype)
                self.lkv_after_let = latent_kv * let_factor
            
            # 存储潜在KV
            if past_key_values is not None:
                
                #对latent_kv缩放
                if self.use_let and self.let_factor is not None:
                    let_factor = self.let_factor.to(device=latent_kv.device, dtype=latent_kv.dtype)
                    latent_kv = latent_kv * let_factor

                commonKV_cache = past_key_values.commonKV_cache
                latent_kv = commonKV_cache.update(latent_kv,self.layer_idx)

                #对lantent_kv恢复
                if self.use_let and self.let_factor is not None:
                    let_factor = self.let_factor.to(device=latent_kv.device, dtype=latent_kv.dtype)
                    latent_kv = latent_kv / let_factor

            key_len = latent_kv.shape[1]


            # 计算qkv（重构）
            query_states = self.q_proj(hidden_states).reshape(batch_size, query_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = self.B_k_proj(latent_kv).reshape(batch_size, key_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = self.B_v_proj(latent_kv).reshape(batch_size, key_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

            # rope
            # copilor修改
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_commonKV(query_states, key_states, cos, sin,cache_position=cache_position)
            

            # 注意力计算：内部会处理GQA，将KV头数复制到与Q一致
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

           
            # attn_output:qkv结果； attn_weights:qv注意力分数
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            

            # [B,HN,S,HD] -> [B,S,H]
            attn_output = attn_output.reshape(batch_size, query_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights
        
        else:
            # 计算出目标张量维度[批次，序列长度，头数目，头特征维度]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # view 会自动计算 -1 的维度（将原始特征维度，拆分成头数目维度和头特征维度）
            # [B,S,H] -> [B,S,HN,HD] -> [B,HN,S,HD]
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            

            # 旋转位置编码：每个头分别做
            cos, sin = position_embeddings
            if self.use_model_commonKV:
                cos = cos[:, cache_position, :]
                sin = sin[:, cache_position, :]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            

            # KVcache存储（不一定会启用KVcache，所以上面才会保留序列长度维度，虽然是1）
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # 注意力计算：内部会处理GQA，将KV头数复制到与Q一致
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

           

            # attn_output:qkv结果； attn_weights:qv注意力分数
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            # [B,HN,S,HD] -> [B,S,H]
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    def switch_to_commonKV(
            self,
            A:torch.tensor,
            B_k:torch.tensor,
            B_v:torch.tensor,
            B_k_bias:torch.tensor = None,
            B_v_bias:torch.tensor = None,
            ):
        """切换到commonKV模式"""
        self.use_commonKV = True

        # 释放原来的Liner
        self.k_proj = None
        self.v_proj = None

        # 设置新的Linear
        self.A_proj = nn.Linear(A.shape[1],A.shape[0],bias=False)
        self.B_k_proj = nn.Linear(B_k.shape[1],B_k.shape[0],bias=(B_k_bias is not None))
        self.B_v_proj = nn.Linear(B_v.shape[1],B_v.shape[0],bias=(B_v_bias is not None))

        self.A_proj.weight = nn.Parameter(A)
        self.B_k_proj.weight = nn.Parameter(B_k)
        self.B_v_proj.weight = nn.Parameter(B_v)
        if B_k_bias is not None:
            self.B_k_proj.bias = nn.Parameter(B_k_bias)
        if B_v_bias is not None:
            self.B_v_proj.bias = nn.Parameter(B_v_bias)
            
        # 将权重参数移动到相同的设备
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.A_proj.to(device=device, dtype=dtype)
        self.B_k_proj.to(device=device, dtype=dtype)
        self.B_v_proj.to(device=device, dtype=dtype)

    def switch_to_model_commonKV(self):
        self.use_model_commonKV = True

    def switch_to_let(
            self,
            active:bool=False,
            let_factor:torch.tensor=None
        ):
        self.use_let = active
        if(let_factor is not None):
            device = self.q_proj.weight.device
            dtype = self.q_proj.weight.dtype
            self.let_factor = nn.Parameter(let_factor.to(device=device, dtype=dtype).detach().clone())
        elif not active:
            self.let_factor = None
    
    def set_save_lkv_after_let(self,active):
        self.save_lkv_after_let = active
        if(not active):
            self.lkv_after_let = None
    
    def get_lkv_after_let(self):
        lkv_after_let = self.lkv_after_let
        self.lkv_after_let = None
        return lkv_after_let
    
    def set_save_lkv(self,active):
        self.save_lkv = active
        if(not active):
            self.lkv = None
    
    def get_lkv(self):
        lkv = self.lkv
        self.lkv = None
        return lkv


# 将前面定义的MLP，RSMNorm，Attension封装成一个解码器
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.save_hidden_state = False
        self.hidden_state = None

        self.save_lkv_after_let = False
        self.save_lkv = False

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 保存Hidden_State
        if self.save_hidden_state:
            self.hidden_state = hidden_states
            


        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def set_save_hidden_state(self,active:bool=False):
        """设置保存隐藏状态,方便提取以实验"""
        self.save_hidden_state = active
        if(active==False):
            self.hidden_state = None

    def get_hidden_state(self):
        """获取隐藏状态"""
        return self.hidden_state

    def set_save_lkv_after_let(self,active):
        self.save_lkv_after_let = active
        self.self_attn.set_save_lkv_after_let(active)

    def get_lkv_after_let(self):
        return self.self_attn.get_lkv_after_let()
    
    def set_save_lkv(self,active):
        self.save_lkv = active
        self.self_attn.set_save_lkv(active)

    def get_lkv(self):
        return self.self_attn.get_lkv()
# llama模型的基类（所有Llama模型的配置）
@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

# 当前Llama模型的特征提取部分（只负责一次前向传播）
@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # CommonKV使用到的一些参数
        self.use_commonKV = False
        self.position_embeddings_table = None
        self.commonKV_meta_data = None

        #let
        self.use_let = False
        self.let_meta_data = None

        #latentKV合并
        self.merge_args = None
        
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if self.use_commonKV and not hasattr(past_key_values, "commonKV_cache"):
                past_key_values.commonKV_cache = LatentKVCache(
                    commonKV_meta_data=self.commonKV_meta_data,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                    batch_size=inputs_embeds.shape[0],
                    hidden_dim=self.config.hidden_size,
                    config=self.config,
                    merge_args = self.merge_args
                )
                past_key_values.get_seq_length = past_key_values.commonKV_cache.get_seq_length


        if cache_position is None:
            if self.use_commonKV and past_key_values is not None and hasattr(past_key_values, "commonKV_cache"):
                past_seen_tokens = past_key_values.commonKV_cache.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.use_commonKV:
            if past_key_values is None and self.position_embeddings_table is not None:
                self.position_embeddings_table.reset()
            if self.position_embeddings_table is None:
                self.position_embeddings_table = AdaptiveRoPECache(
                    head_dim=position_embeddings[0].shape[-1],
                    device=position_embeddings[0].device,
                    dtype=position_embeddings[0].dtype,
                )
            position_embeddings = self.position_embeddings_table.update(
                position_embeddings[0], position_embeddings[1], cache_position
            )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def switch_to_commonKV(self, commonKV_meta_data, commonKV_parameters, merge_args):
        """切换到CommonKV形态"""
        self.use_commonKV = True
        self.commonKV_meta_data = commonKV_meta_data
        self.merge_args = merge_args

        groups = commonKV_meta_data.get("group_lists", [])
        if not groups:
            raise KeyError("Missing CommonKV metadata: group_lists")

        for group_idx, group_layers in enumerate(groups):
            a_key = f"A_group{group_idx}"
            if a_key not in commonKV_parameters:
                raise KeyError(f"Missing CommonKV parameter: {a_key}")
            a_tensor = commonKV_parameters[a_key]
            for layer_idx in group_layers:
                k_key = f"B_{layer_idx}_k"
                v_key = f"B_{layer_idx}_v"
                if k_key not in commonKV_parameters:
                    raise KeyError(f"Missing CommonKV parameter: {k_key}")
                if v_key not in commonKV_parameters:
                    raise KeyError(f"Missing CommonKV parameter: {v_key}")

                self.layers[layer_idx].self_attn.switch_to_commonKV(
                    A=a_tensor,
                    B_k=commonKV_parameters[k_key],
                    B_v=commonKV_parameters[v_key],
                    B_k_bias=commonKV_parameters.get(f"B_{layer_idx}_k_bias"),
                    B_v_bias=commonKV_parameters.get(f"B_{layer_idx}_v_bias"),
                )
        
        
        for i in range(len(self.layers)):
            self.layers[i].self_attn.switch_to_model_commonKV()

    def switch_to_let(self, active, let_meta_data, let_parameters):
        """切换到LET形态"""
        self.use_let = active
        self.let_meta_data = let_meta_data

        let_layers = let_meta_data.get("let_layers", let_meta_data.get("let_layers:", []))
        if not let_layers:
            raise KeyError("Missing LET metadata: let_layers")

        for layer_idx in let_layers:
            let_key = f"let_l{layer_idx}"
            if let_key not in let_parameters:
                raise KeyError(f"Missing LET parameter: {let_key}")

            self.layers[layer_idx].self_attn.switch_to_let(
                active=active,
                let_factor=let_parameters[let_key],
            )

@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        print("🚀 [SUCCESS] 正在使用本地 myLlama 文件夹中的代码进行推理！(commonKV)")

        self.post_init()

        self.use_commonKV = False
        self.commonKV_meta_data = None
        self.use_let = False
        self.let_meta_data = None
        self.merge_args = None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def switch_to_commonKV(self,commonKV_meta_data,commonKV_parameters,merge_args):
        self.merge_args = merge_args
        self.use_commonKV = True
        self.commonKV_meta_data = commonKV_meta_data
        self.model.switch_to_commonKV(commonKV_meta_data,commonKV_parameters,merge_args)

    def set_save_hidden_state(self,active:bool=False):
        """设置DecoderLayer存储隐藏状态"""
        for layer in self.model.layers:
            layer.set_save_hidden_state(active)

    def get_hidden_state(self):
        """获取隐藏状态"""
        hidden_states = []
        for layer in self.model.layers:
            hidden_states.append(layer.get_hidden_state())
        return hidden_states
    
    def switch_to_let(self,active,let_meta_data,let_parameters):
        self.use_let = active
        self.let_meta_data = let_meta_data
        self.model.switch_to_let(active,let_meta_data,let_parameters)
    
class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...

class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`

class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...

__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
