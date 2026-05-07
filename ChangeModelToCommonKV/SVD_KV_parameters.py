import argparse
import os
import torch
import gc
import json
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

# 0.加载参数
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="The path of model",
                    default="./model_cache/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--save_dir", type=str, help="The path to save SVD parameters",
                    default="./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--group_num", type=int, help="The num of layer to join SVD for KV parameters",
                    default=4)
parser.add_argument("--energy_threshold", type=str, help="The proportion of retained energy",
                    default=0.9)
# [begin_layer,end_layer)进行分组合并
parser.add_argument("--begin_layer", type=int, help="The layer beginning to merge",default=0)
parser.add_argument("--end_layer",type=int, help="The layer endding merging",default=32)
args = parser.parse_args()




print("================Beginning to join SVD==================")
# 1.加载模型
print(f"Loading model from {args.model}...")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cpu")
layers = model.model.layers
num_layers = args.end_layer - args.begin_layer
svd_params_to_save = {}


# 2.获取每一组对应的层
remainder = num_layers % args.group_num
regular_group_start = args.begin_layer + remainder
energy_threshold = float(args.energy_threshold)
groups = []
group_r = []
if remainder > 0:
    groups.append(list(range(args.begin_layer, regular_group_start)))
for i in range(regular_group_start, args.end_layer, args.group_num):
    groups.append(list(range(i, i + args.group_num)))

# 3.对每一组进行联合SVD
for i in range(len(groups)):
    group_layers = groups[i]
    print(f"Processing Group {i}: {group_layers}")
    
    d_out = layers[0].self_attn.k_proj.weight.shape[0]


    # 获取待联合SVD的权重矩阵
    weights_to_cat = []
    for l_idx in group_layers:
        wk = layers[l_idx].self_attn.k_proj.weight.data
        wv = layers[l_idx].self_attn.v_proj.weight.data
        weights_to_cat.append(wk)
        weights_to_cat.append(wv)

        # c存储KV参数的偏置项
        if layers[l_idx].self_attn.k_proj.bias is not None:
            svd_params_to_save[f"B_{l_idx}_k_bias"] = layers[l_idx].self_attn.k_proj.bias.data.clone()
        if layers[l_idx].self_attn.v_proj.bias is not None:
            svd_params_to_save[f"B_{l_idx}_v_bias"] = layers[l_idx].self_attn.v_proj.bias.data.clone()
        
    # 输出维度方向拼接: [g * 2 * out, in]
    W_combined = torch.cat(weights_to_cat, dim=0)
    # 转置：
    W_t = W_combined.t().to(torch.float32)
    # 奇异值分解: W_t = U * S * Vh  #经济型SVD：把奇异值矩阵都是0的行或者列的奇异向量先剔除了（S为一维向量）
    U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)   
    

    # 提取能量： 确定压缩秩 r
    energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    r = torch.where(energy >= energy_threshold)[0][0].item() + 1
    group_r.append(r)
    print(f"  Group {i} Rank: {r} Total:{energy.shape[0]} (Saved Energy: {energy[r-1]})")
    print(f"Energy List : {energy}")
    

    # 计算 A: A = U * S (前 r 列), 维度 [in, r]
    A_group = U[:, :r] @ torch.diag(S[:r])
    svd_params_to_save[f"A_group{i}"] = A_group.t().contiguous()
    

    # 计算 B: B_all = Vh (前 r 行), 维度 [r, g * 2 * out]
    B_all = Vh[:r, :]
    B_list = torch.split(B_all, d_out, dim=1)
    for idx, l_idx in enumerate(group_layers):
        # 拿取计算BWo的算子融合
        # wo = layers[l_idx].self_attn.o_proj.weight.data.to(torch.float32).t()
        # bv = B_list[idx * 2 + 1] 
        # bv_fused = bv @ wo
        # 存储:GQA场景下，先不进行Bv和Wo的算子融合，不然算子注意力计算那里太复杂了
        svd_params_to_save[f"B_{l_idx}_k"] = B_list[idx * 2].t().contiguous()
        svd_params_to_save[f"B_{l_idx}_v"] = B_list[idx * 2+1].t().contiguous()
        # svd_params_to_save[f"B_{l_idx}_v"] = bv_fused.t().contiguous()


# 4.释放模型权重
del model
gc.collect()

# 5.存储数据和元数据
save_dir = f"{args.save_dir}/g{args.group_num}_e{args.energy_threshold}_b{args.begin_layer}_e{args.end_layer}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
metadata = {
    "model": str(args.model),
    "layers_num": str(len(layers)),
    "commonKV_layers_num":str(num_layers),
    "group_num": str(args.group_num),
    "energy_threshold": str(args.energy_threshold),
    "group_lists": json.dumps(groups),
    "group_r":json.dumps(group_r)
}
save_path = os.path.join(save_dir, "commonKV_parameters.safetensors")
save_file(svd_params_to_save, save_path, metadata=metadata)
print(f"\nSuccessfully saved SVD parameters to {save_path}")