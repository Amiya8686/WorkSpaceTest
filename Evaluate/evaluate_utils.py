import torch
import os
from tqdm import tqdm
from GetDataSet.datautills import get_loaders

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# 画图
def _save_similarity_plot(title, layer_labels, cosine_values, cka_values, mse_values, save_path, boundary_layers=None, group_lists=None):
    if plt is None:
        return

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    ax_top.plot(layer_labels, cosine_values, marker="o", label="Cosine")
    ax_top.plot(layer_labels, cka_values, marker="s", label="CKA")
    if boundary_layers:
        boundary_x = [layer_idx for layer_idx in layer_labels if layer_idx in boundary_layers]
        boundary_cos = [cosine_values[layer_idx] for layer_idx in boundary_x]
        boundary_cka = [cka_values[layer_idx] for layer_idx in boundary_x]
        ax_top.scatter(boundary_x, boundary_cos, facecolors="none", edgecolors="gray", s=90, linewidths=1.5, label="Boundary")
        ax_top.scatter(boundary_x, boundary_cka, facecolors="none", edgecolors="gray", s=90, linewidths=1.5)
    ax_top.set_ylabel("Similarity")
    ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")

    ax_bottom.plot(layer_labels, mse_values, marker="^", color="tab:red", label="MSE")
    if boundary_layers:
        boundary_x = [layer_idx for layer_idx in layer_labels if layer_idx in boundary_layers]
        boundary_mse = [mse_values[layer_idx] for layer_idx in boundary_x]
        ax_bottom.scatter(boundary_x, boundary_mse, facecolors="none", edgecolors="gray", s=90, linewidths=1.5, label="Boundary")
    ax_bottom.set_xlabel("Layer index")
    ax_bottom.set_ylabel("MSE")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="best")

    if layer_labels:
        tick_labels = [str(layer_idx) for layer_idx in layer_labels]
        dense_locator = FixedLocator(layer_labels)
        dense_formatter = FixedFormatter(tick_labels)
        ax_top.xaxis.set_major_locator(dense_locator)
        ax_top.xaxis.set_major_formatter(dense_formatter)
        ax_bottom.xaxis.set_major_locator(dense_locator)
        ax_bottom.xaxis.set_major_formatter(dense_formatter)
        ax_bottom.set_xlim(min(layer_labels), max(layer_labels))
        ax_top.set_xlim(min(layer_labels), max(layer_labels))
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_bottom.tick_params(axis="x", rotation=0)

    if group_lists:
        _annotate_group_layout(ax_top, group_lists, layer_labels)
        _annotate_group_layout(ax_bottom, group_lists, layer_labels)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _build_layer_labels(layer_num):
    return [i for i in range(layer_num - 1)]

def _build_group_boundary_info(group_lists):
    boundary_layers = set()
    group_text_lines = []
    for group_idx, group_layers in enumerate(group_lists):
        if not group_layers:
            continue
        group_text_lines.append(f"G{group_idx}: {group_layers}")
        boundary_layers.add(group_layers[-1])
    return boundary_layers, "\n".join(group_text_lines)

def _annotate_group_layout(ax, group_lists, layer_labels):
    boundary_layers, group_text = _build_group_boundary_info(group_lists)
    for boundary_layer in boundary_layers:
        if boundary_layer in layer_labels:
            ax.axvline(boundary_layer + 0.5, linestyle="--", color="gray", alpha=0.35)
    if group_text:
        ax.text(
            0.01,
            0.99,
            group_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

def _with_mode_suffix(save_path, use_commonkv):
    base_dir, file_name = os.path.split(save_path)
    stem, ext = os.path.splitext(file_name)
    suffix = "commonKV" if use_commonkv else "origin"
    return os.path.join(base_dir, f"{stem}_{suffix}{ext}")

def _reset_commonkv_rope_cache(model):
    if hasattr(model.model, "position_embeddings_table") and model.model.position_embeddings_table is not None:
        model.model.position_embeddings_table.reset()


# 测试
def eval_my_text(model,tokenizer,args,logger):
    """测试自定义文本"""
    logger.info("===========eval_my_text=============")
    question = "请告诉我糖醋排骨和青椒炒蛋怎么做?"
    messages = [
        {"role": "system", "content": "你是一个乐于帮助用户的中文助手。"},
        {"role": "user", "content": question},
    ]
    model.eval()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    generated_text = tokenizer.decode(generated_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
    logger.info(f"Question:{question}")
    logger.info(f"Response:{generated_text}")

def eval_hidden_state_similarity(model,tokenizer,args,logger):
    """测试hidden_state的层间相似性:cos,CKA,MSE"""
    logger.info("===========测试hidden_state的层间相似性============")
    def linear_cka(x, y):
        x = x.float()
        y = y.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        xy = torch.matmul(x.T, y)
        xx = torch.matmul(x.T, x)
        yy = torch.matmul(y.T, y)

        numerator = (xy * xy).sum()
        denominator = torch.sqrt((xx * xx).sum() * (yy * yy).sum())
        if denominator.item() == 0:
            return torch.zeros((), device=x.device, dtype=torch.float32)
        return numerator / denominator

    # 模型状态初始化
    model.eval()
    model.model.eval()
    model.set_save_hidden_state(True)
    use_cache_origin = model.config.use_cache
    model.config.use_cache = False


    # 加载测试集 并 提取样本
    _,testenc = get_loaders("wikitext2",model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0]
    total_tokens = input_ids.shape[0]
    sample_list = []
    for start in range(0, total_tokens, max_length):
        end = min(start + max_length, total_tokens)
        sample = input_ids[start:end]
        if sample.numel() > 0:
            sample_list.append(sample)

    layer_num = len(model.model.layers)
    metric_sums = {
        f"{i}-{i+1}": {"cosine": 0.0, "cka": 0.0, "mse": 0.0, "tokens": 0}
        for i in range(layer_num - 1)
    }
    layer_labels = _build_layer_labels(layer_num)

    # 用样本来进行预填充，并统计隐藏特征的层间cos，CKA，MSE
    with torch.no_grad():
        for sample in tqdm(sample_list, desc="hidden_state", total=len(sample_list)):
            sample = sample.unsqueeze(0).to(model.device)
            _reset_commonkv_rope_cache(model)
            _ = model(
                input_ids=sample,
                use_cache=False,
                past_key_values=None,
            )

            hidden_states = model.get_hidden_state()
            if hidden_states is None:
                continue
            sample_token_num = hidden_states[0].shape[1]

            for layer_idx in range(layer_num - 1):
                left_state = hidden_states[layer_idx].reshape(-1, hidden_states[layer_idx].shape[-1])
                right_state = hidden_states[layer_idx + 1].reshape(-1, hidden_states[layer_idx + 1].shape[-1])

                cosine_value = torch.nn.functional.cosine_similarity(left_state, right_state, dim=-1).mean()
                cka_value = linear_cka(left_state, right_state)
                mse_value = torch.nn.functional.mse_loss(left_state, right_state, reduction="mean")

                key = f"{layer_idx}-{layer_idx + 1}"
                metric_sums[key]["cosine"] += cosine_value.item() * sample_token_num
                metric_sums[key]["cka"] += cka_value.item() * sample_token_num
                metric_sums[key]["mse"] += mse_value.item() * sample_token_num
                metric_sums[key]["tokens"] += sample_token_num

    # 日志返回结果
    for layer_idx in range(layer_num - 1):
        key = f"{layer_idx}-{layer_idx + 1}"
        token_count = metric_sums[key]["tokens"]
        if token_count == 0:
            continue
        cos = metric_sums[key]["cosine"] / token_count
        cka = metric_sums[key]["cka"] / token_count
        mse = metric_sums[key]["mse"] / token_count
        logger.info(f"layer {key}: cosine:{cos:.6e},CKA:{cka:.6e},MSE:{mse:.6e}")

    if plt is not None:
        cosine_values = []
        cka_values = []
        mse_values = []
        for layer_idx in range(layer_num - 1):
            key = f"{layer_idx}-{layer_idx + 1}"
            token_count = metric_sums[key]["tokens"]
            if token_count == 0:
                cosine_values.append(0.0)
                cka_values.append(0.0)
                mse_values.append(0.0)
                continue
            cosine_values.append(metric_sums[key]["cosine"] / token_count)
            cka_values.append(metric_sums[key]["cka"] / token_count)
            mse_values.append(metric_sums[key]["mse"] / token_count)
        _save_similarity_plot(
            title="hidden_state similarity",
            layer_labels=layer_labels,
            cosine_values=cosine_values,
            cka_values=cka_values,
            mse_values=mse_values,
            save_path=_with_mode_suffix("logs/hidden_state_similarity.png", model.use_commonKV),
        )

    model.set_save_hidden_state(False)
    model.config.use_cache = use_cache_origin

def eval_KVcache_similarity(model,tokenizer,args,logger):
    """测试KVcache的层间相似性:cos,CKA,MSE"""
    logger.info("===========测试KVcache的层间相似性============")
    if model.use_commonKV:
        logger.info("测试KVcache的层间相似性,需要关闭commonKV")
        return
    
    # 初始化模型状态
    def linear_cka(x, y):
        x = x.float()
        y = y.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        xy = torch.matmul(x.T, y)
        xx = torch.matmul(x.T, x)
        yy = torch.matmul(y.T, y)

        numerator = (xy * xy).sum()
        denominator = torch.sqrt((xx * xx).sum() * (yy * yy).sum())
        if denominator.item() == 0:
            return torch.zeros((), device=x.device, dtype=torch.float32)
        return numerator / denominator

    def get_layer_kv(cache, layer_idx):
        if cache is None:
            return None, None
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        if isinstance(cache, (tuple, list)) and len(cache) > layer_idx:
            layer_cache = cache[layer_idx]
            if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                return layer_cache[0], layer_cache[1]
        if hasattr(cache, "to_legacy_cache"):
            legacy_cache = cache.to_legacy_cache()
            if len(legacy_cache) > layer_idx:
                return legacy_cache[layer_idx][0], legacy_cache[layer_idx][1]
        raise TypeError("Unsupported past_key_values format")

    model.eval()
    model.model.eval()
    use_cache_origin = model.config.use_cache
    model.config.use_cache = True


    # 获取样本
    _,testenc = get_loaders("wikitext2",model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0]
    total_tokens = input_ids.shape[0]
    sample_list = []
    for start in range(0, total_tokens, max_length):
        end = min(start + max_length, total_tokens)
        sample = input_ids[start:end]
        if sample.numel() > 0:
            sample_list.append(sample)

    layer_num = len(model.model.layers)
    metric_sums = {
        f"{i}-{i+1}": {
            "k": {"cosine": 0.0, "cka": 0.0, "mse": 0.0, "tokens": 0},
            "v": {"cosine": 0.0, "cka": 0.0, "mse": 0.0, "tokens": 0},
        }
        for i in range(layer_num - 1)
    }
    layer_labels = _build_layer_labels(layer_num)


    # 测试KVcache层间相似性
    try:
        with torch.no_grad():
            for sample in tqdm(sample_list, desc="KVcache", total=len(sample_list)):
                sample = sample.unsqueeze(0).to(model.device)
                outputs = model(
                    input_ids=sample,
                    use_cache=True,
                    past_key_values=None,
                )

                past_key_values = outputs.past_key_values
                if past_key_values is None:
                    continue

                first_k, first_v = get_layer_kv(past_key_values, 0)
                if first_k is None or first_v is None:
                    continue
                sample_token_num = first_k.shape[-2]

                for layer_idx in range(layer_num - 1):
                    left_k, left_v = get_layer_kv(past_key_values, layer_idx)
                    right_k, right_v = get_layer_kv(past_key_values, layer_idx + 1)

                    if left_k is None or right_k is None or left_v is None or right_v is None:
                        continue

                    head_num = left_k.shape[1]
                    k_cosine_sum = 0.0
                    k_cka_sum = 0.0
                    k_mse_sum = 0.0
                    v_cosine_sum = 0.0
                    v_cka_sum = 0.0
                    v_mse_sum = 0.0

                    for head_idx in range(head_num):
                        left_k_head = left_k[:, head_idx, :, :]
                        right_k_head = right_k[:, head_idx, :, :]
                        left_v_head = left_v[:, head_idx, :, :]
                        right_v_head = right_v[:, head_idx, :, :]

                        k_cosine_sum += torch.nn.functional.cosine_similarity(
                            left_k_head, right_k_head, dim=-1
                        ).mean().item()
                        k_cka_sum += linear_cka(
                            left_k_head.reshape(-1, left_k_head.shape[-1]),
                            right_k_head.reshape(-1, right_k_head.shape[-1]),
                        ).item()
                        k_mse_sum += torch.nn.functional.mse_loss(
                            left_k_head, right_k_head, reduction="mean"
                        ).item()

                        v_cosine_sum += torch.nn.functional.cosine_similarity(
                            left_v_head, right_v_head, dim=-1
                        ).mean().item()
                        v_cka_sum += linear_cka(
                            left_v_head.reshape(-1, left_v_head.shape[-1]),
                            right_v_head.reshape(-1, right_v_head.shape[-1]),
                        ).item()
                        v_mse_sum += torch.nn.functional.mse_loss(
                            left_v_head, right_v_head, reduction="mean"
                        ).item()

                    k_cosine = k_cosine_sum / head_num
                    k_cka = k_cka_sum / head_num
                    k_mse = k_mse_sum / head_num

                    v_cosine = v_cosine_sum / head_num
                    v_cka = v_cka_sum / head_num
                    v_mse = v_mse_sum / head_num

                    key = f"{layer_idx}-{layer_idx + 1}"
                    metric_sums[key]["k"]["cosine"] += k_cosine * sample_token_num
                    metric_sums[key]["k"]["cka"] += k_cka * sample_token_num
                    metric_sums[key]["k"]["mse"] += k_mse * sample_token_num
                    metric_sums[key]["k"]["tokens"] += sample_token_num

                    metric_sums[key]["v"]["cosine"] += v_cosine * sample_token_num
                    metric_sums[key]["v"]["cka"] += v_cka * sample_token_num
                    metric_sums[key]["v"]["mse"] += v_mse * sample_token_num
                    metric_sums[key]["v"]["tokens"] += sample_token_num

        # 日志输出
        for layer_idx in range(layer_num - 1):
            key = f"{layer_idx}-{layer_idx + 1}"

            k_tokens = metric_sums[key]["k"]["tokens"]
            if k_tokens > 0:
                k_cos = metric_sums[key]["k"]["cosine"] / k_tokens
                k_cka = metric_sums[key]["k"]["cka"] / k_tokens
                k_mse = metric_sums[key]["k"]["mse"] / k_tokens
                logger.info(f"layer {key} k :cosine{k_cos:.6e},CKA{k_cka:.6e},MSE:{k_mse:.6e}")

        if plt is not None:
            k_cosine_values = []
            k_cka_values = []
            k_mse_values = []
            for layer_idx in range(layer_num - 1):
                key = f"{layer_idx}-{layer_idx + 1}"
                k_tokens = metric_sums[key]["k"]["tokens"]
                if k_tokens == 0:
                    k_cosine_values.append(0.0)
                    k_cka_values.append(0.0)
                    k_mse_values.append(0.0)
                    continue
                k_cosine_values.append(metric_sums[key]["k"]["cosine"] / k_tokens)
                k_cka_values.append(metric_sums[key]["k"]["cka"] / k_tokens)
                k_mse_values.append(metric_sums[key]["k"]["mse"] / k_tokens)
            _save_similarity_plot(
                title="KVcache similarity - K",
                layer_labels=layer_labels,
                cosine_values=k_cosine_values,
                cka_values=k_cka_values,
                mse_values=k_mse_values,
                save_path=_with_mode_suffix("logs/kvcache_similarity_k.png", model.use_commonKV),
            )

        for layer_idx in range(layer_num - 1):
            key = f"{layer_idx}-{layer_idx + 1}"

            v_tokens = metric_sums[key]["v"]["tokens"]
            if v_tokens > 0:
                v_cos = metric_sums[key]["v"]["cosine"] / v_tokens
                v_cka = metric_sums[key]["v"]["cka"] / v_tokens
                v_mse = metric_sums[key]["v"]["mse"] / v_tokens
                logger.info(f"layer {key} v :cosine{v_cos:.6e},CKA{v_cka:.6e},MSE:{v_mse:.6e}")

        if plt is not None:
            v_cosine_values = []
            v_cka_values = []
            v_mse_values = []
            for layer_idx in range(layer_num - 1):
                key = f"{layer_idx}-{layer_idx + 1}"
                v_tokens = metric_sums[key]["v"]["tokens"]
                if v_tokens == 0:
                    v_cosine_values.append(0.0)
                    v_cka_values.append(0.0)
                    v_mse_values.append(0.0)
                    continue
                v_cosine_values.append(metric_sums[key]["v"]["cosine"] / v_tokens)
                v_cka_values.append(metric_sums[key]["v"]["cka"] / v_tokens)
                v_mse_values.append(metric_sums[key]["v"]["mse"] / v_tokens)
            _save_similarity_plot(
                title="KVcache similarity - V",
                layer_labels=layer_labels,
                cosine_values=v_cosine_values,
                cka_values=v_cka_values,
                mse_values=v_mse_values,
                save_path=_with_mode_suffix("logs/kvcache_similarity_v.png", model.use_commonKV),
            )
    finally:
        model.config.use_cache = use_cache_origin


    # 日志输出

def eval_latentKV_similarity(model,tokenizer,args,logger):
    """测试lantentKV的层间相似性:cos,CKA,MSE"""
    logger.info("===========测试latentKV的层间相似性============")
    if not model.use_commonKV:
        logger.info("测试LatentKV的层间相似性,需要启动commonKV")
        return


    # 初始化模型状态
    def linear_cka(x, y):
        x = x.float()
        y = y.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        xy = torch.matmul(x.T, y)
        xx = torch.matmul(x.T, x)
        yy = torch.matmul(y.T, y)

        numerator = (xy * xy).sum()
        denominator = torch.sqrt((xx * xx).sum() * (yy * yy).sum())
        if denominator.item() == 0:
            return torch.zeros((), device=x.device, dtype=torch.float32)
        return numerator / denominator

    model.eval()
    model.model.eval()
    use_cache_origin = model.config.use_cache
    model.config.use_cache = True

    group_lists = []
    boundary_layers = set()
    special_zero_layers = set()


    # 获取样本
    _,testenc = get_loaders("wikitext2",model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0]
    total_tokens = input_ids.shape[0]
    sample_list = []
    for start in range(0, total_tokens, max_length):
        end = min(start + max_length, total_tokens)
        sample = input_ids[start:end]
        if sample.numel() > 0:
            sample_list.append(sample)

    layer_num = len(model.model.layers)
    metric_sums = {
        f"{i}-{i+1}": {"cosine": 0.0, "cka": 0.0, "mse": 0.0, "tokens": 0}
        for i in range(layer_num - 1)
    }
    layer_labels = _build_layer_labels(layer_num)


    # 测试Latent层间相似性
    try:
        with torch.no_grad():
            for sample in tqdm(sample_list, desc="latentKV", total=len(sample_list)):
                sample = sample.unsqueeze(0).to(model.device)
                _reset_commonkv_rope_cache(model)
                outputs = model(
                    input_ids=sample,
                    use_cache=True,
                    past_key_values=None,
                )

                past_key_values = outputs.past_key_values
                if past_key_values is None or not hasattr(past_key_values, "commonKV_cache"):
                    continue

                common_kv_cache = past_key_values.commonKV_cache
                if not group_lists:
                    commonKV_meta_data = common_kv_cache.get_commonKV_meta_data()
                    group_lists = commonKV_meta_data.get("group_lists", []) if commonKV_meta_data is not None else []
                    if not group_lists:
                        continue
                    boundary_layers, _ = _build_group_boundary_info(group_lists)

                    layer_to_group = {}
                    for group_idx, group_layers in enumerate(group_lists):
                        for layer_idx in group_layers:
                            layer_to_group[layer_idx] = group_idx

                    for layer_idx in range(layer_num - 1):
                        left_group = layer_to_group.get(layer_idx)
                        right_group = layer_to_group.get(layer_idx + 1)
                        if left_group is None or right_group is None or left_group != right_group:
                            special_zero_layers.add(layer_idx)

                for layer_idx in range(layer_num - 1):
                    key = f"{layer_idx}-{layer_idx + 1}"

                    if layer_idx in special_zero_layers:
                        metric_sums[key]["cosine"] += 0.0
                        metric_sums[key]["cka"] += 0.0
                        metric_sums[key]["mse"] += 0.0
                        metric_sums[key]["tokens"] += 1
                        continue

                    left_latent_kv = common_kv_cache.get_latent_kv(layer_idx)
                    right_latent_kv = common_kv_cache.get_latent_kv(layer_idx + 1)

                    if left_latent_kv is None or right_latent_kv is None:
                        metric_sums[key]["tokens"] += 1
                        continue

                    pair_token_num = min(left_latent_kv.shape[1], right_latent_kv.shape[1])
                    if pair_token_num == 0:
                        metric_sums[key]["tokens"] += 1
                        continue

                    if left_latent_kv.shape[1] != right_latent_kv.shape[1]:
                        logger.warning(
                            f"latentKV length mismatch at layer {key}: "
                            f"left={left_latent_kv.shape[1]}, right={right_latent_kv.shape[1]}, "
                            f"use shared prefix {pair_token_num}"
                        )

                    left_state = left_latent_kv[:, :pair_token_num, :].reshape(-1, left_latent_kv.shape[-1])
                    right_state = right_latent_kv[:, :pair_token_num, :].reshape(-1, right_latent_kv.shape[-1])

                    cosine_value = torch.nn.functional.cosine_similarity(left_state, right_state, dim=-1).mean()
                    cka_value = linear_cka(left_state, right_state)
                    mse_value = torch.nn.functional.mse_loss(left_state, right_state, reduction="mean")

                    metric_sums[key]["cosine"] += cosine_value.item() * pair_token_num
                    metric_sums[key]["cka"] += cka_value.item() * pair_token_num
                    metric_sums[key]["mse"] += mse_value.item() * pair_token_num
                    metric_sums[key]["tokens"] += pair_token_num

        # 日志输出
        for layer_idx in range(layer_num - 1):
            key = f"{layer_idx}-{layer_idx + 1}"
            token_count = metric_sums[key]["tokens"]
            if token_count == 0:
                continue
            cos = metric_sums[key]["cosine"] / token_count
            cka = metric_sums[key]["cka"] / token_count
            mse = metric_sums[key]["mse"] / token_count
            logger.info(f"layer {key}: cosine:{cos:.6e},CKA:{cka:.6e},MSE:{mse:.6e}")

        if plt is not None:
            cosine_values = []
            cka_values = []
            mse_values = []
            for layer_idx in range(layer_num - 1):
                key = f"{layer_idx}-{layer_idx + 1}"
                token_count = metric_sums[key]["tokens"]
                if token_count == 0:
                    cosine_values.append(0.0)
                    cka_values.append(0.0)
                    mse_values.append(0.0)
                    continue
                cosine_values.append(metric_sums[key]["cosine"] / token_count)
                cka_values.append(metric_sums[key]["cka"] / token_count)
                mse_values.append(metric_sums[key]["mse"] / token_count)

            save_path = "logs/latentkv_similarity.png"
            _save_similarity_plot(
                title="latentKV similarity",
                layer_labels=layer_labels,
                cosine_values=cosine_values,
                cka_values=cka_values,
                mse_values=mse_values,
                save_path=_with_mode_suffix(save_path, model.use_commonKV),
                    boundary_layers=boundary_layers.union(special_zero_layers),
                group_lists=group_lists,
            )
    finally:
        model.config.use_cache = use_cache_origin

def eval_latentKV_modulus(model,tokenizer,args,logger):
    """对比lantentKV的模长"""
    logger.info("========modulus of latentKV========")
    if not model.use_commonKV:
        logger.info("测试LatentKV的模长,需要启动commonKV")
        return

    # 获取commonKV的元数据（可以从模型中获得）
    commonKV_meta_data = None
    group_lists = []

    # 获取测试样本：只需要一个长度为2048的样本进行前向传播即可
    _, testenc = get_loaders("wikitext2", model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0][:max_length]
    if input_ids.numel() == 0:
        return

    model.eval()
    model.model.eval()
    use_cache_origin = model.config.use_cache
    model.config.use_cache = True

    # 进行一次前向传播（预先填充）
    try:
        with torch.no_grad():
            sample = input_ids.unsqueeze(0).to(model.device)
            _reset_commonkv_rope_cache(model)
            outputs = model(
                input_ids=sample,
                use_cache=True,
                past_key_values=None,
            )

            past_key_values = outputs.past_key_values
            if past_key_values is None or not hasattr(past_key_values, "commonKV_cache"):
                return

            commonKV_cache = past_key_values.commonKV_cache
            commonKV_meta_data = commonKV_cache.get_commonKV_meta_data()
            if commonKV_meta_data is None:
                return

            group_lists = commonKV_meta_data.get("group_lists", [])
            if not group_lists:
                return

            # 输入前20个token的latentKV的模长
            for group_layers in group_lists:
                if not group_layers:
                    continue

                logger.info(f"====group {group_layers}====")

                token_num = 0
                for layer_idx in group_layers:
                    latent_kv = commonKV_cache.get_latent_kv(layer_idx)
                    if latent_kv is None:
                        continue
                    token_num = max(token_num, min(latent_kv.shape[1], 20))

                for token_idx in range(token_num):
                    logger.info(f"token {token_idx}:")
                    for layer_idx in group_layers:
                        latent_kv = commonKV_cache.get_latent_kv(layer_idx)
                        if latent_kv is None or token_idx >= latent_kv.shape[1]:
                            continue

                        modulus = torch.norm(latent_kv[0, token_idx, :], p=2).item()
                        logger.info(f"layer{layer_idx}: {modulus:.6g}")
    finally:
        model.config.use_cache = use_cache_origin

def eval_outliner_cosine_similarity(model,tokenizer,args,logger):
    """获取latentKV层间余弦相似性的离群值"""
    logger.info("======== The cosine similarity of the least similar latentKV ======== ")
    if not model.use_commonKV:
        logger.info("测试LatentKV的层间余弦相似性离群值,需要启动commonKV")
        return
    
    #获取获取测试样本:2048长度
    _, testenc = get_loaders("wikitext2", model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0][:max_length]
    if input_ids.numel() == 0:
        return

    
    #获取commomKV元数据
    model.eval()
    model.model.eval()
    use_cache_origin = model.config.use_cache
    model.config.use_cache = True


    #前向传播（预填充）
    try:
        with torch.no_grad():
            sample = input_ids.unsqueeze(0).to(model.device)
            _reset_commonkv_rope_cache(model)
            outputs = model(
                input_ids=sample,
                use_cache=True,
                past_key_values=None,
            )

            past_key_values = outputs.past_key_values
            if past_key_values is None or not hasattr(past_key_values, "commonKV_cache"):
                return

            commonKV_cache = past_key_values.commonKV_cache
            commonKV_meta_data = commonKV_cache.get_commonKV_meta_data()
            if commonKV_meta_data is None:
                return

            group_lists = commonKV_meta_data.get("group_lists", [])
            if not group_lists:
                return

            def _get_cosine_values(left_latent_kv, right_latent_kv):
                pair_token_num = min(left_latent_kv.shape[1], right_latent_kv.shape[1])
                if pair_token_num == 0:
                    return None

                left_state = left_latent_kv[:, :pair_token_num, :].reshape(-1, left_latent_kv.shape[-1])
                right_state = right_latent_kv[:, :pair_token_num, :].reshape(-1, right_latent_kv.shape[-1])
                return torch.nn.functional.cosine_similarity(left_state, right_state, dim=-1)

            #从past_key_values中拿到commonKV_cache,并输出离群余弦相似性token(10个最大值，10个最小值，均值)
            for group_layers in group_lists:
                if not group_layers or len(group_layers) < 2:
                    continue

                for layer_offset in range(len(group_layers) - 1):
                    left_layer_idx = group_layers[layer_offset]
                    right_layer_idx = group_layers[layer_offset + 1]

                    left_latent_kv = commonKV_cache.get_latent_kv(left_layer_idx)
                    right_latent_kv = commonKV_cache.get_latent_kv(right_layer_idx)
                    if left_latent_kv is None or right_latent_kv is None:
                        continue

                    cosine_values = _get_cosine_values(left_latent_kv, right_latent_kv)
                    if cosine_values is None or cosine_values.numel() == 0:
                        continue

                    k_num = min(10, cosine_values.numel())
                    sorted_values, sorted_indices = torch.sort(cosine_values)

                    logger.info(f"====layer {left_layer_idx}-{right_layer_idx}====")
                    logger.info("least similar:")
                    for idx in range(k_num):
                        token_idx = sorted_indices[idx].item()
                        logger.info(f"token{token_idx}: cosim: {sorted_values[idx].item():.6g}")

                    logger.info("most similar:")
                    for idx in range(k_num):
                        token_idx = sorted_indices[-(idx + 1)].item()
                        logger.info(f"token{token_idx}: cosim: {sorted_values[-(idx + 1)].item():.6g}")

                    logger.info("mean:")
                    logger.info(f"cosim: {cosine_values.mean().item():.6g}")
    finally:
        model.config.use_cache = use_cache_origin

def eval_latentKV_similarity_to_mean_value(model,tokenizer,args,logger):
    """测试lantenKV相对层间单位方向向量平均插值的余弦相似性"""
    logger.info("======== eval_latentKV_cos_similarity_to_mean_value ========")
    if not model.use_commonKV:
        logger.info("测试LatentKV相对层间单位向量平均插值的余弦相似性,需要启动commonKV")
        return
    
    #获取样本：长度为2048
    _, testenc = get_loaders("wikitext2", model=args.model)
    max_length = min(model.config.max_position_embeddings, 2048)
    input_ids = testenc.input_ids[0]
    total_tokens = input_ids.shape[0]
    sample_list = []
    for start in range(0, total_tokens, max_length):
        end = min(start + max_length, total_tokens)
        sample = input_ids[start:end]
        if sample.numel() > 0:
            sample_list.append(sample)

    if len(sample_list) == 0:
        return

    layer_num = len(model.model.layers)
    layer_labels = _build_layer_labels(layer_num)

    #记录模型状态
    model.eval()
    model.model.eval()
    use_cache_origin = model.config.use_cache
    model.config.use_cache = True

    group_lists = []
    group_cosine_sums = []
    group_sample_counts = []
    boundary_layers = set()

    #前向传播，拿到past_key_values中的commonKV_cache
    try:
        with torch.no_grad():
            for sample in tqdm(sample_list, desc="latentKV_mean", total=len(sample_list)):
                sample = sample.unsqueeze(0).to(model.device)
                _reset_commonkv_rope_cache(model)
                outputs = model(
                    input_ids=sample,
                    use_cache=True,
                    past_key_values=None,
                )

                past_key_values = outputs.past_key_values
                if past_key_values is None or not hasattr(past_key_values, "commonKV_cache"):
                    continue

                common_kv_cache = past_key_values.commonKV_cache
                if not group_lists:
                    commonKV_meta_data = common_kv_cache.get_commonKV_meta_data()
                    group_lists = commonKV_meta_data.get("group_lists", []) if commonKV_meta_data is not None else []
                    if not group_lists:
                        continue

                    boundary_layers, _ = _build_group_boundary_info(group_lists)

                    group_cosine_sums = [{layer_idx: 0.0 for layer_idx in group_layers} for group_layers in group_lists]
                    group_sample_counts = [{layer_idx: 0 for layer_idx in group_layers} for group_layers in group_lists]

                for group_idx, group_layers in enumerate(group_lists):
                    if not group_layers:
                        continue

                    # 先获得组内层 latentKV 的 token 级单位向量平均插值
                    layer_unit_kvs = []
                    group_token_num = None
                    for layer_idx in group_layers:
                        latent_kv = common_kv_cache.get_latent_kv(layer_idx)
                        if latent_kv is None:
                            group_token_num = 0
                            break

                        latent_kv = latent_kv.float()
                        latent_kv_unit = latent_kv / latent_kv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        layer_unit_kvs.append((layer_idx, latent_kv_unit))

                        current_token_num = latent_kv_unit.shape[1]
                        if group_token_num is None:
                            group_token_num = current_token_num
                        else:
                            group_token_num = min(group_token_num, current_token_num)

                    if group_token_num is None or group_token_num == 0 or len(layer_unit_kvs) == 0:
                        continue

                    group_mean = torch.stack(
                        [latent_kv_unit[:, :group_token_num, :] for _, latent_kv_unit in layer_unit_kvs],
                        dim=0,
                    ).mean(dim=0)

                    # 计算每一层相对这个均值的余弦相似性，并对 token 取均值
                    for layer_idx, latent_kv_unit in layer_unit_kvs:
                        token_cosine = torch.nn.functional.cosine_similarity(
                            latent_kv_unit[:, :group_token_num, :],
                            group_mean,
                            dim=-1,
                        )
                        layer_cosine = token_cosine.mean().item()
                        group_cosine_sums[group_idx][layer_idx] += layer_cosine
                        group_sample_counts[group_idx][layer_idx] += 1

        # 输出
        for group_idx, group_layers in enumerate(group_lists):
            if not group_layers:
                continue

            group_text = "[" + ",".join(str(layer_idx) for layer_idx in group_layers) + "]"
            logger.info(f"====group {group_text}====")
            for layer_idx in group_layers:
                sample_count = group_sample_counts[group_idx][layer_idx]
                if sample_count == 0:
                    continue
                layer_cosine = group_cosine_sums[group_idx][layer_idx] / sample_count
                logger.info(f"layer{layer_idx}: {layer_cosine:.6g}")

        if plt is not None and group_lists:
            cosine_values = [0.0 for _ in layer_labels]
            cka_values = [0.0 for _ in layer_labels]
            mse_values = [0.0 for _ in layer_labels]

            for group_idx, group_layers in enumerate(group_lists):
                for layer_idx in group_layers:
                    sample_count = group_sample_counts[group_idx][layer_idx]
                    if sample_count == 0:
                        continue
                    cosine_values[layer_idx] = group_cosine_sums[group_idx][layer_idx] / sample_count

            _save_similarity_plot(
                title="latentKV mean similarity",
                layer_labels=layer_labels,
                cosine_values=cosine_values,
                cka_values=cka_values,
                mse_values=mse_values,
                save_path=_with_mode_suffix("logs/latentKV_mean_similarity.png", model.use_commonKV),
                boundary_layers=boundary_layers,
                group_lists=group_lists,
            )
    finally:
        model.config.use_cache = use_cache_origin