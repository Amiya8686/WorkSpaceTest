import gc
import json
import os

import torch
from safetensors.torch import save_file
from transformers.masking_utils import create_causal_mask

from GetDataSet.datautills import get_loaders
from myLlama_commonKV.modeling_llama import AdaptiveRoPECache


def train_let_parameters(model, commonKV_meta_data, let_meta_data, args, logger):
    """训练 let 参数。"""
    logger.info("======== train let factor ========")
    if not model.use_commonKV:
        logger.info("Train let factor need to turn on commonKV")
        return

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    model.model.eval()

    nsamples = int(getattr(args, "nsamples", 20))
    epoches = int(getattr(args, "epoches", getattr(args, "epochs", 1)))
    loss_scale_factor = float(getattr(args, "loss_scale_factor", 1.0))
    let_lr = float(getattr(args, "let_lr", 1e-2))
    max_seq_len = min(model.config.max_position_embeddings, int(getattr(args, "train_max_seq_len", 2048)))

    logger.info(f"nsamples {nsamples}; epoches {epoches}; loss_scale_factor:{loss_scale_factor:g}")

    data_loader, _ = get_loaders("wikitext2", model=args.model)
    train_batches = []
    for batch in data_loader:
        sample = batch[0] if isinstance(batch, (list, tuple)) else batch
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        if sample.dim() > 2:
            sample = sample.view(sample.shape[0], -1)
        sample = sample[:, :max_seq_len].contiguous()
        if sample.numel() == 0:
            continue
        train_batches.append(sample)
        if len(train_batches) >= nsamples:
            break

    if len(train_batches) == 0:
        logger.info("No training samples found")
        model.config.use_cache = use_cache
        return

    common_groups = commonKV_meta_data.get("group_lists", []) if commonKV_meta_data is not None else []
    let_layers = set()
    if let_meta_data is not None:
        let_layers = set(let_meta_data.get("let_layers", let_meta_data.get("let_layers:", [])))

    train_groups = []
    for group_layers in common_groups:
        if not group_layers:
            continue
        if let_layers:
            group_layers = [layer_idx for layer_idx in group_layers if layer_idx in let_layers]
        if len(group_layers) < 2:
            continue
        train_groups.append(group_layers)

    if len(train_groups) == 0:
        logger.info("No merged groups need LET training")
        model.config.use_cache = use_cache
        return

    device = model.device

    def _normalize_batch(batch):
        if isinstance(batch, (list, tuple)):
            sample = batch[0]
        else:
            sample = batch
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        if sample.dim() > 2:
            sample = sample.view(sample.shape[0], -1)
        return sample[:, :max_seq_len].contiguous()

    def _build_model_inputs(sample_ids):
        sample_ids = sample_ids.to(device)
        hidden_states = model.model.embed_tokens(sample_ids)

        cache_position = torch.arange(0, hidden_states.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)

        attention_mask = create_causal_mask(
            config=model.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        if model.model.use_commonKV:
            if model.model.position_embeddings_table is None:
                model.model.position_embeddings_table = AdaptiveRoPECache(
                    head_dim=position_embeddings[0].shape[-1],
                    device=position_embeddings[0].device,
                    dtype=position_embeddings[0].dtype,
                )
            else:
                model.model.position_embeddings_table.reset()
            position_embeddings = model.model.position_embeddings_table.update(
                position_embeddings[0],
                position_embeddings[1],
                cache_position,
            )

        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings

    def _forward_prefix(hidden_states, attention_mask, position_ids, cache_position, position_embeddings, end_layer_idx):
        with torch.no_grad():
            for layer_idx in range(end_layer_idx):
                layer = model.model.layers[layer_idx].to(device)
                layer.eval()
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                model.model.layers[layer_idx] = layer.cpu()
        return hidden_states

    def _build_input_cache():
        input_cache = []
        for batch in train_batches:
            sample_ids = _normalize_batch(batch)
            hidden_states, attention_mask, position_ids, cache_position, position_embeddings = _build_model_inputs(sample_ids)
            hidden_states = hidden_states.detach()
            attention_mask = attention_mask.detach() if attention_mask is not None else None
            position_ids = position_ids.detach() if position_ids is not None else None
            cache_position = cache_position.detach() if cache_position is not None else None
            if position_embeddings is not None:
                position_embeddings = tuple(item.detach() for item in position_embeddings)
            input_cache.append((hidden_states, attention_mask, position_ids, cache_position, position_embeddings))
        return input_cache

    def _build_next_layer_cache(layer_idx, current_cache):
        next_cache = []
        with torch.no_grad():
            for hidden_states, attention_mask, position_ids, cache_position, position_embeddings in current_cache:
                layer = model.model.layers[layer_idx].to(device)
                layer.eval()
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                model.model.layers[layer_idx] = layer.cpu()
                hidden_states = hidden_states.detach()
                attention_mask = attention_mask.detach() if attention_mask is not None else None
                position_ids = position_ids.detach() if position_ids is not None else None
                cache_position = cache_position.detach() if cache_position is not None else None
                if position_embeddings is not None:
                    position_embeddings = tuple(item.detach() for item in position_embeddings)
                next_cache.append((hidden_states, attention_mask, position_ids, cache_position, position_embeddings))
        return next_cache

    def _build_group_mean_cache(layer_input_cache, group_layers):
        group_mean_cache = []
        for hidden_states, attention_mask, position_ids, cache_position, position_embeddings in layer_input_cache:
            group_mean = _compute_group_mean(
                hidden_states,
                attention_mask,
                position_ids,
                cache_position,
                position_embeddings,
                group_layers,
            )
            if group_mean is not None:
                group_mean = group_mean.detach()
            group_mean_cache.append(group_mean)
        return group_mean_cache

    def _compute_group_mean(sample_hidden_states, attention_mask, position_ids, cache_position, position_embeddings, group_layers):
        probe_hidden = sample_hidden_states.detach()
        latent_kv_unit_list = []

        with torch.no_grad():
            for layer_idx in group_layers:
                layer = model.model.layers[layer_idx].to(device)
                layer.eval()
                layer.self_attn.set_save_lkv(True)
                probe_hidden = layer(
                    hidden_states=probe_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                latent_kv = layer.self_attn.get_lkv()
                layer.self_attn.set_save_lkv(False)
                if latent_kv is not None:
                    latent_kv = latent_kv.to(device).float()
                    latent_kv = latent_kv / latent_kv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    latent_kv_unit_list.append(latent_kv)
                model.model.layers[layer_idx] = layer.cpu()

        if len(latent_kv_unit_list) == 0:
            return None

        return torch.stack(latent_kv_unit_list, dim=0).mean(dim=0)

    def _cosine_mean_loss(lkv, lkv_mean):
        lkv = lkv.to(lkv_mean.device).float()
        lkv_mean = lkv_mean.float()
        token_cos = torch.nn.functional.cosine_similarity(lkv, lkv_mean, dim=-1)
        mean_cos = token_cos.mean()
        target = torch.ones_like(mean_cos)
        loss = torch.nn.functional.mse_loss(mean_cos, target) * loss_scale_factor
        return loss, mean_cos

    def _format_max_memory_gb():
        if device.type != "cuda" or not torch.cuda.is_available():
            return "0G"
        max_memory_bytes = torch.cuda.max_memory_allocated(device)
        max_memory_gb = max_memory_bytes / (1024 ** 3)
        return f"{max_memory_gb:.2f}G"

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm = model.model.norm.to(device)

    try:
        for group_layers in train_groups:
            logger.info(f"===group {group_layers}===")

            layer_input_cache = _build_input_cache()
            group_mean_cache = _build_group_mean_cache(layer_input_cache, group_layers)

            for layer_idx in group_layers:
                layer = model.model.layers[layer_idx].to(device)
                if layer.self_attn.let_factor is None:
                    logger.warning(f"layer {layer_idx} has no let_factor, skip")
                    model.model.layers[layer_idx] = layer.cpu()
                    continue

                optimizer = torch.optim.AdamW([layer.self_attn.let_factor], lr=let_lr, weight_decay=0.0)

                for epoch_idx in range(epoches):
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device)
                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss_sum = 0.0
                    epoch_scaled_loss_sum = 0.0
                    epoch_cosim_sum = 0.0
                    sample_count = 0

                    for sample_idx, (hidden_states, attention_mask, position_ids, cache_position, position_embeddings) in enumerate(layer_input_cache):
                        group_mean = group_mean_cache[sample_idx]
                        if group_mean is None:
                            continue

                        layer = model.model.layers[layer_idx].to(device)
                        layer.eval()

                        layer.self_attn.set_save_lkv_after_let(True)
                        _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        current_lkv = layer.self_attn.get_lkv_after_let()
                        layer.self_attn.set_save_lkv_after_let(False)

                        if current_lkv is None:
                            continue

                        loss, mean_cos = _cosine_mean_loss(current_lkv, group_mean)
                        loss.backward()
                        epoch_loss_sum += loss.item() / loss_scale_factor
                        epoch_scaled_loss_sum += loss.item()
                        epoch_cosim_sum += mean_cos.item()
                        sample_count += 1

                    optimizer.step()

                    if sample_count > 0:
                        logger.info(
                            f"layer {layer_idx}: epoch {epoch_idx}: loss:{epoch_loss_sum / sample_count:.6g};{epoch_scaled_loss_sum / sample_count:.6g}, cosim:{epoch_cosim_sum / sample_count:.6g}, max_memory:{_format_max_memory_gb()}"
                        )

                if layer_idx != group_layers[-1]:
                    layer_input_cache = _build_next_layer_cache(layer_idx, layer_input_cache)

                model.model.layers[layer_idx] = layer.cpu()
                del optimizer
                torch.cuda.empty_cache()

            for layer_idx in group_layers:
                model.model.layers[layer_idx] = model.model.layers[layer_idx].cpu()
            torch.cuda.empty_cache()

    finally:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, "position_embeddings_table"):
            model.model.position_embeddings_table = None
        gc.collect()
        torch.cuda.empty_cache()
        model.config.use_cache = use_cache

    let_parameters_to_save = {}
    let_layers_to_save = []
    for group_layers in train_groups:
        for layer_idx in group_layers:
            let_factor = model.model.layers[layer_idx].self_attn.let_factor
            if let_factor is None:
                continue
            let_parameters_to_save[f"let_l{layer_idx}"] = let_factor.detach().to(torch.float16).cpu().clone()
            let_layers_to_save.append(layer_idx)

    let_layers_to_save = sorted(set(let_layers_to_save))
    metadata = {
        "model": str(commonKV_meta_data.get("model", "") if commonKV_meta_data is not None else ""),
        "layers_num": str(commonKV_meta_data.get("layers_num", 0) if commonKV_meta_data is not None else 0),
        "let_layers_num": str(len(let_layers_to_save)),
        "let_layers": json.dumps(let_layers_to_save),
        "let_layers:": json.dumps(let_layers_to_save),
    }

    save_root = args.let_save_dir if getattr(args, "let_save_dir", None) else os.path.dirname(getattr(args, "let_parameters", ""))
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, "let_parameters.safetensors")
    save_file(let_parameters_to_save, save_path, metadata=metadata)
    logger.info(f"Saved trained let parameters to {save_path}")




