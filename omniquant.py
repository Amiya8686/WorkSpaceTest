import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}

# 添加新模块
def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    #Edit:训练模式关闭MiniCache
    if "llama" in args.net.lower() and args.use_llama_minicache:
        lm.model.model.config.minicache_config["active"] = False

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False


    #针对不同模型，准备引用变量
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        # 这里的"qkv","out","fc1"分别对应原文中流程图的，除了qk那里的let变换
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    # 将第一个中间层移动到GPU；控制amp(用fp16还是fp32训练)
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0}

    #捕获器：要捕获哪一层的输入，就用Catcher替换它：然后存到inps中
    #Cache中存元数据：比如sample计数器，掩码矩阵
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            # 利用抛出异常来打断前向传播：训练第一层时，还不需要跑后面的层
            raise ValueError
    
    # catch the first layer input
    # 捕捉第一层的输入（embed后的输出）
    # 实现方式：用catcher来替换第一个中间层layer,捕获其输入
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                # 一个batch就是一个样本: batch[0]样本本身,batch[1]标签
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    # 已经捕获，将layers换回来；并将layers的参与移回cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    # fp_inps_2 = layer(quant_inps) 【开启aug_loss:quant_inps 对齐 fp_inps】：不需要学习处理量化噪声累计，难度更低
    # fp_inps   = layer(fp_inps)    【不开启aug_loss:quant_inps 对齐 fp_inps_2】:需要学习处理量化噪声累计，难度更高
    # 后面代码实际上：启动aug_loss后，会把fp_inps_2对齐的loss加上去
    
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input

    
    # 掩码矩阵批次维度对齐
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None


    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None


    # 重新开始训练，还是对已经训练好的进行修正
    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    # 获取开始层间相似性约束的层
    #Edit:训练模式关闭MiniCache


    begin_layer = args.begin_layer
    end_layer = args.end_layer
    if(not args.layer_similarity):
        begin_layer = len(layers)
        end_layer=begin_layer

    # 模型前半部分，正常处理
    for i in range(0,begin_layer):
        logger.info(f"=== Start quantize front layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model：获得待对齐的张量（关闭量化）
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        

        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # 对llama模型不使用平移因子（它原本的特征均值就倾向于0）
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            # 使用smoothquant初始化scale,outliner-suppression来初始shift
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))

        # 若resume,则加载OmniQuant参数            
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 开始训练
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        # OmniQuant调整之后再量化
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        # quant_out = qlayer(quant_inps[index:index+args.batch_size], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        # # 核心：损失函数，我们要修改这里
                        # loss = loss_func(fp_inps[index:index+args.batch_size], quant_out)
                        # if args.aug_loss:
                        #     loss += loss_func(fp_inps_2[index:index+args.batch_size], quant_out)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size], attention_mask=attention_mask_batch,position_ids=position_ids)
                        loss = loss_func(fp_inps[index:index+args.batch_size].view(-1, fp_inps.shape[-1]), 
                                         quant_out.view(-1, quant_out.shape[-1]))
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size].view(-1, fp_inps_2.shape[-1]), 
                                              quant_out.view(-1, quant_out.shape[-1]))
                        else:
                            # 让fp输出的loss翻倍
                            loss += loss
                    
                    # 如果出现了NaN(Not a number:一般是溢出了)就在终端提示并停止训练
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    # 统计量
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        

        # real smooth and quantization:将得到的OmniQuant参数，进行真正的算子融合
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            # 更新quant_inps，并保存这一层训练得到的OmniQuant参数
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            # 调回fp16
            qlayer.half() 
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            # 非训练模型，直接移动会cpu（逐层推理和训练）
            # 调回fp16
            qlayer.half() 
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        if args.real_quant:
            # 没有实现激活权重同时量化时的，量化参数算子融合代码，所以只能pack仅权重量化的参数
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module
                
        del layer
        torch.cuda.empty_cache()


    # 模型中间部分部分，加入层间相似性约束
    if args.layer_similarity:
        # 中间解码器输出（两个解码器为一组）
        fp_inps_inter = copy.deepcopy(fp_inps)
        fp_inps_2_inter = copy.deepcopy(fp_inps_2)
    for i in range(begin_layer,end_layer,2):
        logger.info(f"=== Start quantize mid layer {i} and {i+1} ===")
        # 初始化两层
        layer1 = layers[i].to(dev)
        layer2 = layers[i+1].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer1 = copy.deepcopy(layer1)
            qlayer2 = copy.deepcopy(layer2)
            for name, module in qlayer1.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer1, quantlinear)    
            for name, module in qlayer2.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer2, quantlinear)    
        else:
            #Edited:增加save_kv参数:存储KV(方便计算kv相似度loss)，关闭MiniCache(训练时不开启MiniCache)
            save_kv = args.layer_similarity
            qlayer1 = DecoderLayer(lm.model.config, layer1, args)
            qlayer2 = DecoderLayer(lm.model.config, layer2, args)
        qlayer1 = qlayer1.to(dev)
        qlayer2 = qlayer2.to(dev)
        # obtain output of full-precision model：获得待对齐的张量（关闭量化）
        set_quant_state(qlayer1, weight_quant=False, act_quant=False)
        set_quant_state(qlayer2, weight_quant=False, act_quant=False)
        
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps_inter[j] = qlayer1(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        fp_inps[j] = qlayer2(fp_inps_inter[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2_inter[j] = qlayer1(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                            fp_inps_2[j] = qlayer2(fp_inps_2_inter[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        


        # init smooth parameters
        set_quant_state(qlayer1, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        set_quant_state(qlayer2, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer1.let = args.let
        qlayer2.let = args.let
        use_shift = True 
        # 对llama模型不使用平移因子（它原本的特征均值就倾向于0）
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer1.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer1.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            qlayer2.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer2.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer1.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            # 使用smoothquant初始化scale,outliner-suppression来初始shift
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer1.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer1.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
            for name,module in qlayer2.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            # 使用smoothquant初始化scale,outliner-suppression来初始shift
                            act = act_scales[f"{layer_name_prefix}.{i+1}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i+1}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer2.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer2.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))

        # 若resume,则加载OmniQuant参数            
        if args.resume:
            qlayer1.load_state_dict(omni_parameters[i], strict=False)
            qlayer2.load_state_dict(omni_parameters[i+1], strict=False)
        

        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer1.float()      # required for AMP training
                qlayer2.float()
            # create optimizer
            let_params = list(let_parameters(qlayer1, use_shift)) + list(let_parameters(qlayer2, use_shift))
            lwc_params = list(lwc_parameters(qlayer1)) + list(lwc_parameters(qlayer2))
            optimizer = torch.optim.AdamW([
                {"params": let_params, "lr": args.let_lr}, 
                {"params": lwc_params, "lr": args.lwc_lr}
            ], weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()



            # 启动qkv和attn_score的存储
            qlayer1.set_saveqkv_state(True)
            qlayer2.set_saveqkv_state(True)
            if args.qk_product_loss:
                qlayer1.set_save_attn_score_state(True)
                qlayer2.set_save_attn_score_state(True)

            
            # 开始训练
            for epochs in range(args.epochs * 2):
                loss_list_1 = []
                loss_list_2 = []
                loss_list_sim_k = []
                loss_list_sim_v = []
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        # OmniQuant调整之后再量化
                        smooth_and_quant_temporary(qlayer1, args, is_llama)
                        smooth_and_quant_temporary(qlayer2, args, is_llama)

                        # 奇数层特征对齐
                        # 避免批次维度报错
                        curr_quant_inp = quant_inps[index:index+args.batch_size]
                        if curr_quant_inp.dim() == 2:
                            curr_quant_inp = curr_quant_inp.unsqueeze(0)
                        
                        quant_out_1 = qlayer1(curr_quant_inp, attention_mask=attention_mask_batch,position_ids=position_ids)
                        loss_1 = loss_func(fp_inps_inter[index:index+args.batch_size].view(-1, fp_inps_inter.shape[-1]), quant_out_1.view(-1, quant_out_1.shape[-1]))
                        if args.aug_loss:
                            loss_1 += loss_func(fp_inps_2_inter[index:index+args.batch_size].view(-1, fp_inps_2_inter.shape[-1]), quant_out_1.view(-1, quant_out_1.shape[-1]))
                        else:
                            # 让fp输出的loss翻倍
                            loss_1 += loss_1    
                        # 偶数层特征对齐

                        curr_quant_inp_2 = quant_out_1
                        if curr_quant_inp_2.dim() == 2:
                            curr_quant_inp_2 = curr_quant_inp_2.unsqueeze(0)

                        quant_out_2 = qlayer2(curr_quant_inp_2, attention_mask=attention_mask_batch,position_ids=position_ids)
                        loss_2 = loss_func(fp_inps[index:index+args.batch_size].view(-1, fp_inps.shape[-1]), quant_out_2.view(-1, quant_out_2.shape[-1]))
                        if args.aug_loss:
                            loss_2 += loss_func(fp_inps_2[index:index+args.batch_size].view(-1, fp_inps_2.shape[-1]), quant_out_2.view(-1, quant_out_2.shape[-1]))
                        else:
                            loss_2 += loss_2
                        
                        # 维护KV层间相似性
                        q1,k1,v1,q2,k2,v2, = qlayer1.q, qlayer1.k,qlayer1.v, qlayer2.q,qlayer2.k,qlayer2.v
                        qlayer1.q = qlayer1.k = qlayer1.v = qlayer2.q = qlayer2.k = qlayer2.v = None

                        # #原本通过余弦相似度来约束的代码
                        # cos = torch.nn.CosineSimilarity(dim=-1)
                        # sim_k = cos(k1, k2)
                        # sim_v = cos(v1, v2)
                        # # 惩罚相似度过小的部分
                        # loss_sim_k = loss_func(sim_k, torch.full_like(sim_k, 0.8))
                        # loss_sim_v = loss_func(sim_v, torch.full_like(sim_v, 0.8))
                        # # #最终loss

                        #现在改用对比合并前后的KVcache来构造Loss
                        a,b,c,d = args.sim_a,args.sim_b,args.sim_c,args.sim_d
                        mc_alpha,mc_gamma = args.minicache_alpha,args.minicache_gamma
                        ## copilor修改开始
                        # 1. 计算k1和k2的模长和单位方向向量
                        norm_k1 = torch.norm(k1, p=2, dim=-1, keepdim=True) + 1e-6
                        norm_k2 = torch.norm(k2, p=2, dim=-1, keepdim=True) + 1e-6
                        dir_k1 = k1 / norm_k1
                        dir_k2 = k2 / norm_k2
                        
                        # 2. 计算点积和角距离
                        dot_product_k = torch.sum(dir_k1 * dir_k2, dim=-1).clamp(-1.0, 1.0)
                        dk = torch.acos(dot_product_k) * (1 / torch.pi)
                        
                        # 3.计算离群值判断阈值与离群矩阵
                        dk_max = torch.max(dk, dim=-1).values.detach()  # [B, H, S] -> [B, H]
                        dk_min = torch.min(dk, dim=-1).values.detach()
                        dk_threshold = dk_min + mc_gamma * (dk_max - dk_min)
                        is_outlier_k = dk > dk_threshold.unsqueeze(-1)
                        
                        # 4. 获得合并，然后解压的k1,k2
                        k_mc_dir = (1 - mc_alpha) * dir_k1 + mc_alpha * dir_k2
                        k_mc_dir = k_mc_dir / (torch.norm(k_mc_dir, p=2, dim=-1, keepdim=True) + 1e-6)
                        k1_mc = k_mc_dir * norm_k1
                        k2_mc = k_mc_dir * norm_k2
                        
                        # 5. 计算loss_sim_k，屏蔽离群值loss计算
                        outlier_mask_k = ~is_outlier_k 

                        # 看是测点积误差还是MSE误差
                        if args.qk_product_loss:
                            attn_score1,attn_score2 = qlayer1.attn_score, qlayer2.attn_score
                            qlayer1.attn_score = qlayer2.attn_score = None
                            attn1 = qlayer1.self_attn
                            attn2 = qlayer2.self_attn
                            
                            #量化->算注意力权重->掩码->掩码注意力权重（不用用注意力分数，softmax之后Loss太小了）
                            q1 = attn1.qkt_matmul.quant_x1(q1)
                            k1_mc = attn1.qkt_matmul.quant_x2(k1_mc)
                            attn_score_mc1 = attn1.qkt_matmul(q1, k1_mc.transpose(2, 3)) / math.sqrt(attn1.head_dim)
                            attn_score_mc1 = attn_score_mc1+ attention_mask_batch
                            attn_score_mc1 = torch.max(attn_score_mc1, torch.tensor(torch.finfo(attn_score_mc1.dtype).min))
                            loss_sim_k = nn.functional.mse_loss(attn_score1, attn_score_mc1)

                            q2 = attn2.qkt_matmul.quant_x1(q2)
                            k2_mc = attn2.qkt_matmul.quant_x2(k2_mc)
                            attn_score_mc2 = attn2.qkt_matmul(q2, k2_mc.transpose(2, 3)) / math.sqrt(attn2.head_dim)
                            attn_score_mc2 = attn_score_mc2+ attention_mask_batch
                            attn_score_mc2 = torch.max(attn_score_mc2, torch.tensor(torch.finfo(attn_score_mc2.dtype).min))
                            loss_sim_k += nn.functional.mse_loss(attn_score2, attn_score_mc2)

                            attn_score1 = attn_score2 = None
                            attn1 = attn2 = None
                            attn_score_mc1 = attn_score_mc2 = None

                        else:
                            loss_sim_k = loss_func(k1_mc[outlier_mask_k], k1[outlier_mask_k]) + loss_func(k2_mc[outlier_mask_k], k2[outlier_mask_k])


                        # 对v做同样的处理
                        norm_v1 = torch.norm(v1, p=2, dim=-1, keepdim=True) + 1e-6
                        norm_v2 = torch.norm(v2, p=2, dim=-1, keepdim=True) + 1e-6
                        dir_v1 = v1 / norm_v1
                        dir_v2 = v2 / norm_v2
                        
                        dot_product_v = torch.sum(dir_v1 * dir_v2, dim=-1).clamp(-1.0, 1.0)
                        dv = torch.acos(dot_product_v) * (1 / torch.pi)
                        dv_max = torch.max(dv, dim=-1).values.detach()
                        dv_min = torch.min(dv, dim=-1).values.detach()
                        dv_threshold = dv_min + mc_gamma * (dv_max - dv_min)
                        is_outlier_v = dv > dv_threshold.unsqueeze(-1)
                        
                        v_mc_dir = (1 - mc_alpha) * dir_v1 + mc_alpha * dir_v2
                        v_mc_dir = v_mc_dir / (torch.norm(v_mc_dir, p=2, dim=-1, keepdim=True) + 1e-6)
                        v1_mc = v_mc_dir * norm_v1
                        v2_mc = v_mc_dir * norm_v2
                        
                        outlier_mask_v = ~is_outlier_v
                        loss_sim_v = loss_func(v1_mc[outlier_mask_v], v1[outlier_mask_v]) + loss_func(v2_mc[outlier_mask_v], v2[outlier_mask_v])
                        
                        loss = a*loss_1 + b*loss_2 + c*loss_sim_k + d*loss_sim_v
                    # 如果出现了NaN(Not a number:一般是溢出了)就在终端提示并停止训练
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    # 统计量
                    loss_list_1.append(loss_1.detach().cpu())
                    loss_list_2.append(loss_2.detach().cpu())
                    loss_list_sim_k.append(loss_sim_k.detach().cpu())
                    loss_list_sim_v.append(loss_sim_v.detach().cpu())
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    omni_parameters_temp = list(get_omni_parameters(qlayer1, use_shift)) + list(get_omni_parameters(qlayer2, use_shift))
                    norm = loss_scaler(loss, optimizer,parameters= omni_parameters_temp).cpu()
                    norm_list.append(norm.data)

                loss_mean_1 = torch.stack(loss_list_1).mean()
                loss_mean_2 = torch.stack(loss_list_2).mean()
                loss_mean = torch.stack(loss_list).mean()
                loss_mean_sim_k = torch.stack(loss_list_sim_k).mean()
                loss_mean_sim_v = torch.stack(loss_list_sim_v).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} and layer {i+1} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                logger.info(f"loss1:{loss_mean_1:.6f} : {a*loss_mean_1:.6f} ,loss2:{loss_mean_2:.6f} : {b*loss_mean_2:.6f}")
                logger.info(f"loss_sim_k:{loss_mean_sim_k:.6f} : {c*loss_mean_sim_k:.6f}, loss_sim_v:{loss_mean_sim_v:.6f} : {d*loss_mean_sim_v:.6f}")
            clear_temp_variable(qlayer1)
            clear_temp_variable(qlayer2)
            del optimizer

            qlayer1.set_saveqkv_state(False)
            qlayer2.set_saveqkv_state(False)
            qlayer1.set_save_attn_score_state(False)
            qlayer2.set_save_attn_score_state(False)
        

        # real smooth and quantization:将得到的OmniQuant参数，进行真正的算子融合
        smooth_and_quant_inplace(qlayer1, args, is_llama)
        smooth_and_quant_inplace(qlayer2, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            # 更新quant_inps，并保存这一层训练得到的OmniQuant参数
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer1(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        quant_inps[j] = qlayer2(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            # 调回fp16
            qlayer1.half() 
            qlayer2.half()
            register_scales_and_zeros(qlayer1)
            register_scales_and_zeros(qlayer2)
            layers[i] = qlayer1.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer1)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
            layers[i+1] = qlayer2.to("cpu")
            omni_parameters[i+1] = omni_state_dict(qlayer2)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            # 非训练模型，直接移动会cpu（逐层推理和训练）
            # 调回fp16
            qlayer1.half() 
            qlayer2.half()
            register_scales_and_zeros(qlayer1)
            register_scales_and_zeros(qlayer2)
            qlayer1.set_saveqkv_state(False)
            qlayer2.set_saveqkv_state(False)
            layers[i] = qlayer1.to("cpu")
            layers[i+1] = qlayer2.to("cpu")
        

        if args.real_quant:
            # 没有实现激活权重同时量化时的，量化参数算子融合代码，所以只能pack仅权重量化的参数
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer1)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer1, q_linear)       
                print(f"pack quantized {name} finished")
                del module

            named_linears = get_named_linears(qlayer2)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer2, q_linear)       
                print(f"pack quantized {name} finished")
                del module
                    
        del layer1,layer2
        torch.cuda.empty_cache()


    # 模型后面部分，正常处理
    for i in range(end_layer,len(layers)):
        logger.info(f"=== Start quantize back layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model：获得待对齐的张量（关闭量化）
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        

        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # 对llama模型不使用平移因子（它原本的特征均值就倾向于0）
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            # 使用smoothquant初始化scale,outliner-suppression来初始shift
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))

        # 若resume,则加载OmniQuant参数            
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 开始训练
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        # OmniQuant调整之后再量化
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        # quant_out = qlayer(quant_inps[index:index+args.batch_size], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        # # 核心：损失函数，我们要修改这里
                        # loss = loss_func(fp_inps[index:index+args.batch_size], quant_out)
                        # if args.aug_loss:
                        #     loss += loss_func(fp_inps_2[index:index+args.batch_size], quant_out)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size], attention_mask=attention_mask_batch,position_ids=position_ids)
                        loss = loss_func(fp_inps[index:index+args.batch_size].view(-1, fp_inps.shape[-1]), 
                                         quant_out.view(-1, quant_out.shape[-1]))
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size].view(-1, fp_inps_2.shape[-1]), 
                                              quant_out.view(-1, quant_out.shape[-1]))
                        else:
                            loss += loss
                    
                    # 如果出现了NaN(Not a number:一般是溢出了)就在终端提示并停止训练
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    # 统计量
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        

        # real smooth and quantization:将得到的OmniQuant参数，进行真正的算子融合
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            # 更新quant_inps，并保存这一层训练得到的OmniQuant参数
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            # 调回fp16
            qlayer.half() 
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            # 非训练模型，直接移动会cpu（逐层推理和训练）
            # 调回fp16
            qlayer.half() 
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        if args.real_quant:
            # 没有实现激活权重同时量化时的，量化参数算子融合代码，所以只能pack仅权重量化的参数
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module
                
        del layer
        torch.cuda.empty_cache()



    if args.layer_similarity:
        logger.info("======layer_similarity_train_argument======")
        logger.info(f"a:{args.sim_a},b:{args.sim_b},c:{args.sim_c},d:{args.sim_d}")
        logger.info(f"begin_layer:{args.begin_layer},end_layer:{args.end_layer}")
        logger.info(f"minicache_alpha:{args.minicache_alpha},minicache_gamma:{args.minicache_gamma}")
        logger.info("===========================================")


    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

