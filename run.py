
# 辅助工具
import argparse
from Evaluate.logger import create_logger
from Evaluate import evaluate_utils

# 基础模型接口
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 自定义组件
from myLlama_commonKV.modeling_llama import LlamaForCausalLM
from myLlama_commonKV.configuration_llama import LlamaConfig

# commonKV
from safetensors import safe_open
from safetensors.torch import load_file
from ChangeModelToCommonKV.utils import get_commonKV_data

# let
from ScaleOperation.utils import get_let_data

# let训练
from ScaleOperationTrain.train import train_let_parameters

def evaluate(model,tokenizer,args,logger):
    if args.eval_my_text:
        evaluate_utils.eval_my_text(model,tokenizer,args,logger)
    if args.eval_hidden_state_similarity:
        evaluate_utils.eval_hidden_state_similarity(model,tokenizer,args,logger)
    if args.eval_KVcache_similarity:
        evaluate_utils.eval_KVcache_similarity(model,tokenizer,args,logger)
    if args.eval_latentKV_similarity:
        evaluate_utils.eval_latentKV_similarity(model,tokenizer,args,logger)
    if args.eval_latentKV_modulus:
        evaluate_utils.eval_latentKV_modulus(model,tokenizer,args,logger)
    if args.eval_outliner_cosine_similarity:
        evaluate_utils.eval_outliner_cosine_similarity(model,tokenizer,args,logger)
    if args.eval_latentKV_similarity_to_mean_value:
        evaluate_utils.eval_latentKV_similarity_to_mean_value(model,tokenizer,args,logger)


def main():
    # 1. 加载参数
    parser = argparse.ArgumentParser()

    # 模型路径
    parser.add_argument("--model", type=str, help="The path of model",
                        default="./model_cache/Meta-Llama-3.1-8B-Instruct")
    # 启动commonKV
    parser.add_argument("--use_commonKV", action="store_true", help="Turn on commonKV")
    parser.add_argument("--commonKV_parameters", type=str, help="The path of commonkv parameters",
                        default="./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32/commonKV_parameters.safetensors")
    
    # 启动let
    parser.add_argument("--use_let", action="store_true", help="Turn on let")
    parser.add_argument("--let_parameters", type=str, help="The path of let parameters",
                        default="./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32/let_parameters.safetensors")

    # let缩放因子训练
    parser.add_argument("--train_let",action="store_true")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--nsamples",type=int,default=20)
    parser.add_argument("--epoches",type=int,default=20)
    parser.add_argument("--loss_scale_factor",type=float,default=100)
    parser.add_argument("--let_lr",type=float,default=1e-2)
    parser.add_argument("--let_save_dir",type=str,default="./SVD_KV_parameters/trained_let_parameters")

    # 合并算法
    parser.add_argument("--is_merge",action="store_true")
    parser.add_argument(
        "--merge_algorithm", 
        type=str, 
        choices=["MiniCache_X", "Mean", "Fisher_Mean"],
        default="MiniCache_X",                          
        help="Select the algorithm to merge latentKV"
    )
    parser.add_argument("--MiniCache_X_merge_step",type=int,default=8)

    #测试
    parser.add_argument("--log_path",type=str)
    parser.add_argument("--eval_my_text", action="store_true")
    parser.add_argument("--eval_hidden_state_similarity",action="store_true")
    parser.add_argument("--eval_KVcache_similarity",action="store_true")
    parser.add_argument("--eval_latentKV_similarity",action="store_true")
    parser.add_argument("--eval_latentKV_modulus",action="store_true")
    parser.add_argument("--eval_outliner_cosine_similarity",action="store_true")
    parser.add_argument("--eval_latentKV_similarity_to_mean_value",action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    args = parser.parse_args()


    # 2.日志生成器
    logger = create_logger(args.log_path)

    # 3. 加载分词器 和 模型（LlamaForCausalLM）
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = LlamaConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        config = config,
        dtype=torch.bfloat16, 
        device_map="auto",          
        trust_remote_code=True      
    )

    logger.info(f"Model is loaded on: {model.device}")
    logger.info(f"Model config max_position_embeddings: {model.config.max_position_embeddings}")

    # 4. 转为CommonKV模式
    merge_args = {
        "is_merge" : args.is_merge,
        "merge_algorithm" : args.merge_algorithm,
        "MiniCache_X_merge_step" : args.MiniCache_X_merge_step
    }


    if args.use_commonKV:
        commonKV_meta_data, commonKV_parameters = get_commonKV_data(args.commonKV_parameters)
        model.switch_to_commonKV(commonKV_meta_data, commonKV_parameters, merge_args)

    # 5.启动let变换
    if args.use_let:
        let_meta_data, let_parameters = get_let_data(args.let_parameters)
        model.switch_to_let(True,let_meta_data,let_parameters)

    # 6. 训练
    if args.train_let:
        train_let_parameters(model,commonKV_meta_data,let_meta_data,args,logger)

    # 7 .测试
    evaluate(model,tokenizer,args,logger)

if __name__ == "__main__":
    main()