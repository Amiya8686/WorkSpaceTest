import argparse
import torch
from safetensors.torch import load_file
from safetensors import safe_open


parser = argparse.ArgumentParser()
parser.add_argument("--SVD_KV_parameters", type=str, help="The path of SVD KV parameters ",
                    default="./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9/commonKV_parameters.safetensors")
parser.add_argument("--energy_threshold", type=float, help="The proportion of retained energy",
                    default=0.9)
args = parser.parse_args()


# 1.读取safetensors文件，获取元数据和张量数据
print(f"Loading: {args.SVD_KV_parameters}")
tensors = load_file(args.SVD_KV_parameters)
with safe_open(args.SVD_KV_parameters, framework="pt") as f:
    metadata = f.metadata()
    print(f"Metadata: {metadata}")



# 2. 获取每一组的A的Key
a_keys = sorted([k for k in tensors.keys() if "A_group" in k], 
                key=lambda x: int(x.split('group')[-1])) # group10 排在 group2之后

# 3. 分别SVD
print(f"\n{'Group':<12} | {'Original Rank':<15} | {'Sub-Rank':<10} | {'Ratio':<10} | {'Energy'}")
print("-" * 75)
for key in a_keys:
    # 获取A并转置
    A = tensors[key].to(torch.float32).t() 
    
    # 奇异值分解：A 的形状为 [in, r]，SVD 后 S 的长度为 r
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # 奇异值向量保留
    energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    idx = torch.where(energy >= args.energy_threshold)[0][0].item()
    sub_rank = idx + 1
    
    # 计算比例：子秩 / 原始压缩后的秩
    original_r = len(S)
    ratio = sub_rank / original_r
    print(f"{key:<12} | {original_r:<15} | {sub_rank:<10} | {ratio:.2%} | {energy[idx]:.4f}")