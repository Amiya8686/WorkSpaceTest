import argparse
import os
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# 保证从工作区根目录直接执行脚本时，能找到同级包
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ChangeModelToCommonKV.utils import get_commonKV_data

parser = argparse.ArgumentParser()
parser.add_argument("--commonKV_parameters", type=str, help="The path of commonKV paramters",
                    default="./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32/commonKV_parameters.safetensors")
parser.add_argument("--save_dir", type=str, help="The path to save let paramters")
args = parser.parse_args()


# 1.读取commonKV_parameters的元数据
commonKV_parameters_path = args.commonKV_parameters
commonKV_meta_data, _ = get_commonKV_data(commonKV_parameters_path)

group_num = commonKV_meta_data.get("group_num", 0)
energy_threshold = commonKV_meta_data.get("energy_threshold", 0.0)

# 2.根据秩，为每一组的每一层创建缩放因子(一个一维的全1张量)
let_parameters_to_save = {}
let_layers_num = 0
let_layers = []

group_lists = commonKV_meta_data.get("group_lists", [])
group_r = commonKV_meta_data.get("group_r", [])
for group_idx, group_layers in enumerate(group_lists):
    let_init = torch.ones(group_r[group_idx], dtype=torch.float16)
    for layer_idx in group_layers:
        let_parameters_to_save[f"let_l{layer_idx}"] = let_init.clone()
        let_layers_num += 1
        let_layers.append(layer_idx)

 
# 3.准备元数据 和 张量 python对象
metadata = {
    "model": str(commonKV_meta_data.get("model", "")),
    "layers_num": str(commonKV_meta_data.get("layers_num", 0)),
    "let_layers_num": str(let_layers_num),
    "let_layers":json.dumps(let_layers)
}


# 4.将其存入safetensors文件
save_root = args.save_dir if args.save_dir else os.path.dirname(commonKV_parameters_path)
os.makedirs(save_root, exist_ok=True)
save_path = os.path.join(save_root, "let_parameters.safetensors")
save_file(let_parameters_to_save, save_path, metadata=metadata)