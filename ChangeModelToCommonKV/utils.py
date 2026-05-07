import json

from safetensors import safe_open
from safetensors.torch import load_file


# 读取commonKV参数
def get_commonKV_data(path):
    # 读取commonKV元数据 并 对元数据进行格式转换（不要再存str）
    with safe_open(path, framework="pt", device="cpu") as f:
        commonkv_meta_data = f.metadata() or {}

    # 对元数据进行格式转换
    if "model" in commonkv_meta_data:
        commonkv_meta_data["model"] = str(commonkv_meta_data["model"])
    if "layers_num" in commonkv_meta_data:
        commonkv_meta_data["layers_num"] = int(commonkv_meta_data["layers_num"])
    if "commonKV_layers_num" in commonkv_meta_data:
        commonkv_meta_data["commonKV_layers_num"] = int(commonkv_meta_data["commonKV_layers_num"])
    if "group_num" in commonkv_meta_data:
        commonkv_meta_data["group_num"] = int(commonkv_meta_data["group_num"])
    if "energy_threshold" in commonkv_meta_data:
        commonkv_meta_data["energy_threshold"] = float(commonkv_meta_data["energy_threshold"])
    if "group_lists" in commonkv_meta_data:
        commonkv_meta_data["group_lists"] = json.loads(commonkv_meta_data["group_lists"])
    if "group_r" in commonkv_meta_data:
        commonkv_meta_data["group_r"] = json.loads(commonkv_meta_data["group_r"])

    # 读取commonKV参数
    commonkv_parameters = load_file(path, device="cpu")

    # 返回元数据 和 参数
    return commonkv_meta_data, commonkv_parameters



