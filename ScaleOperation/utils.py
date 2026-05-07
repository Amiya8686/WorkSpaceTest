import json

from safetensors import safe_open
from safetensors.torch import load_file


# 读取let参数
def get_let_data(path):

    # 读取let元数据 并 对元数据进行格式转换（不要再存str）
    with safe_open(path, framework="pt", device="cpu") as f:
        let_meta_data = f.metadata() or {}

    # 对元数据进行格式转换
    if "model" in let_meta_data:
        let_meta_data["model"] = str(let_meta_data["model"])
    if "layers_num" in let_meta_data:
        let_meta_data["layers_num"] = int(let_meta_data["layers_num"])
    if "let_layers_num" in let_meta_data:
        let_meta_data["let_layers_num"] = int(let_meta_data["let_layers_num"])
    if "let_layers" in let_meta_data:
        let_meta_data["let_layers"] = json.loads(let_meta_data["let_layers"])

    # 读取let参数
    let_parameters = load_file(path, device="cpu")

    # 返回元数据 和 参数
    return let_meta_data, let_parameters



