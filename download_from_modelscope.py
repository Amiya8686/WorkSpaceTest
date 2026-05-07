import os
import argparse
from modelscope.hub.snapshot_download import snapshot_download


# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, help="The repo id of model on modelscope",
                    default="LLM-Research/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--local_dir", type=str, help="The local path to save the model",
                    default="./model_cache/Meta-Llama-3.1-8B-Instruct")
args = parser.parse_args()


# 下载模型
repo_id = args.repo_id
local_dir = args.local_dir
print(f"Preparing to download the word...")
print(f"Starting to download the model {repo_id}, please be patient")
try:
    snapshot_download(
        repo_id=repo_id,
        cache_dir=os.path.dirname(local_dir),                       # 指定缓存根目录(下载过程中断点重传，哈希等)
        local_dir=local_dir,                                        # 模型最终保存路径
        ignore_file_pattern=["*.pth", "*.msgpack", "original/*"],   # 忽略无关文件
        revision="master",                                          # 下载的仓库分支
    )
    print(f"Download complete! Model saved in {local_dir} ")
except Exception as e:
    print(f"An error occurred during the download process: {e}")