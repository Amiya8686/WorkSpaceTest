import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(model,nsamples,seqlen,seed):
    """模型路径，样本数，样本长度，种子"""
    print("get_wikitext2")
    # 加载数据
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)

    # 分词：文本 -> 词索引序列 (词索引 -> 特征向量 是模型的embedding做的) 
    tokenizer = AutoTokenizer.from_pretrained(model)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # 提取训练样本：
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_loaders(name,model,nsamples=128,seqlen=2048,seed=0):
    "数据集，模型路径，样本数，样本长度，种子"
    if 'wikitext2' in name:
        return get_wikitext2(model,nsamples,seqlen,seed)
