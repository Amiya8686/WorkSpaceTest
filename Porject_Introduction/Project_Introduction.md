# 项目介绍

## 复现CommonKV

### 预处理：对KV参数进行联合SVD

#### 算法部分

**参数：**组数（多少层为一组），保留百分之几的能量

对原模型的KV参数进行联合SVD，存到SVD_KV_parameters中。

根据保存多少能量，来决定保存前多少个特征值（秩）。



#### 测试部分

看看联合SVD之后的能量集中程度



#### 存储的safetensors文件格式约定

```python
# meta data
metadata = {
    "model": args.model,
    "layers_num": str(num_layers),
    "group_num": str(args.group_num),
    "energy_threshold": str(args.energy_threshold)
}


# tensors
svd_kv_paramters_to_saved={
    A_group0: ;
    A_group1: ;
    B_0_k: ;        # B时和Wo进行了算子融合的
    B_0_k_bias: ;
    B_0_V: ;
    B_0_V_bias: ;
    B_1_k: ;
    B_1_V: ;  
    group_lists: json.dumps(groups); #每一组对应的层列表
}
```



### ROPE问题

由于无法对潜在KV进行ROPE，我们每次前向传播对潜在KV重构后，都要重新进行ROPE

实现一个AdaptiveROPECache类，维护整张ROPE表，在前向传播时position_embeddings传入所有ROPE参数



**存储开销分析：**

- KVcache占用： head_num * head_dim * 2 * layer_num

- ROPE参数占用： head_dim * 2   

- ROPE参数存储远小于KVcache占用



**计算开销估计：**

- A、B、ROPE参数都是热数据，会被缓存锁住，所以开销不算大



### KVcache管理问题

我们实现一个KVcache管理类，对每一组的KVcache进行单独管理，可以自定义合并逻辑

将LatentKVCache挂在到past_key_values中参与前向传播



### 合并层间的KV

暂且使用均值合并（其实我们打算先不启动合并）





### 项目启动

#### 加载CommonKV参数

**参数加载：**推理时先加载原模型参数，然后加载SVD_KV_parameters（同一份参数注册到不同的Decoder中，用的只是一份参数）

**推理：**直接像commonKV论文那样，调用A和B来进行运算就行





## 加入缩放因子

### 初始化缩放因子

读取SVD_parameters，然后获得一个初始化的缩放因子参数（由于缩放因子和秩有关，所以和具体的SVD参数有关）

let_parameters.safetensors格式的设计

```python
# meta data
metadata = {
    "model": ;
    "layers_num": ;
    "let_layers_num:";
    "let_layers:";
}


# tensors
let_parameters_to_save={
    let_l0:;
    let_l1:;
    let_l{layer_idx}:;
}
```



### 缩放因子挂载

直接通过LlamaModel的switch_to_let接口进行挂载（记得封装为nn.Parameters类型）





### 缩放因子训练

**设计思路：**

获得第一层输入

获得掩码矩阵

非合并组层，直接跳过



组合并层，先传播一次获得均值

再回来，一层层进行均值对齐

Loss:计算方法：

获得 lkv0 与 lkvm 的余弦值矩阵：均值接近1



保存：

最后从模型中取回缩放因子，进行保存



**日志示例：**

```
======== train let factor ========
nsamples 20; epoches 2; loss_scale_factor:100
===group [0,1,2]===
layer 0: epoch 0: loss:0.111;11.1, cosim:0.95, max_memory:16G
layer 0: epoch 1: loss:0.099:9.9, cosim:0.95,  max_memory:12G
layer 1: epoch 0: loss:0.66;66, cosim:0.95,  max_memory:12G
layer 1: epoch 1: loss:0.09;9, cosim:0.95,  max_memory:12G
...
===group [3,4,5]===
layer 3: epoch 0: loss:0.111;11.1, cosim:0.95,  max_memory:12G
layer 3: epoch 1: loss:0.099:9.9, cosim:0.95,  max_memory:12G
layer 4: epoch 0: loss:0.66;66, cosim:0.95,  max_memory:12G
layer 4: epoch 1: loss:0.09;9, cosim:0.95,  max_memory:12G
...
```





## 层间合并

### 分块离群判断的MiniCache（MiniCache_X）

分解为单位方向向量和模长

模长单独保留

单位方向向量取均值

离群值，固定若干步选取余弦相似性最小的->离群矩阵（一维张量）+ 原始单位方向向量：合并步先设成8

这个余弦相似性如何衡量？均值？**首尾两层？**



存储结构：

（1）合并后的单位方向向量

（2）模长矩阵 （[batch_size,seqlen]）: 每一层都要存一个

（3）离群矩阵（[bacth_size,seqlen]）：所有层共用一个即可

（4）离群值：（outlinerlen，hidden_dim）每一层存一个



分块判断离群值好处：

（1）离群值分布更离散，方便hold住语义结构

（2）最近一段时间的latentKV原精度保留，它们可能更重要（之前的一篇论文QAQ）



### 直接取均值（Mean）





### Fisher信息加权均值（Fisher_Mean）





## 测试函数

### 测试层间的latentKV模长差异

**eval_latentKV_modulus**

**PS:**测试是否有单独保留模长的必要

**日志格式：**输入若干个token的每一层latentKV的模长

```
========modulus of latentKV========
====group [0,1,2]====
token 0:
layer0: 0.555
layer1: 2.33
layer2: 0.4154
token 1: 0.66
layer0: 0.55
layer1: 0.66
layer2: 0.137
====group [3,4,5]====
token 0:
layer0: 0.555
layer1: 2.33
layer2: 0.4154
token 1: 0.66
layer0: 0.55
layer1: 0.66
layer2: 0.137





```



### 测试离群token的分布

**eval_outliner_of_cosine_similarity**

输入，最不相似和最相似的若干个latentKV，和平均相似性

**PS：**看一下离群分布

**日志格式：**

```
======== The cosine similarity of the least similar latentKV ======== 
====layer 0-1====
least similar:
token2: cosim: 0.154
token1: cosim: 0.22
most similar:
token7: cosim: 0.97
token4: cosim: 0.96
mean:
cosim: 0.66
====layer 1-2====
least similar:
token4: cosim: 0.14
token1: cosim: 0.32
most similar:
token8: cosim: 0.94
token9: cosim: 0.82
mean:
cosim: 0.55

```



## 测试latentKV相对单位方向向量平均插值的余弦相似性：

**eval_latentKV_similarity_to_mean_value**

我们的合并算法，基于均值，我们需要查看latentKV相对均值的余弦相似性

**日志格式：**

```
========eval_latentKV_cos_similarity_to_mean_value========
====group [0,1,2]====
layer0: 0.55
layer1: 0.67
layer2: 0.57
====group [3,4,5]====
layer0: 0.55
layer1: 0.67
layer2: 0.57
```





