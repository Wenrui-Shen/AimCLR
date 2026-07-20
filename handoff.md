# AimCLR / ReSA / OSE / CTR-GCN 实验交接

> 面向完全没有上下文的新会话。最后更新：2026-07-21。
>
> 请先完整阅读本文，再检查代码和服务器状态。不要根据旧聊天标题或文件名猜测实验协议。

## 0. 一句话概况

我们正在研究：用每类一个带标签 exemplar 构造 OSE 类别原型，把 `Lproto`、`Lmix-proto`
和 `Lmix-ins` 加到骨架自监督基线上，并回答两个问题：

1. 原型使用无标签 queue 邻居还是同一带标签 exemplar 的多个弱增强视图更好？
2. OSE 与 ReSA、AimCLR 分别是否存在目标冲突，以及 ST-GCN 换成 CTR-GCN 后效果和代价如何？

当前最重要的实验事实是：

```text
ST-GCN + ReSA + OSE M-F
Q8: 78.80
Q4: 79.98
Q0: 77.22
MV4: 79.47
```

MV4 比单视图 Q0 高 2.25，距离 Q4 只差 0.51，说明同一 exemplar 的多个独立弱增强
已经补回大部分有效变化。Q4 仍略高，但单次实验的 0.51 差距不足以证明两者存在稳定差异。
不能把结论简化为“top-k 越低越好”或“弱多视图已经严格等价于真实实例邻居”。

当前代码已经实现纯弱增强多视图原型（MV4）、10 层 ST-GCN 宽度的 CTR-GCN，以及
按当前协议重构的现代 AimCLR+OSE。CTR 尚无可比较结果，AimCLR A0/A1/A2 尚未运行。
旧 AimCLR+OSE 的 LP 约 66，经检查不能直接证明理论冲突，因为旧实现和评估协议也存在
多项混杂问题。

## 1. 工作区与统一实验协议

```text
本地仓库：D:\Program\codex\program\AimCLR
服务器：/home/user9/public3/swr/AimCLR
服务器环境：swr_aimclr
Dataset：NTU60
Protocol：xsub
Stream：joint
输入：[N, 3, 50, 25, 2]
Pretext：300 epochs
正式 linear evaluation：200 epochs
Batch size：128
正式增强：weak + weak
Dropout：0
单 GPU
```

服务器数据和 checkpoint 不在本地工作区。不要假设本地配置的 `device` 与服务器当前
设备相同。

统一术语：

```text
ReSA-only = Lcluster
B0        = Lcluster + Lproto
M-P       = B0 + Lmix-proto
M-F       = B0 + Lmix-proto + Lmix-ins
```

不要再用含糊的 “proto-only” 指代不同 loss 组合。

## 2. 已得到的 ST-GCN 正式结果

以下均为 NTU60 xsub joint、weak+weak、dropout0、batch128。正式横向比较应使用
pretext300 和 LP200。

| 版本 | Pretext checkpoint | OSE queue top-k | exemplar views | LP Top-1 |
|---|---:|---:|---:|---:|
| B0 | 120 | 8 | 1 | 74.39 |
| B0 | 300 | 8 | 1 | 75.95 |
| M-P | 300 | 8 | 1 | 78.20 |
| M-F | 300 | 8 | 1 | 78.80 |
| M-F | 300 | 4 | 1 | **79.98** |
| M-F | 300 | 0 | 1 | **77.22** |
| M-F / MV4 | 300 | 0 | 5 | **79.47** |

已确认的差值：

```text
Q4 - Q8 = +1.18
Q0 - Q4 = -2.76
Q0 - Q8 = -1.58
MV4 - Q0 = +2.25
MV4 - Q4 = -0.51
MV4 - Q8 = +0.67

Q8 下：
M-P - B0 = +2.25
M-F - B0 = +2.85
Lmix-ins 在 M-P 上额外贡献 = +0.60
```

解释：

- Q8 到 Q4 提升，支持“降低邻居污染、提高原型纯度”。
- Q4 到 Q0 大幅下降，说明不能无限降低 top-k。
- Q0 的 77.22 使用的是 `ose_exemplar_views: 1`，即只有一个在线 exemplar 视图；
  它不是多弱增强原型实验。
- Q4 的原型实际有五个组成：一个 exemplar anchor + 四个 queue 邻居。
- 新 MV4 也刻意使用五个组成：一个在线弱增强 + 四个 EMA 弱增强，以便公平比较
  “组成数量相同、来源不同”。
- MV4=79.47 表明弱多视图能补回 Q0 丢失的大部分信息；Q4 的真实类内实例目前只保留
  0.51 的小幅优势，最终是否稳定需要后续多 seed，而不是现在继续调 MV1/MV2。

历史上的 500-epoch schedule 或不同总 epoch 下的同名 checkpoint 不得混入上表。
Cosine LR 依赖总 epoch，同名 epoch 并不代表相同训练状态。

## 3. 当前代码状态

写本文前工作区是干净的，最新提交为：

```text
2d584eb ose proto change and ctrgcn test
93438d2 ctrgcn layer
9bcbf9d ctrgcn
676f680 Lmix
a98e33d OSEchange
9725427 dropout change and augmentation change
d622e0b resa
```

当前工作区除本文件外，还有尚未提交的现代 AimCLR+OSE 重构、配置和测试。不要误删或
用旧版 `OSEAimCLR` 覆盖。

关键文件：

```text
net/ose_resa.py
processor/pretrain_ose_resa.py
net/ctrgcn.py
net/aimclr.py
net/ose_aimclr.py
processor/pretrain_ose_aimclr.py
feeder/ntu_feeder.py
tests/test_ose_resa_lmix.py
tests/test_ose_aimclr.py
tests/test_ctrgcn.py
```

现有 ReSA/CTR 配置和本次新增的 AimCLR 配置：

```text
config/ntu60/pretext/
  pretext_ose_resa_lmix_full_mv4_xsub_joint.yaml
  pretext_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
  pretext_aimclr_a0_xsub_joint.yaml
  pretext_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
  pretext_ose_aimclr_q4_lmix_full_xsub_joint.yaml

config/ntu60/linear_eval/
  linear_eval_ose_resa_lmix_full_mv4_xsub_joint.yaml
  linear_eval_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
  linear_eval_aimclr_a0_xsub_joint.yaml
  linear_eval_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
  linear_eval_ose_aimclr_q4_lmix_full_xsub_joint.yaml
```

本地静态编译检查曾通过。由于本地 Python 没有可用的 `torch`，动态单元测试必须在
服务器环境执行：

```bash
python -m unittest tests.test_ctrgcn tests.test_ose_resa_lmix tests.test_ose_aimclr
```

不要声称服务器动态测试已经通过，除非新会话实际运行并看到结果。

## 4. 当前 ReSA+OSE 实现必须保持的语义

### 4.1 特征空间

当前 ReSA 本身有三种表示：

```text
H：encoder 特征
Z：projector 特征
Q：predictor 输出
```

当前代码：

- ReSA 的 Sinkhorn assignment 从 encoder `H` 的 batch 关系构造。
- ReSA 的跨视图预测 loss 使用 `Q` 对 teacher `Z`。
- OSE exemplar、queue、prototype、mixed branch 都在 projector `Z` 空间。

ReSA 论文关于 encoder 特征更适合聚类的分析，不代表可以直接把 OSE 原型随手换到
`H`。OSE 的分类分布、teacher target 和 mixed embedding 当前都在 `Z`；如果测试
encoder-space prototype，必须作为独立 Q4 消融，统一修改 prototype、student logits、
teacher target 和 mix 分支的空间，不能混用 H/Z/Q。

该 H-vs-Z 消融仍未实现、未运行，优先级低于 MV4 和干净的 AimCLR 对照。

### 4.2 Lmix

输入混合为：

```python
mixed_view = beta * view_b + (1.0 - beta) * view_a[mix_index]
```

`mixed_view` 只走：

```text
encoder_q -> projector_q -> normalized mixed_z
```

它不走 predictor，不进入 Sinkhorn，不进入 queue，不走 teacher。`beta` 是输入混合
权重；target 只在 loss 内构造并 detach，target 不是网络输入。

### 4.3 Exemplar 与标签

- 每类固定选一个 exemplar，seed 为 0。
- exemplar 必须从无标签训练 loader 中排除。
- exemplar 不得进入无标签 sampler 或被当作 queue 普通样本。
- 标签只能用于选择固定 exemplar，以及离线/日志中的邻居 purity 诊断。
- 标签不得进入模型、loss、queue 检索或 top-k 决策。
- exemplar cache 会验证 class IDs、indices、seed、样本数和标签一致性；不要退回旧版
  AimCLR 那种弱校验。

## 5. 已实现但尚未出结果的纯弱增强多视图原型

配置：

```text
config/ntu60/pretext/pretext_ose_resa_lmix_full_mv4_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_resa_lmix_full_mv4_xsub_joint.yaml
```

关键参数：

```yaml
ose_topk: 0
ose_exemplar_views: 5
ose_exclude_exemplars: True
ose_mix_proto_weight: 1.0
ose_mix_ins_weight: 1.0
```

这里的文件名 `MV4` 指“四个额外 EMA 弱增强视图”，总组成数是五个：

```text
1 个 online weak exemplar view（有梯度）
+ 4 个 EMA weak exemplar views（no_grad）
= 5 个 prototype components
```

每次 `_exemplar_batch()` 都从原始 exemplar 再独立调用 feeder 的 `_aug`。当前 weak
augmentation 是 temporal crop + shear，不包含 standard rotation。

四个额外 EMA 视图顺序前向。为了不让这些额外前向重复改变 teacher BN buffer，
`_teacher_exemplar_projection()` 会保存并恢复 running mean、running var 和
`num_batches_tracked`。

prototype 的各组成先归一化，再根据它们与 online exemplar anchor 的相似度做 softmax
加权。当前代码不对加权和再次归一化；top-k 为 0 时不读取 queue 邻居。

一个容易误解的细节：当前 ReSA 代码即使 `ose_topk: 0`，仍会把 teacher batch
embedding 写入内部 queue；该 queue 不参与 MV4 prototype 或 ReSA loss，因此没有语义
冲突，只产生少量无用维护开销。若以后要做“代码层面完全无 queue”的版本，可以再跳过
enqueue，但不要把这项工程优化混进第一次 MV4 结果。

当前状态：

```text
实现：完成
配置：完成
单元测试：已添加，本地无法动态运行
服务器 300-epoch pretext：已完成
LP200：79.47
```

## 6. CTR-GCN 已做的工作和当前运行状态

### 6.1 公平宽度版本

官方风格 10 层 CTR-GCN 默认宽度远大于本项目 AimCLR ST-GCN：

```text
官方 CTR-GCN：
64,64,64,64,128,128,128,256,256,256

本项目 ST-GCN：
16,16,16,16,32,32,32,64,64,256
```

`net/ctrgcn.py` 已支持显式 `layer_channels`。当前公平比较配置保留 CTR 动态图单元、
10 层深度和 stride 位置，但把每层宽度对齐 ST-GCN：

```yaml
num_layers: 10
layer_channels: [16, 16, 16, 16, 32, 32, 32, 64, 64, 256]
```

必须准确称它为“10-layer CTR-GCN with ST-GCN-matched widths”，不要称为官方宽度
CTR-GCN。它使用官方风格的 CTR 运算和深度/下采样结构，但宽度是我们的受控变量。

对应配置：

```text
pretext_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
linear_eval_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
```

配置锁定 Q4 M-F、weak+weak、dropout0、batch128、pretext300、LP200，用于和当前
ST-GCN 79.98 做 backbone-only 比较。

### 6.2 服务器实测

用户在服务器手工把设备改为 0 后运行：

```bash
python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
```

日志确认：

```text
device: [0]
parameters: 12.177 M
Iter 0:  10:03:01
Iter 100: 10:06:37
```

也就是约 216 秒/100 iter，约 2.16 秒/iter。40031 个无标签样本、batch128，
每 epoch 约 312–313 iter，300 epochs 约 9.4 万 iter：

```text
纯迭代估计约 56.3 小时
考虑数据、日志、存盘：约 58–62 小时
```

用户明确认为这个时长不可接受。

此前“CTR 一直停在 Iter 0、后启动的 ST-GCN 已到 Iter 100”不能直接解释为死锁：

- 日志只在整个 forward、backward、optimizer step 完成后打印 `Iter 0 Done`。
- CTR 的动态关系张量比 ST-GCN 固定邻接昂贵得多。
- 当时还可能有 GPU 设备竞争；换到 device0 后已证明能正常推进。

本文不知道服务器上的 CTR 进程现在是否仍在运行。新会话第一步应只读检查进程和日志，
不要擅自 kill、resume 或重启。

本地 CTR 配置当前仍写 `device: [1]`，而服务器成功日志是手工改成 `[0]`。启动任何
新任务前必须明确检查设备；不要直接复制本地 device 值。

### 6.3 不要误解减少层数

历史上做过 10/8/3 层 CTR 尝试。三层版本是本项目人为定义的
`64 -> 128 -> 256, stride 1/2/2`，不是 SCD-Net；SCD-Net 的三层 CTR 是
`64 -> 256 -> 64`、全 stride1，且输出时空特征给 Transformer。

减少 CTR 层数会同时改变容量、感受野和下采样路径，不能再作为与 ST-GCN 的严格
backbone-only 比较。若因预算运行，只能明确标为 depth proxy/消融，不能用来下结论
“CTR-GCN 与 ST-GCN 谁更好”。

## 7. 前向次数与速度结论

当前 ReSA baseline 的 backbone 前向：

```text
online view_a
online view_b
teacher view_a
teacher view_b
= 4 次
```

因此：

```text
ReSA-only       = 4 次
B0/Q0/Q4/Q8     = 5 次（再加 online exemplar）
M-P/M-F         = 6 次（再加 online mixed）
ReSA MV4 M-F    = 10 次（再加 4 个 EMA exemplar views）
```

AimCLR baseline 的 backbone 前向约为 3 次：

```text
online normal
online extreme（drop 特征由同一次 encoder 调用返回）
teacher key
```

所以“去掉 ReSA 换成 AimCLR 会减少前向”方向上是对的，但只少一个 baseline forward：

```text
现代 AimCLR + MV4 M-F 预计约 9 次
ReSA + MV4 M-F 为 10 次
```

MV4 的四个额外 teacher views 才是新的主要成本。因此换 SSL baseline 不会自动让
CTR-GCN 训练从约 60 小时变成很短。把四个视图拼 batch 可以减少 Python 调用次数，
但不会减少总样本计算量，还会增加峰值显存。

“分支算完马上 backward 并释放 activation”的方案主要解决显存，不减少 FLOPs，
而且因重算 exemplar 可能更慢。若实现，必须做到：

1. teacher 分支 no_grad，先缓存小 embedding。
2. 分阶段对 loss 项 backward，梯度累积。
3. 每个逻辑 iteration 只做一次 optimizer step。
4. 每个逻辑 iteration 只做一次 EMA 更新和一次 queue 更新。
5. 若重算 exemplar，必须防止 BN running stats 重复更新。
6. 用同 seed 小规模比较 loss、梯度和最终 LP，不能只看能否跑通。

它仍未实现，不是当前速度问题的现成解决方案。

## 8. AimCLR 与 OSE 冲突的分析结论

### 8.1 用户的核心理解基本正确，但表述要精确

旧 AimCLR+OSE 不是同一块 queue 张量同时扮演负样本和 OSE 原型邻居：

- AimCLR 有自己的 `queue` 和 projector。
- OSE 另建 `ose_queue` 和 `ose_projector_q/k`。
- warmup 结束时 OSE 从 AimCLR head/queue 拷贝一次，之后两者各自更新。

真正的冲突是“相同来源的样本语义 + 共享 encoder 的优化目标”，不是同一 tensor 被
两个 loss 原地拉扯。

AimCLR 在 mining epoch 之前把非配对 queue 样本视为负样本；旧 OSE 同时从同一批
无标签样本形成的另一条 queue 取 top-8，作为类别 prototype 的组成。OSE 在 epoch 21
后启用、AimCLR 约到 epoch 150 才开始 top-1 邻居挖掘，因此 21–150 epoch 冲突最强：

```text
AimCLR：该邻居不是同一实例 -> 推远
OSE：该邻居与某类 exemplar 相似 -> 聚入原型并拉近相关样本
```

150 epoch 后 AimCLR 只提升少量 top-1 邻居为正样本，OSE top-8 中的大部分样本仍可能
是 AimCLR 的负样本，所以冲突只缓和、不消失。

queue 张量是 detached 的，不能写成“一个 loss 直接推远存储特征，另一个直接拉近
同一个存储特征”。梯度冲突发生在后续新样本和共享 encoder 上：

```text
AimCLR 倾向实例均匀/可分
OSE 倾向类别聚集
```

### 8.2 纯弱增强 prototype 能解决什么

若 OSE prototype 只由带标签 exemplar 的多个弱增强视图构造，不取任何无标签 queue
邻居，就消除了最明确的矛盾：

> 某个无标签样本既是 AimCLR 的负样本，又是 OSE 类别 prototype 的组成。

推荐未来的干净 AimCLR+OSE：

- AimCLR 保留自己的原生 queue，只服务 AimCLR。
- OSE 不建立/读取邻居 queue。
- OSE prototype 只来自固定 labeled exemplar 的弱增强多视图。
- exemplar 从无标签 loader 严格排除。
- OSE 使用独立、现代的 projector，先减少 head-level 冲突。
- encoder 仍共享，因此实例区分与类别聚集的天然梯度差异不会完全消失。

所以它是“移除 queue 身份矛盾”，不是数学上保证两种 loss 零冲突。

### 8.3 为什么 ReSA 更兼容

当前 ReSA 的 `Lcluster` 是 batch-level 软关系学习，不维护一个用于硬负样本判定的
持久 queue，也不会主动把 OSE queue 邻居推远。当前持久 queue 主要只服务 OSE 原型。

因此 ReSA+queue-neighbor OSE 没有 AimCLR 那种明确的硬矛盾，更可能互补：

```text
ReSA：学习当前 batch 的软关系结构
OSE：用少量类别 exemplar 给关系空间提供类别锚点
```

但不能说完全没有冲突。ReSA 可能按主体、视角、速度等 nuisance 聚类，仍可能与
OSE 的动作类别锚点产生梯度分歧。区别是软的、动态的，而不是硬负样本逻辑冲突。

### 8.4 Q0 结果对跨 baseline 的意义

ReSA 下 Q4 > Q0 说明真实实例多样性很有价值，但不能推导 AimCLR 下也一定 Q4 > MV4。
合理的预期反而是：

```text
ReSA：Q4 可能优于纯 MV4，因为没有硬负样本冲突且有实例多样性
AimCLR：纯 MV4 可能优于 Q4，因为移除了更强的 queue 目标冲突
```

这正是需要实验验证的 `SSL baseline × prototype source` 交互。

## 9. 旧 AimCLR+OSE LP≈66 的代码审计

相关文件：

```text
net/aimclr.py
net/ose_aimclr.py
processor/pretrain_ose_aimclr.py
config/ntu60/pretext/pretext_ose_aimclr_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_aimclr_xsub_joint.yaml
```

不能把约 66 单独归因于理论冲突，至少有以下混杂。

### 9.1 高置信度问题

1. **没有排除 exemplar。**
   旧 processor 选择每类 exemplar 后没有从无标签 loader 删除它们。exemplar 可能进入
   AimCLR queue/OSE queue，被当负样本，或成为自己/增强版本的近邻。协议不干净。

2. **旧 LP 配置默认只有 100 epoch。**
   当前正式结果是 LP200。如果 66 没有命令行覆盖到 200，它与 79.98 不能直接比较。

3. **head 和训练协议差异巨大。**

   ```text
   旧 AimCLR+OSE：feature_dim128、2-layer hidden256、queue32768、dropout0.5
   当前 ReSA+OSE：feature_dim256、3-layer hidden2048、queue8192、dropout0
   ```

4. **所有 OSE loss 一次性打开。**
   旧实现不能干净拆 B0、M-P、M-F，因此无法定位究竟是哪一项造成下降。

5. **没有 queue sample index 和 purity 日志。**
   无法检查旧 top-8 邻居到底有多少同类。

6. **prototype 梯度路径与当前实现不同。**
   旧 align/mix 使用 detached key prototype，另算一份 student exemplar prototype
   主要用于 dispersion；并不等价于当前的 online anchor + detached memory 语义。

7. **旧实现额外前向和 BN 处理不同。**
   它分别计算 key exemplar、student exemplar 和 mixed branch，总计约六次 backbone
   forward；不能拿它的效率或训练动态直接代表将来的现代实现。

### 9.2 条件性风险

- 旧 AimCLR/OSE enqueue 使用 `keys.device.index` 给写入位置额外加 batch offset。
  如果单卡实际是物理/logical `cuda:1`，而不是通过 `CUDA_VISIBLE_DEVICES` 映射成
  `cuda:0`，queue pointer 可能错位。旧 66 使用哪种设备方式需要查原日志。
- Git 历史中旧 OSE 初版之后还有 `ose path correct`、`ose update`、`tau update`。
  必须知道 66 checkpoint 的精确 commit，才能判断它是否包含后来修复的问题。
- 配置中的 tau 最终调用值是 student 0.1、teacher 0.04；函数默认值曾相反，但当前
  forward 会显式传参。不要在不知道 checkpoint commit 的情况下武断说 66 一定由
  tau 默认值造成。

结论：旧 66 只能作为“值得重新做干净实验”的线索，不能作为 queue 冲突的因果证据。
未来不要直接修修补补复用旧 `OSEAimCLR` 得出论文结论。

## 10. 当前真正卡在哪里

1. **CTR-GCN 速度不可接受。**
   10-layer、ST-width、Q4 M-F 可以跑，但预计完整 pretext 约 58–62 小时；用户不接受。
   当前服务器进程是否仍在运行未知。

2. **ReSA MV4 已有单次正式结果，但 0.51 差距尚未做多 seed。**
   MV4=79.47，Q4=79.98。主矩阵完成前不要先花预算追这个小差距。

3. **现代 AimCLR+OSE 已完成本地重构，尚缺服务器动态测试和正式结果。**
   A0、MV4 M-F、Q4 M-F 的 pretext300/LP200 配置已添加；本地无 `torch`，不能声称
   动态测试已经通过。

4. **旧 66 的实验身份不完整。**
   缺精确 commit、pretext checkpoint、LP epoch、设备、work_dir 和 seed。

5. **计算预算与因果完整性冲突。**
   只跑 AimCLR MV4 并与旧 66 比，无法区分 prototype source、实现修复、head 和 LP
   schedule。至少需要现代 AimCLR baseline 和同实现下的 Q4/MV4 对照。

## 11. 推荐的下一步实验顺序

### P0：先检查服务器现状，不做破坏性操作

只读确认：

```text
CTR 进程是否仍在运行
当前 epoch/iter
GPU0/GPU1 占用
日志和 work_dir
Q0/Q4/Q8 checkpoint、LP 配置、seed、最终日志路径
```

未经用户确认，不停止进程、不删 work_dir、不自动重启长任务。

### P1：ST-GCN ReSA MV4（已完成）

先用已经实现的配置：

```bash
python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_lmix_full_mv4_xsub_joint.yaml
```

启动前把 `device` 改成用户确认的空闲卡，使用独立 work_dir。完成 pretext300 后跑
对应 LP200。

这一步回答：

```text
同样五个 prototype components：
1 exemplar + 4 queue neighbors（Q4，79.98）
vs
1 online weak + 4 EMA weak（MV4，79.47）
```

结果解释：MV4 比 Q0 高 2.25、比 Q4 低 0.51。弱视图变化补回了大部分信息；真实类内
实例目前只有小幅优势，单次结果不足以判断该优势是否稳定。

### P2：现代 AimCLR+OSE（实现完成，等待验证和运行）

当前未提交重构已做到：

1. 保持原始 AimCLR baseline 和它自己的 queue。
2. exemplar 从无标签 loader 排除。
3. OSE 使用独立现代 projector；不要与 AimCLR 的 128-D instance head 强行共用。
4. MV4 在 `ose_topk: 0` 时既不读也不写 OSE queue；Q4 使用独立 OSE queue。
5. 支持 B0、M-P、M-F 独立开关。
6. 使用和当前 ReSA 版本一致的 mix 公式、detach 语义、温度和 exemplar seed。
7. LP 改为正式 200 epoch，独立 work_dir。
8. 测试锁定 AimCLR baseline 3 次、MV4 M-F 9 次 backbone forward；正式运行仍需记录显存和 iter time。
9. 添加 loader 排除、prototype component、loss switch 和梯度路径测试。
10. `ose_warmup_epoch: 0`，从 epoch 1 的第一个 batch 同时启用 AimCLR、Lproto 和配置的 Lmix。

建议实验顺序：

```text
A0：原始 AimCLR，LP200
A1：AimCLR + MV4 M-F，LP200
A2：AimCLR + Q4 M-F（同一现代实现），LP200
然后再拆 B0 / M-P / M-F
```

若预算只能先跑一个 OSE 版本，优先 A1；但不能用 A1 对旧 66 做严格因果结论。A2 是
验证 queue conflict 的必要受控对照。

最终理想矩阵：

| SSL baseline | Q4 queue prototype | MV4 weak-view prototype |
|---|---:|---:|
| ReSA | 79.98 | 79.47 |
| AimCLR | 待用现代实现重跑 | 待跑 |

### P3：再决定 CTR

有三种明确选择，必须让用户决定：

1. 接受约 60 小时，完成 10-layer ST-width 严格比较。
2. 跑较短 depth proxy，但明确不作为正式 ST-vs-CTR 结论。
3. 暂停 CTR，把预算用于更重要的 ReSA/AimCLR × prototype source 矩阵。

分阶段 backward 只在显存成为问题时实现；它不会根治 2.16 秒/iter 的计算成本。

### P4：低优先级消融

- Q4 下 OSE prototype 使用 encoder H 还是 projector Z。
- MV1/MV2/MV4 的视图数量—速度—精度折中。
- 是否完全跳过 topk0 时无用的 ReSA queue enqueue。
- 多 seed 复验最优点。

这些都应在主矩阵之后，避免同时改变太多变量。

## 12. 绝对不要再踩的坑

1. 不要说“top-k 越低越好”；Q0=77.22 已反证。
2. 不要把 Q0 单 exemplar 当成 MV4 多弱增强结果。
3. 不要把旧 AimCLR queue 和 OSE queue 说成同一个张量；冲突在共享样本语义和 encoder。
4. 不要拿旧 LP100 的约 66 与当前 LP200 的 79.98 直接比较。
5. 不要直接复用旧 `OSEAimCLR` 得出因果结论。
6. 不要让 exemplar 回到无标签 loader、AimCLR queue 或 OSE neighbor queue。
7. 不要让标签进入训练 loss、模型输入、top-k 或 queue；标签仅用于 exemplar 选择和诊断。
8. 不要把 OSE prototype、exemplar、teacher target 混在 H/Z/Q 不同空间。
9. 不要让 mixed branch 进入 predictor、Sinkhorn、teacher 或 queue。
10. 不要把 probability target 当作输入混合；输入权重是 `beta`。
11. 不要恢复 weak+standard rotation；当前正式协议是 weak+weak。
12. 不要仅凭 purity、entropy 或 pretext loss 选择最终模型；LP200 才是主要指标。
13. 不要混用不同 total-epoch cosine schedule 的 checkpoint。
14. 不要复用 work_dir 覆盖旧实验。
15. 不要把 `weights + start_epoch` 当完整 resume；optimizer、scheduler、EMA、queue 状态可能缺失。
16. 不要把 ST-width CTR 称为官方宽度 CTR。
17. 不要把本项目三层 CTR 称为 SCD-Net。
18. 不要把“Iter 0 尚未打印”直接判定为死锁；先看 GPU、进程和分支计时。
19. 不要认为减少 Python forward 调用就等于减少 FLOPs。
20. 不要认为分阶段 backward 会加速；它主要省峰值显存，可能因重算更慢。
21. 不要默认启用 AMP、降低 batch、Tiny CTR、CPU offload 或全量 checkpoint；这些不是
    用户当前确认的正式协议。
22. 不要直接使用未正确同步 Sinkhorn/queue 的 DataParallel 做多卡正式实验。
23. 不要擅自停止、删除、覆盖服务器任务或数据。
24. 不要提交 `__pycache__`、`.pyc` 或服务器生成数据。
25. 本地 MV4/CTR 配置目前写 device1；服务器成功使用 device0。每次启动都重新确认。

## 13. 新会话接手后的第一组动作

先在本地只读：

```powershell
git status --short
git log -10 --oneline
git diff --check
```

然后向用户报告以下三点，不要立即启动训练：

```text
1. 已读 handoff，当前主结果是 Q4=79.98、MV4=79.47、Q0=77.22。
2. 现代 AimCLR+OSE 已在本地重构，尚需服务器动态测试和 A0/A1/A2 正式实验。
3. 需要先确认服务器 CTR 进程是否仍在跑，以及下一张可用 GPU。
```

若用户让继续代码工作，先审查并验证当前未提交的现代 AimCLR+OSE；若用户让继续实验，
按 A0、A1（MV4 M-F）、A2（Q4 M-F）补齐 AimCLR 行。任何完整 300-epoch 任务启动前都要
让用户知道预计时长和使用的设备、配置、work_dir。
