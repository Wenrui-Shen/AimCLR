# AimCLR / ReSA / OSE 实验交接

> 面向一个完全没有上下文的新会话。最后更新：2026-07-21。
>
> 接手后请先完整阅读本文，再检查代码、配置和服务器状态。不要从旧聊天标题、旧目录名或
> checkpoint 名字猜实验协议，也不要未经用户确认启动、停止或覆盖服务器任务。

## 0. 最重要的结论和最新决定

我们正在研究 OSE（One-Shot Exemplar）能否作为骨架自监督学习的即插即用类别空间模块：
每个动作类别只使用一个固定带标签 exemplar 构造类别原型，让无标签样本在预训练期间接触
类别空间，再用正式 linear evaluation 衡量表征质量。

已经得到的核心结果（NTU60 xsub joint，ST-GCN，weak+weak，dropout0，pretext300，LP200）：

```text
ReSA + OSE M-F
Q8:  78.80
Q4:  79.98
Q0:  77.22
MV4: 79.47

现代 AimCLR + OSE MV4 M-F
LP: 约 72.56
```

`Q4` 表示一个在线 exemplar anchor 加 4 个无标签 OSE queue 邻居；`MV4` 表示一个在线
弱增强 exemplar 视图加 4 个 EMA 弱增强视图，总 prototype component 数都为 5。

ReSA 中 MV4 比 Q0 高 2.25，说明多弱视图补回了大部分有效变化；MV4 只比 Q4 低 0.51，
单 seed 不足以证明差距稳定。现代 AimCLR+OSE MV4 M-F 跑出约 72.56，明显不理想，但
当前尚没有同协议 A0（原版 AimCLR）的正式结果，不能精确声称“比 A0 降了多少”。

目前的工作假设是：AimCLR 的 instance queue 把所有历史非配对样本当负样本，其中存在
同类别假负样本；OSE 又希望共享 encoder 暴露类别聚集结构。单纯把 OSE loss 叠到 AimCLR
上可能产生实例均匀化与类别聚集之间的目标冲突。把 OSE 从 epoch 1 启用还是稍后启用不是
主要解释，因为 ReSA+OSE 也是从 epoch 1 启用且效果良好。

### 用户刚刚确认的下一步

不继续优先做 AimCLR A0/A1/A2，也暂不继续 CTR-GCN；下一步考虑并实现：

> **ST-GCN ReSA Q4+M-F + OSE 类别修正的双 weak instance queue 对比学习。**

新增 queue 对比 loss 的权重固定为：

```yaml
queue_contrast_weight: 1.0
```

不要把它误写成“四个标量 loss”。ReSA Q4+M-F 原本已经有四项：

```text
Lcluster + Lproto + Lmix-proto + Lmix-ins
```

加入修正 queue 对比后，实际总目标是五个标量项：

```text
Ltotal = Lcluster
       + 1.0 * Lproto
       + 1.0 * Lmix-proto
       + 1.0 * Lmix-ins
       + 1.0 * Lqueue-corr
```

若把 `Lmix-proto + Lmix-ins` 合称 M-F，可以口头称“四组目标”，但代码、日志和论文描述
必须明确实际是五项 loss。

## 1. 工作区、仓库和统一实验协议

```text
本地仓库：D:\Program\codex\program\AimCLR
服务器：/home/user9/public3/swr/AimCLR
服务器 conda 环境：swr_aimclr
Dataset：NTU60
Protocol：xsub
Stream：joint
输入：[N, 3, 50, 25, 2]
Backbone：当前正式主线使用 ST-GCN
Pretext：300 epochs
Linear evaluation：200 epochs
Batch size：128
增强：weak + weak
Dropout：0
正式训练：单 GPU
```

服务器数据、日志和 checkpoint 不在本地仓库。配置中的 `device` 不代表服务器当前空闲卡，
每次运行前都必须重新确认。

在本次 handoff 编辑之前，工作区是干净的，最新提交为：

```text
c24df06 aimclr change
5dad59f epoch change
46030cf aimclr rechange
2d584eb ose proto change and ctrgcn test
93438d2 ctrgcn layer
9bcbf9d ctrgcn
676f680 Lmix
a98e33d OSEchange
```

`c24df06` 已包含 AimCLR queue 原地修改导致 backward 失败的修复和相应测试。本次只修改
`handoff.md`；不要误以为现代 AimCLR+OSE 仍是未提交代码。

本地 Python 没有可用的 `torch`，只能做语法/静态检查；动态测试要到服务器环境运行。
没有实际看到服务器测试输出之前，不要声称动态测试已通过。

统一术语：

```text
ReSA-only = Lcluster
B0        = Lcluster + Lproto
M-P       = B0 + Lmix-proto
M-F       = B0 + Lmix-proto + Lmix-ins
```

不要用含糊的 “proto-only” 指代不同 loss 组合。

## 2. 已有正式结果和仍缺失的结果

以下 ST-GCN 结果均为 NTU60 xsub joint、weak+weak、dropout0、batch128。正式横向比较使用
pretext300 和 LP200。

| 版本 | Pretext checkpoint | OSE top-k | exemplar views | LP Top-1 |
|---|---:|---:|---:|---:|
| ReSA B0 | 120 | 8 | 1 | 74.39 |
| ReSA B0 | 300 | 8 | 1 | 75.95 |
| ReSA M-P | 300 | 8 | 1 | 78.20 |
| ReSA M-F | 300 | 8 | 1 | 78.80 |
| ReSA M-F | 300 | 4 | 1 | **79.98** |
| ReSA M-F | 300 | 0 | 1 | **77.22** |
| ReSA M-F / MV4 | 300 | 0 | 5 | **79.47** |
| AimCLR + OSE MV4 M-F | 300 | 0 | 5 | **约 72.56** |

已确认的 ReSA 差值：

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

必须牢记：

- Q0=77.22 是单个在线 exemplar 视图，不是 MV4。
- Q4 和 MV4 都有 5 个 prototype components，但来源不同。
- AimCLR 约 72.56 使用的是现代重构后的 MV4 M-F，不是旧约 66 的实现。
- 同协议 A0（原版 AimCLR）结果尚未在当前会话中报告，因此不能定量写 A1 相比 baseline
  下降多少；用户主观确认它表现偏低，但论文结论仍需 A0。
- 用户跑过 CTR-GCN epoch120 的 LP 并感觉偏低，但准确 Top-1 没有在当前会话记录，绝对
  不要猜数字。
- 当前唯一明确记录的 ST-GCN epoch120 数字是 ReSA B0 Q8 的 74.39；没有 ST-GCN
  Q4 M-F epoch120 的已知结果。
- 不得混用不同 total-epoch cosine schedule 下的同名 epoch checkpoint。

## 3. 已经完成的代码工作

### 3.1 ReSA+OSE 和 MV4

关键文件：

```text
net/ose_resa.py
processor/pretrain_ose_resa.py
feeder/ntu_feeder.py
tests/test_ose_resa_lmix.py
```

已实现：

- ReSA 的 `Lcluster`。
- OSE `Lproto`、`Lmix-proto`、`Lmix-ins`。
- Q0/Q4/Q8 的 OSE neighbor prototype。
- MV4：1 个 online weak exemplar + 4 个 EMA weak exemplar views。
- exemplar 从无标签 loader 排除。
- exemplar cache 的 class、index、seed、样本数和标签一致性校验。
- mixed branch、prototype component 和梯度路径测试。

MV4 配置：

```text
config/ntu60/pretext/pretext_ose_resa_lmix_full_mv4_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_resa_lmix_full_mv4_xsub_joint.yaml
```

MV4 的四个额外 EMA exemplar forward 会保存并恢复 teacher BN running mean、running var
和 `num_batches_tracked`，避免同一 iteration 重复污染 EMA BN buffer。

### 3.2 CTR-GCN

`net/ctrgcn.py` 已支持 10-layer、ST-GCN-matched widths：

```yaml
num_layers: 10
layer_channels: [16, 16, 16, 16, 32, 32, 32, 64, 64, 256]
```

这不是官方宽度 CTR-GCN。对应配置：

```text
config/ntu60/pretext/pretext_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml
```

服务器实测约 2.16 秒/iter，完整 pretext300 估计约 58–62 小时，用户认为不可接受。
目前服务器 CTR 进程状态未知；不要擅自停止、续跑或删除。

### 3.3 现代 AimCLR+OSE 重构

关键文件：

```text
net/aimclr.py
net/ose_aimclr.py
processor/pretrain_ose_aimclr.py
tests/test_ose_aimclr.py
```

配置：

```text
config/ntu60/pretext/pretext_aimclr_a0_xsub_joint.yaml
config/ntu60/pretext/pretext_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
config/ntu60/pretext/pretext_ose_aimclr_q4_lmix_full_xsub_joint.yaml

config/ntu60/linear_eval/linear_eval_aimclr_a0_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_aimclr_q4_lmix_full_xsub_joint.yaml
```

重构已完成：

- 保留 AimCLR 原生 128-D instance head 和 32768 queue。
- OSE 使用独立 256-D、3-layer projector 和独立 OSE neighbor queue。
- MV4 的 `ose_topk: 0` 不读也不写 OSE neighbor queue。
- 支持 B0、M-P、M-F 独立 loss 开关。
- exemplar 严格从无标签 loader 排除。
- LP 改为 200 epochs，使用独立 work_dir。
- `ose_warmup_epoch: 0`，从 epoch 1 第一个 batch 启用 OSE；不要恢复“前 20 epoch 只跑
  AimCLR”的旧 schedule。

首次服务器运行曾在 epoch 1 backward 报错：

```text
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation: [1, 128, 32768]
```

原因是负样本 logits 使用 `self.queue.detach()`，它仍与原 queue 共享存储；forward 在
backward 前 enqueue 覆盖了 queue，触发 autograd version mismatch。修复是所有相关负样本
路径都使用：

```python
self.queue.clone().detach()
```

该修复和 backward 测试已提交在 `c24df06`。随后现代 AimCLR+OSE MV4 M-F 已能完成训练
并得到约 72.56 LP。

## 4. 必须保持的 ReSA/OSE 语义

ReSA 当前有三种表示：

```text
H：encoder 特征
Z：projector 特征
Q：predictor 输出
```

- ReSA 的 Sinkhorn assignment 从 encoder `H` 的 batch 关系构造。
- ReSA 跨视图预测使用 online predictor `Q` 对 teacher projector `Z`。
- OSE exemplar、neighbor queue、prototype、teacher target 和 mixed branch 都在 `Z` 空间。
- 不得随意混用 H/Z/Q。若未来做 encoder-space prototype，必须作为独立完整消融。

输入混合为：

```python
mixed_view = beta * view_b + (1.0 - beta) * view_a[mix_index]
```

`mixed_view` 只走 `encoder_q -> projector_q -> normalized mixed_z`；不走 predictor，
不进 Sinkhorn，不进 teacher，不进任何 queue。`beta` 是输入混合权重，probability target
只在 loss 内构造并 detach，不是模型输入。

标签使用边界：

- 每类用 seed0 固定选择一个 labeled exemplar。
- exemplar 必须从无标签训练 loader 中排除。
- exemplar 不能成为普通 queue 样本。
- 真标签只允许用于 exemplar 选择和离线/日志诊断。
- 真标签不能进入模型输入、loss、类别 queue、top-k 或负样本修正。

## 5. 为什么提出“类别修正的 instance queue”

ReSA 的关系学习是当前 batch 上的 B×B soft assignment，每个 batch 重算，不维护用于硬
负样本判定的持久 instance queue。AimCLR/MoCo 风格对比则使用跨 batch 的历史 queue：

```text
当前 online weak query q
配对 EMA weak key k+
历史 EMA instance queue {k_j}, j=1...K
```

原始 InfoNCE 把 queue 中所有非配对项都视为负样本，包括真实同类别样本。这会让实例级
目标与 OSE 的类别聚集目标拉扯共享 encoder。

新的想法不是把同类 queue 样本直接提升为正样本，而是使用 OSE 已经产生的软类别分布，
只削弱高置信度疑似同类样本在 InfoNCE 分母中的负作用。这是保守的 false-negative
debiasing。

它与 ReSA soft weight 不同：

```text
ReSA assignment：B×B、batch-level、从无标签 H 相似度经 Sinkhorn 构造、作为软预测目标。
类别修正权重：B×K、跨 batch、由 exemplar-anchored OSE 类别分布构造、只缩放负样本分母。
```

## 6. 下一步要实现的精确方案

### 6.1 基础模型

基础必须是已有最佳点：

```text
ST-GCN + ReSA Q4 + M-F
LP200 baseline = 79.98
```

Q4 语义必须保持：

```text
ose_topk: 4
ose_exemplar_views: 1
ose_lambda: 1.0
ose_mix_proto_weight: 1.0
ose_mix_ins_weight: 1.0
```

注意：本地 `pretext_ose_resa_lmix_full_xsub_joint.yaml` 当前是 Q8，不是 Q4；本地尚无
独立的 ST-GCN Q4 M-F 配置。不要直接改写或复用 Q8 work_dir。实现时要新建名称清晰的
Q4+queue-correction pretext/LP 配置和独立 work_dir。

### 6.2 只增加双 weak 对比，不搬完整 AimCLR

本实验只增加一个 MoCo 风格双 weak InfoNCE 分支：

```text
online weak view_a -> shared encoder_q -> independent 128-D instance projector_q -> q
EMA weak view_b    -> shared encoder_k -> independent 128-D instance projector_k -> k+
```

推荐的首版就是单向 `q(view_a)` 对 `k(view_b)`，每个 iteration 入队 `k(view_b)`。不要在
首版自行改成双向平均或同时入队两套 key，否则会改变 loss 权重、queue 周转速度和计算语义。

必须复用 ReSA 已经算出的 weak backbone features，不增加 ST-GCN backbone forward。
新增的只是独立 instance projector 和 queue logits。EMA instance projector 由 online
instance projector 初始化并随现有 momentum 机制更新。

明确不加入：

- extreme view；
- dropped extreme branch；
- DDM；
- AimCLR NNM/mining epoch；
- AimCLR 的整套 processor。

### 6.3 三个严格对齐的 queue

新增三个 buffer，共用一个 pointer：

```text
instance_queue:   [128, 32768]  # 历史 EMA instance embedding
category_queue:   [60, 32768]   # 同一历史样本的 OSE 软类别分布
confidence_queue: [32768]       # 同一历史样本的类别置信度
instance_queue_ptr
```

同一个 slot 的 instance feature、category distribution 和 confidence 必须来自同一个样本，
入队和 wrap-around 覆盖必须完全同步。

这三个 buffer 属于新的 instance contrast 分支。它们与 ReSA+OSE 原有的 8192-D（容量）
OSE neighbor queue 是不同用途的 queue：

```text
OSE neighbor queue：给 Q4 prototype 检索 4 个邻居。
instance queue：给 InfoNCE 提供 32768 个历史候选。
category/confidence queue：instance queue 的语义 sidecar。
```

绝对不要合并 pointer、容量或张量，也不要让一个 queue 冒充另一个。

### 6.4 类别信息来源

对当前 EMA weak key，复用 Q4 OSE 已经计算出的 teacher category target：

```text
p_i in R^60, sum(p_i)=1
```

不要增加额外 backbone forward，不要用 ground-truth label，也不要用 hard pseudo label。
`p_i` 在 queue correction 路径中必须 detach。

置信度使用归一化熵：

```text
c_i = 1 - H(p_i) / log(60)
```

- 接近 one-hot：`c_i -> 1`。
- 接近均匀：`c_i -> 0`。

### 6.5 负样本权重

当前样本与第 j 个 queue 样本的类别相似度：

```text
s_ij = p_i^T p_j
```

负样本权重：

```text
w_ij = 1 - c_i * c_j * s_ij
```

所有 `p`、`c`、`w` 在该修正分支 detach：

- 双方高置信且类别相同：`w_ij` 接近 0，显著削弱假负样本。
- 类别不同：`w_ij` 接近 1。
- 任一侧不确定：`w_ij` 接近 1，退化为原始 InfoNCE，不贸然修正。

不做 hard threshold，不把同类项直接提升为 positive。

### 6.6 修正后的 InfoNCE

原始 logits：

```text
l_pos = q_i^T k_i+ / T
l_neg_ij = q_i^T instance_queue_j / T
```

只修改 negative logits：

```python
negative_logits = negative_similarity / temperature
negative_logits = negative_logits + torch.log(
    negative_weight.clamp_min(1e-6)
)
```

这等价于：

```text
exp(l_pos) + sum_j w_ij * exp(l_neg_ij)
```

必须先除温度，再加 `log(w)`；positive logit 不加权。`w` 不能在 temperature 之前乘到
cosine similarity 上，那不是同一个目标。

queue 初始状态：

```text
category_queue = 每列均匀 1/60
confidence_queue = 0
```

于是初始 `w=1`，严格退化为原始 queue InfoNCE。类别修正从 epoch 1 开始，但随着 OSE
置信度自然增强；不设置额外 hard warmup。

计算当前 logits 时必须读取“入队前”的历史 queue；然后再同步写入当前 `k,p,c`。如果
forward 内在 backward 前 enqueue，参与 logits 的 `instance_queue` 必须
`clone().detach()`，不能只 `detach()`。

### 6.7 总 loss 和日志

用户明确指定新增权重为 1：

```text
Ltotal = Lcluster
       + Lproto
       + Lmix-proto
       + Lmix-ins
       + 1.0 * Lqueue-corr
```

至少记录：

```text
total
cluster
proto / align / disp
mix_proto
mix_ins
queue_corr
mean_category_confidence
mean_negative_weight
min_negative_weight
```

诊断日志不能使用真标签参与训练；若离线统计 false-negative purity，必须清楚标注为只读
诊断，不能反馈到 loss。

## 7. 实现与验证清单

下一会话若获用户授权继续代码，应按以下顺序：

1. 先只读审查 `net/ose_resa.py` 和 `processor/pretrain_ose_resa.py` 当前 weak feature、
   teacher target、EMA 更新与 enqueue 顺序。
2. 设计独立 instance projector，不改变原 ReSA/OSE projector、predictor 和 Q4 prototype。
3. 增加 aligned instance/category/confidence buffers 和单一 pointer。
4. 复用现有 OSE teacher target，detach 后计算 confidence 和 B×K negative weights。
5. 实现单向 weak queue InfoNCE，权重固定为 1。
6. 新建专用 ST-GCN Q4 M-F queue-correction pretext/LP 配置和独立 work_dir；不要改写
   Q8、MV4 或已有 79.98 实验目录。
7. 添加单元测试，再做静态检查。
8. 同步服务器后先跑动态 unit test 和 1–2 iteration smoke test，确认 forward、backward、
   EMA、queue pointer、显存和速度。
9. 只有用户确认 GPU、预计时长、配置和 work_dir 后，才启动 pretext300；随后 LP200。

必须覆盖的测试：

- 三个 queue 的 slot 对齐与 pointer wrap-around。
- category uniform/confidence0 时，修正 logits 与原始 InfoNCE 完全一致。
- 高置信同类项权重接近 0；高置信异类项权重接近 1。
- positive logit 不被修改。
- `p/c/w` 无梯度，instance query/head/shared encoder 有梯度。
- 当前 batch 不在自身 logits 计算前入队。
- enqueue 后完整 `loss.backward()` 不触发 autograd version mismatch。
- state_dict 保存/加载包含所有 projector、queue 和 pointer。
- 不使用 ground-truth label。
- backbone forward 次数不因新增 queue 分支增加。
- 关闭 queue contrast 开关时，数值和原 ReSA Q4+M-F 路径保持一致。

正式比较：

```text
R0：ReSA Q4+M-F                         = 79.98（已有）
R2：ReSA Q4+M-F + corrected weak queue = 待跑
```

若 R2 提升，只能先说明“加入类别修正 queue 的整个分支有益”。要严格证明“修正本身”有益，
后续仍需控制组：

```text
R1：ReSA Q4+M-F + raw weak queue（w=1）
```

比较 `R2-R1` 才能隔离 category correction 的因果贡献。当前用户决定先考虑 R2，不要未经
确认自动增加另一场 300-epoch R1。

## 8. 当前卡点

1. **新的 ReSA queue-correction 分支尚未实现。** 当前只有设计，没有对应代码、配置、
   测试或服务器结果。
2. **本地缺少专用 ST-GCN Q4 M-F 配置。** 现有普通 `pretext_ose_resa_lmix_full...`
   是 Q8；必须新建配置，不能原地修改造成实验身份混乱。
3. **本地没有 torch。** 动态 backward、queue wrap 和显存测试必须在服务器执行。
4. **AimCLR A0 结果未报告。** 约 72.56 的下降幅度无法精确量化。
5. **CTR epoch120 LP 准确数字未记录，完整 CTR 训练又过慢。** 当前不应把它当主线。
6. **服务器当前进程和 GPU 状态未知。** 新会话不要假设某张卡空闲。

## 9. 绝对不要再踩的坑

1. 不要说“top-k 越低越好”；Q0=77.22 已经反证。
2. 不要把 Q0 单 exemplar 当成 MV4 多弱增强。
3. 不要把新方案说成四个标量 loss；实际是五项，只有分组口径才是四组。
4. 不要把完整 AimCLR 搬到 ReSA：首版不含 extreme、drop、DDM、NNM 或 mining。
5. 不要擅自改成双向 InfoNCE或每步入队两套 weak key；首版是单向 q(view_a)-k(view_b)。
6. 不要让新增 instance head 与 ReSA/OSE projector 强行共用；它是独立 128-D head。
7. 不要增加新的 backbone forward；复用 ReSA 已经算出的 weak H/features。
8. 不要把 OSE neighbor queue、instance queue 和 category sidecar queue 混为一谈。
9. 不要让三个 sidecar queue 的 pointer 或覆盖位置错位。
10. 不要用 hard pseudo label；存 60 维 soft probability 和 entropy confidence。
11. 不要让 category weight 对 OSE 反传；`p/c/w` 必须 detach。
12. 不要修改 positive logit；类别权重只作用于 queue negative denominator。
13. 不要把 `w` 乘在 cosine similarity 上；正确实现是温度之后加 `log(w)`。
14. 不要忘记 `clamp_min(1e-6)`，否则 `log(0)` 会产生无穷。
15. 不要在计算当前 logits 前把当前 key 入队。
16. 不要再次使用 `queue.detach()` 后原地覆盖同一存储；必须 clone 或延后 enqueue，否则
    backward 会触发 version mismatch。
17. 不要使用真标签修正负样本；标签只用于 exemplar 选择和只读诊断。
18. 不要让 exemplar 回到无标签 loader、instance queue 或 OSE neighbor queue。
19. 不要混用 H/Z/Q 空间中的 prototype、teacher target 或 mixed embedding。
20. 不要让 mixed branch 进入 predictor、Sinkhorn、teacher 或任何 queue。
21. 不要恢复前 20 epoch 只跑 baseline；当前协议从 epoch 1 启用所有配置目标。
22. 不要把 warmup 当作 AimCLR 低结果的唯一解释；ReSA 从 epoch 1 启用 OSE 仍有效。
23. 不要拿旧 LP100 约 66 与当前 LP200 结果直接比较。
24. 不要在没有 A0 的情况下精确声称 72.56 比 baseline 下降多少。
25. 不要混用不同 total-epoch cosine schedule 的 checkpoint。
26. 不要复用 work_dir 覆盖 Q4=79.98、MV4=79.47 或其他旧实验。
27. 不要把 `weights + start_epoch` 当作完整 resume；optimizer、scheduler、EMA 和 queue
    状态可能都没有恢复。
28. 不要把 ST-width CTR 称为官方宽度 CTR，也不要把本项目三层 CTR 称为 SCD-Net。
29. 不要把日志暂未打印 `Iter 0 Done` 直接判为死锁；先检查 GPU、进程和分支耗时。
30. 不要认为减少 Python forward 次数等于减少 FLOPs，也不要认为分阶段 backward 会加速。
31. 不要直接用未同步 Sinkhorn/queue 的 DataParallel 做正式多卡实验。
32. 不要擅自停止、删除、覆盖服务器进程、checkpoint、work_dir 或数据。
33. 不要提交 `__pycache__`、`.pyc` 或服务器生成数据。
34. 每次运行前重新确认 `device`；本地配置值不代表服务器空闲 GPU。

## 10. 已知命令（仅供核对，不代表下一步优先级）

现代 AimCLR A0：

```bash
python main.py pretrain_aimclr --config config/ntu60/pretext/pretext_aimclr_a0_xsub_joint.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_a0_xsub_joint.yaml
```

现代 AimCLR+OSE MV4 M-F（已得到约 72.56）：

```bash
python main.py pretrain_ose_aimclr --config config/ntu60/pretext/pretext_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_ose_aimclr_mv4_lmix_full_xsub_joint.yaml
```

CTR-GCN 10-layer ST-width Q4 M-F epoch120 LP 的已知一行命令：

```bash
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_ose_resa_ctrgcn10_stwidth_q4_lmix_full_xsub_joint.yaml --weights ./data/ntu60_cs/ose_resa_ctrgcn10_stwidth_q4_lmix_full_joint/pretext/epoch120_model.pt --work_dir ./data/ntu60_cs/ose_resa_ctrgcn10_stwidth_q4_lmix_full_joint/linear_eval_epoch120
```

新的 ReSA Q4+M-F+corrected queue 配置尚不存在。不要伪造可运行命令；实现并验证配置后再
给用户正式 pretext/LP 一行命令。

## 11. 新会话接手后的第一组动作

先在本地只读：

```powershell
git status --short
git log -10 --oneline
git diff --check
```

然后完整审查：

```text
net/ose_resa.py
processor/pretrain_ose_resa.py
tests/test_ose_resa_lmix.py
config/ntu60/pretext/pretext_ose_resa_lmix_full_xsub_joint.yaml
```

向用户报告：

```text
1. 已读 handoff，已有主结果 Q4=79.98、MV4=79.47、Q0=77.22、AimCLR+MV4 M-F≈72.56。
2. 当前决定是实现 ReSA Q4+M-F + 类别修正双 weak instance queue，新增权重为 1。
3. 该分支目前尚未实现；会先完成代码、配置、测试和短 smoke，不会擅自启动长训练。
```

若用户授权实现，严格按第 6、7 节执行。任何完整 pretext300 启动前，都要让用户知道：

```text
使用的 GPU
配置文件
work_dir
预计时长
是否从头训练
```
