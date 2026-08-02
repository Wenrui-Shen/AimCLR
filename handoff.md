# AimCLR / ReSA / OSE 研究交接

最后更新：2026-08-02。本文写给一个完全没有上下文的新会话；不要依赖旧聊天记录。

## 0. 2026-08-02 A3 二阶段主线覆盖说明

A2 线性评估只有 `73.79`，说明在同一阶段把 OSE 类别收缩与 AimCLR
memory-bank instance discrimination 强行合并仍然存在明显目标冲突。当前新增 A3：

```text
Stage1: 完全默认 AimCLR (`pretext_aimclr_xsub_joint.yaml`), 300 epochs
Stage2: ReSA + OSE P1 Q4 M-F, 300 epochs
```

Stage2 的具体过渡语义：

- 从 AimCLR checkpoint 的 `encoder_q` 加载 backbone；
- 把 AimCLR `encoder_q.fc` 的 `256 -> 256 -> 128` MLP 原样迁移为
  Stage2 的 ReSA/OSE 共用 projector；
- online 参数加载完成后复制到 EMA 分支，Stage2 起点两支严格一致；
- 不迁移 AimCLR queue、NNM、DDM 或任何 memory-bank 对比损失；
- ReSA predictor 在 A3 首版关闭，避免引入随机 head；
- 在第一次优化前，用迁移后的 EMA encoder-projector 和固定 seed 的 weak
  views 离线填满 OSE queue，并排除 one-shot exemplars；
- Stage2 只优化 `LReSA + Lproto + Lmix-proto + Lmix-ins`；
- prototype 固定使用最佳 P1 互斥 Q4，其他 P0/P2/P3 配置不恢复。

新增入口和配置：

```text
processor/pretrain_ose_resa_stage2.py
config/ntu60/pretext/pretext_ose_resa_a3_stage2_p1_q4_mf_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_resa_a3_stage2_p1_q4_mf_xsub_joint.yaml
```

正式运行：

```bash
python main.py pretrain_ose_resa_stage2 \
  --config config/ntu60/pretext/pretext_ose_resa_a3_stage2_p1_q4_mf_xsub_joint.yaml

python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_resa_a3_stage2_p1_q4_mf_xsub_joint.yaml
```

## 0.1 2026-07-29 A2 历史主线说明

本节记录 A2 当时的覆盖指令；现已被上面的 A3 二阶段方案取代。

新增正式结果：

```text
P0 = 78.79
P1 = 79.75
P2 = 78.15
P3 = 78.80
```

P1 仍是可靠 prototype 阶段的最佳版本；继续修改 prototype 聚合没有超过 P1。A2 当时尝试在 AimCLR 原生正负关系中解决 OSE 与实例对比学习的冲突：

```text
A2 = AimCLR
   + native 128-d AimCLR/OSE shared embedding
   + one shared AimCLR EMA key queue
   + P1 mutually-exclusive Q4 neighbors
   + Lproto + Lmix-proto + Lmix-ins
   + OSE-constrained NNM in weak-query/weak-key contrast
```

关键实现语义：

- 不再使用独立 `ose_projector` 或 `ose_queue`；
- exemplar、queue、prototype、weak contrast 和 mix 全部使用 AimCLR 原生归一化 head 输出；
- P1 先为每个已填充 queue 槽位确定唯一 owner，再为每类取 Top-4；
- P1 邻居既参与 prototype，也构成受 OSE 约束的 NNM 候选池；
- EMA weak key 对 P1 prototype 做 `argmax`，只启用预测类别对应的候选池；
- AimCLR 的 normal/extreme/dropped-extreme 三路相似度在该池内取最大值，只选一个额外 queue 正样本；
- 若候选池为空则退化为原始配对正样本；其余所有 queue 项仍是负样本；
- 同一原始样本的历史 queue 条目在选择前排除；
- A2 用受约束 NNM 替代原生全局 NNM，不做二者并集；不再存在 soft-positive target 或 `ose_positive_weight`；
- OSE 只读取旧 queue，当前 EMA key 在所有 target/loss 完成后只 enqueue 一次；
- queue 的 `sample_indices` 是同一个 feature queue 的元数据 sidecar，用于同样本过滤和诊断，不是第二个 queue；
- `mining_epoch=150` 是唯一阶段开关：epoch 150 及以前为 AimCLR，epoch 150 以后同时启用 P1 prototype、受约束 NNM、Lproto 和两项 Lmix。

A2 阶段配置目录只保留了两份 OSE 配置：

```text
config/ntu60/pretext/pretext_ose_aimclr_a2_p1_q4_mf_xsub_joint.yaml
config/ntu60/linear_eval/linear_eval_ose_aimclr_a2_p1_q4_mf_xsub_joint.yaml
```

原 ReSA/OSE P0–P3 的八份 pretext/linear-eval 配置已按用户要求删除；`net/ose_resa.py`、processor 和测试代码仍保留为历史实现与参考，不代表当前实验入口。

正式运行：

```bash
python main.py pretrain_ose_aimclr \
  --config config/ntu60/pretext/pretext_ose_aimclr_a2_p1_q4_mf_xsub_joint.yaml

python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_aimclr_a2_p1_q4_mf_xsub_joint.yaml
```

## 1. 我们正在做什么

我们在研究一个用于骨架自监督预训练的 OSE（One-Shot Exemplar）类别空间模块：

- 每个动作类别只使用一个固定、带标签的 exemplar；
- 其余预训练样本不使用标签；
- 当前基础 SSL 方法是 ReSA，backbone 是 ST-GCN；
- 目标是在实例级 SSL 上加入可靠的类别语义，并判断类别关系能否反过来改善 ReSA 的实例关系学习。

这个设定不能称为“完全无监督”，应称为：

- one-shot-assisted self-supervised learning；或
- label-efficient self-supervised learning。

当前研究主线分成两个问题：

1. **可靠类别原型**：单 exemplar 覆盖不足；加入 queue 邻居后，又存在跨类重复、污染、稀释和陈旧特征问题。
2. **类别信息的完整利用**：当前 OSE 类别信息只用于 projector 空间的 prototype loss，ReSA 在 encoder 空间构造的 Sinkhorn B×B 关系尚未使用 OSE 软类别关系。

已经确定的总体方案是：

```text
可靠 Q4 类别原型
    +
OSE soft relation 引导 ReSA assignment（尚未实现）
```

用户已明确否决置信度路线。不要再加入 entropy confidence、JS confidence/gate、阈值、置信度 queue 或基于置信度的动态 K。

## 2. 当前方法与表示空间

### 2.1 H / Z / Q 不得混用

```text
H = encoder/backbone feature
    ReSA 在 H 上构造 B×B 相似关系和 Sinkhorn assignment

Z = projector feature
    OSE exemplar、queue、prototype、类别 target 和 M-F 都在 Z 上

Q = predictor output
    用作 ReSA online 跨视图预测
```

维度即使相同，也不能把 predictor 输出拿去建 OSE queue，或用 projector feature 偷换 ReSA 原本的 encoder 关系。

### 2.2 ReSA 确实使用 EMA Encoder

论文 Figure 5 只画了概念信息流，把两条分支都简写成 `Encoder`，没有完整展示参数更新。论文附录和官方仓库实际实现包含：

- online encoder/projector；
- momentum/EMA encoder/projector；
- EMA 参数更新；
- teacher 分支 `no_grad`。

原始双视图 ReSA 每轮通常有 4 次 backbone 前向：

```text
online view_a + online view_b
EMA view_a    + EMA view_b
```

当前 OSE Q4 M-F、`ose_exemplar_views=1` 增加 online exemplar 和 mixed view，因此每轮为：

```text
online：view_a、view_b、exemplar、mixed_view = 4 次
EMA：   view_a、view_b                         = 2 次
合计：                                           6 次
```

P0–P3 只改变 prototype 内部算法，不改变前向次数。

### 2.3 View 增强协议

ReSA/OSE 使用专用 feeder：`feeder/ose_resa_feeder.py`。

默认按顺序遍历：

```text
temporal_crop -> shear -> rotation
```

每个增强对每个 view 独立以 `p=0.5` 触发。不是“随机选择其中一个”，而是逐项 Bernoulli，因此一个 view 可以组合 0–3 个增强。

两个无标签 view 和每个 exemplar view 都重新独立采样。

### 2.4 当前完整损失

```text
L = Lcluster(ReSA)
  + Lproto
  + Lmix-proto
  + Lmix-ins
```

当前四部分权重均为 1。

- `Lcluster`：原 ReSA 跨视图关系预测。
- `Lproto`：online view_b 对 OSE prototypes 的类别分布，匹配 EMA view_a 的 detached 类别 target，并包含 prototype dispersion。
- `Lmix-proto`：mixed sample 匹配两端样本类别分布的 beta 插值。
- `Lmix-ins`：mixed sample 在当前 EMA batch 中同时匹配原位置和 permutation 位置。

mixed branch：

- 只走 online encoder-projector；
- 不走 predictor；
- 不参与 Sinkhorn；
- 不进入 queue。

### 2.5 Queue 更新顺序

必须先使用旧 queue 完成 prototype、target、logits 和 loss，再 enqueue 当前 batch 的 EMA `Z`：

```text
read old queue
-> build prototypes/targets/losses
-> enqueue current teacher features
```

不能提前 enqueue，否则当前样本会检索到自身，造成同批次泄漏。

## 3. P0–P3 可靠原型阶段

所有阶段固定 Q4、M-F、exemplar seed0 和同一正式协议；每一步只改一个因素。

邻居竞争分数：

```text
g[c,j] = alpha * sim(anchor_c, z_j)
       - (1-alpha) * max_{d != c} sim(anchor_d, z_j)

alpha = 0.75
```

它是排序分数，不是置信度，也没有通过/拒绝阈值。

### P0：当前 Q4 基线

- 每类独立 Top-4；
- 同一个 queue 样本可以进入多个类别；
- components 为 1 个 online exemplar + 最多 4 个 EMA queue 邻居；
- `g[c,j]` 只用于选择；
- 聚合权重用 raw `sim(anchor, component)` softmax；
- 聚合后不重新归一化。

### P1：互斥邻居分配

先让每个 queue 样本只属于一个类别：

```text
owner(j) = argmax_c g[c,j]
```

类别 `c` 只在 `owner(j)=c` 的候选中取 Top-4。同一样本不会进入多个类别；候选不足时只使用已有候选和 exemplar，不从其他类别强行补齐。

P1 与 P0 的唯一目标差异是互斥分配。

### P2：alpha-consistent aggregation

在 P1 上，让选择与聚合使用同一个竞争分数：

```text
g_anchor[c] = alpha * 1
            - (1-alpha) * max_{d != c} sim(anchor_d, anchor_c)

w = softmax([g_anchor, selected_neighbor_scores])
prototype = sum_i w_i * component_i
```

首版 softmax 温度固定 1，不新增超参。

### P3：最终 prototype normalization

在 P2 上只增加：

```text
prototype = normalize(prototype)
```

如果 P3 下降，不要为了形式统一强留归一化；prototype norm 可能携带内部一致性信息。

## 4. 已完成的实现

当前 Git 状态基线：

```text
HEAD = c3f6bf4 update
branch = main
origin/main = c3f6bf4
```

已完成：

1. `net/ose_resa.py`
   - online/EMA encoder 和 projector；
   - ReSA cluster loss；
   - OSE queue 和 sample-index sidecar；
   - P0/P1/P2/P3，配置项 `ose_prototype_stage: 0/1/2/3`；
   - `Lproto`、`Lmix-proto`、`Lmix-ins`；
   - corrected instance queue 的历史探索代码仍保留，但 P0–P3 默认关闭。

2. `processor/pretrain_ose_resa.py`
   - 固定 exemplar 的选择、缓存和校验；
   - exemplar 从无标签 loader 中排除；
   - cosine LR 和 EMA momentum；
   - loss 组合、queue 时序和诊断日志；
   - purity、overlap、component count 等只读诊断。

3. `feeder/ose_resa_feeder.py`
   - 三项增强依次遍历；
   - 每项独立 p=0.5；
   - 两个 view 独立采样；
   - 可返回原数据集 sample index。

4. 测试文件
   - `tests/test_ose_resa_prototypes.py`；
   - `tests/test_ose_resa_lmix.py`；
   - `tests/test_ose_resa_queue_corr.py`。

5. 配置只保留 P0–P3：
   - `config/ntu60/pretext/pretext_ose_resa_p{0,1,2,3}_q4_mf_xsub_joint.yaml`；
   - `config/ntu60/linear_eval/linear_eval_ose_resa_p{0,1,2,3}_q4_mf_xsub_joint.yaml`。

旧 ReSA-only、CTR、MV4、corrected queue 和 OSE+AimCLR 的专用配置已经删除。原始 AimCLR 代码和原始 AimCLR 配置没有因为这条新主线而修改。

6. 新增迁移文档
   - `BASELINE_MIGRATION_GUIDE.md`；
   - 详细说明如何把本项目创新迁移到另一套 baseline；
   - 当前是未跟踪文件，交接时 `git status` 显示 `?? BASELINE_MIGRATION_GUIDE.md`，不要误删。

## 5. 已有实验结果

### 5.1 当前新增强正式协议

统一协议：

```text
NTU60 xsub joint
ST-GCN
temporal_crop/shear/rotation，各自独立 p=0.5
dropout=0
batch=128
pretext=300 epochs
linear evaluation=200 epochs
exemplar seed=0
```

| 方法 | LP Best Top-1 | 状态 |
|---|---:|---|
| P0：Q4 M-F | 78.79 | 已完成 |
| P1：P0 + 互斥邻居 | **79.75** | 已完成，P1-P0=+0.96 |
| P2 | 待运行 | 尚无结果 |
| P3 | 待运行 | 尚无结果 |

P1 的 +0.96 是当前最重要的新结果，但目前只有 exemplar seed0，不能提前宣称稳定提升。

用户观察到 P0/P1 的 linear-eval 差异更多出现在早期，后期/last acc 接近。当前解释是：P1 可能让特征更容易被线性头优化，但最终线性可分上限接近。正式记录必须同时包含：

- early acc；
- best acc；
- best epoch；
- last acc。

当前 LP 配置冻结 backbone，只训练分类 `fc`；`base_lr=3`，`step=[80]`，训练 200 epoch，最后学习率仍为 0.3。因此 last acc 不一定比 best 更能代表性能。

### 5.2 历史旧协议结果

旧统一协议为 weak+weak，其结果不能与新增强 P0/P1 做严格单因素差值：

| 方法 | LP Top-1 |
|---|---:|
| ReSA + OSE M-F，Q0 | 77.22 |
| ReSA + OSE M-F，Q4 | **79.98** |
| ReSA + OSE M-F，Q8 | 78.80 |
| ReSA + OSE M-F，MV4 | 79.47 |
| ReSA + OSE Q4 M-F + corrected instance queue | 77.44 |
| AimCLR A0 | 75.33 |
| AimCLR + OSE MV4 M-F | 约 72.56 |

能安全得到的结论：

- Q0 说明单 exemplar 覆盖不足；
- Q8 不如 Q4，说明更多邻居不一定更好；
- 不能把 Q4>Q8 直接解释成“Q4 purity 更高”；
- corrected instance queue=77.44，是负结果，不进入主线；
- 直接把 OSE/M-F 叠加到 AimCLR 没有成功，迁移到新 baseline 需要先理清表示和 teacher/queue 语义。

### 5.3 CTR-GCN 中断结果

10-layer CTR-GCN（ST-GCN-matched widths）在 pretext epoch190/300 因断电中断。使用 `epoch190_model.pt` 做 LP200 得到：

```text
76.15
```

这不是完整预训练结果，不能与 ST-GCN 的 300 epoch 结果做正式 backbone 差值结论，也不能把 `weights + start_epoch` 当作完整恢复。

## 6. P1 purity 出现 n/a 的含义

日志中的 per-class `purity n/a` 不是数值 NaN，而是该类别没有有效 queue 邻居，purity 分母为 0。

P1 中每个 queue 样本只能归属于一个 owner，所以某些类别可能：

- 在 queue 冷启动时没有候选；
- 竞争不到任何样本；
- 只有 exemplar，没有 queue component。

此时训练不会报错，prototype 仍至少包含 exemplar。

解释标准：

- 只在第一批或训练初期出现：正常；
- 个别类别偶尔出现：P1 互斥分配的预期退化；
- queue 已满后某些类别连续多个 epoch 为 n/a：类别饥饿，需要检查 exemplar、竞争分数和 component count；
- 大量类别长期 n/a：不能忽略，P1 可能过度集中分配。

`n/a` 与 `purity=0` 不同：后者表示选到了邻居但全部标签错误。

## 7. 当前卡在哪里

当前没有已知代码崩溃阻塞，主要卡在正式实验和下一模块尚未完成：

1. P0、P1 已完成；P2、P3 尚未运行。
2. 尚未根据 P0–P3 选出最终可靠原型 `P*`。
3. P0/P1 只有 exemplar seed0，稳定性未知。
4. P1 的 per-class `n/a`/component shortage 需要结合完整日志判断是冷启动还是长期类别饥饿。
5. OSE-guided Semantic ReSA 尚未实现。
6. 不确定三组单元测试是否已在服务器上完整运行；虽然 P0/P1 已完成端到端训练，下一会话仍应先补跑测试并保存结果。
7. 本地新建的 `BASELINE_MIGRATION_GUIDE.md` 尚未纳入 Git。

## 8. 下一步计划

### 第一步：保护当前工作区

新会话开始先运行：

```bash
git status --short
git diff --check
```

预期至少看到本次更新的 `handoff.md` 和未跟踪的 `BASELINE_MIGRATION_GUIDE.md`。不要删除、覆盖或回退它们。

### 第二步：服务器测试和 P1 日志核对

在有 torch 的环境运行：

```bash
python -m unittest \
  tests.test_ose_resa_prototypes \
  tests.test_ose_resa_lmix \
  tests.test_ose_resa_queue_corr
```

同时检查 P1 日志：

- queue 何时填满；
- `neighbor_overlap` 是否为 0；
- 平均 `components`；
- 哪些类别长期 `purity n/a`；
- P0/P1 early、best、best epoch、last acc。

### 第三步：正式运行 P2

```bash
python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_p2_q4_mf_xsub_joint.yaml

python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_resa_p2_q4_mf_xsub_joint.yaml
```

比较 `P2-P1`，只归因于 alpha-consistent aggregation。

### 第四步：正式运行 P3

```bash
python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_p3_q4_mf_xsub_joint.yaml

python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_resa_p3_q4_mf_xsub_joint.yaml
```

比较 `P3-P2`，只归因于最终 prototype normalization。

正式训练必须从头 pretext300 + LP200。短 smoke 只能排查 forward/backward/queue/显存，不能写进论文结果。

### 第五步：选择 P*

综合：

- LP best、last 和曲线；
- overlap；
- component count；
- per-class purity/n/a；
- 是否出现类别饥饿；
- 训练稳定性。

从 P0–P3 选出最优且稳定的版本记为 `P*`。不要仅依据 0.1–0.2 的单 seed 波动下结论。

### 第六步：实现 OSE-guided Semantic ReSA

设计为：

```text
Pbar = (P_teacher_a + P_teacher_b) / 2       # BxC, detach
G = Pbar @ Pbar.T                            # BxB
S_ins = online_H_a.detach() @ teacher_H_a.T
S_sem = S_ins + lambda_r * (G - 1/C)
A_sem = Sinkhorn(S_sem)
```

关键要求：

- `Pbar/G` 全部 detach；
- 不使用 ground-truth label；
- 不增加 confidence、实例 head 或实例 queue；
- 不增加 backbone forward；
- 均匀类别分布时 `G=1/C`，严格退化到原 ReSA；
- prototype/category target 要提前到 assignment 之前计算；
- 仍然只能读取旧 queue，最后再 enqueue。

先用 checkpoint 对 `lambda_r` 做只读 assignment 扫描，例如：

```text
0.05 / 0.1 / 0.25 / 0.5
```

正式训练只跑选定值，避免无边界网格搜索。

### 第七步：2×2 因果消融

| 实验 | Prototype | ReSA assignment | 目的 |
|---|---|---|---|
| T00 | P0 | 原始 ReSA | 统一基线 |
| T10 | P* | 原始 ReSA | 可靠原型独立贡献 |
| T01 | P0 | Semantic ReSA | 类别关系独立贡献 |
| T11 | P* | Semantic ReSA | 完整核心方法 |

必须计算：

```text
Delta_P = T10 - T00
Delta_R_base = T01 - T00
Delta_R_reliable = T11 - T10
Interaction = (T11-T10) - (T01-T00)
```

如果 T10/T01 各自提升但 T11 不再提升，说明两个模块可能重叠，不能宣称互补。

### 第八步：多 exemplar seed

架构搜索阶段固定 seed0。最终至少重跑 T00 和 T11 的 exemplar seed 0/1/2，报告：

- mean±std；
- 最差 seed；
- 每个 seed 的 exemplar 索引。

one-shot 方法不能只报告 seed0。

## 9. 测试必须覆盖的语义

1. 关闭 OSE 时旧 ReSA 路径保持一致。
2. online/EMA 初始化一致，EMA 参数无梯度。
3. P0 数值复现旧 Q4 聚合。
4. P1 同一个 queue sample 不会分给多个类别。
5. P1 每类最多 4 个邻居，候选不足可安全退化。
6. P2 与 P1 选择相同，只改变 aggregation score。
7. P3 输出单位范数 prototype。
8. M-F 两项可以独立开关。
9. mixed branch 不进入 predictor、Sinkhorn 或 queue。
10. 当前 batch 的所有 loss/target 完成后才 enqueue。
11. backward 后 online 有梯度，EMA/target/queue 无梯度。
12. state_dict 包含 EMA、projector、queue、pointer 和 filled state。
13. Semantic ReSA 均匀 `P` 时与原 similarity/Sinkhorn 一致。
14. DDP 时必须额外验证 all-gather、queue 和 pointer 跨卡一致；当前正式实验是单 GPU。

## 10. 绝对不要再踩的坑

1. **不要称完全无监督。** 每类一个带标签 exemplar 是 one-shot-assisted 设置。
2. **不要混比旧 weak+weak 与新增强协议。** 历史 Q4=79.98 不是新 P0 的结果；新 P0=78.79。
3. **不要说 Top-K 越小越好。** Q0=77.22 已反证；Q4>Q8 也不等于 Q4 purity 必然更高。
4. **不要重新引入置信度设计。** 用户已明确否决 entropy/JS/阈值/confidence queue/动态 K。
5. **不要把 corrected instance queue 放回主线。** 它得到 77.44，且给 ReSA 人为增加额外实例队列，不够自然。
6. **不要让 ground-truth label 进入训练关系。** 标签只用于 exemplar 选择和离线诊断。
7. **不要混用 H/Z/Q 或 online/EMA。** 尤其不要用 predictor 输出建 OSE prototype。
8. **不要提前 enqueue 当前 batch。** 必须先完成 logits、target、assignment 和 loss。
9. **不要把 purity n/a 当成 NaN 或 purity=0。** 它表示没有有效邻居；长期出现才说明类别饥饿。
10. **不要一次改多个阶段。** 结果必须按 P1-P0、P2-P1、P3-P2 归因。
11. **不要把 early acc、best acc 和 last acc 混为一个结论。** 当前 LP 的 last 不一定是最优点。
12. **不要用短跑作为论文结果。** 不同 total-epoch cosine schedule 的同名 checkpoint 也不能直接比较。
13. **不要把 `weights + start_epoch` 当完整 resume。** optimizer、scheduler、EMA、queue、pointer 和 RNG 没恢复就不是同一实验。
14. **不要拿 CTR-GCN epoch190 与 ST-GCN epoch300 下正式 backbone 结论。** 76.15 只是中断 checkpoint 的记录。
15. **不要改动或覆盖原始 AimCLR 文件和历史 work_dir。** 新模块、配置和目录保持隔离。
16. **不要仅凭 seed0 宣称稳定。** P1 +0.96 是积极信号，但最终必须多 seed。
17. **不要擅自停止、继续、删除或覆盖服务器任务。** 新会话默认服务器状态未知，先向用户确认 GPU、任务状态、配置、work_dir 和预计时长。
18. **不要误删 `BASELINE_MIGRATION_GUIDE.md`。** 它是本会话新增、尚未跟踪的重要迁移文档。

## 11. 当前最简交接结论

当前主线已经从“尝试各种 OSE 组合”收敛为：

```text
第一阶段：P0 -> P1 -> P2 -> P3，得到可靠 Q4 prototype P*
第二阶段：用 OSE soft relation 修正 ReSA BxB assignment
第三阶段：用 2×2 消融验证两部分是否独立、互补
第四阶段：多 exemplar seed 验证稳定性
```

已知最新正式结果是：

```text
P0 = 78.79
P1 = 79.75
P1-P0 = +0.96
```

下一项正式任务不是继续讨论 P1，而是：**先核对测试和 P1 日志，然后从头运行 P2 pretext300 + LP200；完成后再运行 P3。**

Semantic ReSA 仍是设计方案，不是已经实现或验证的贡献。
