# AimCLR Stage2 RSDG 研究交接（当前唯一有效版本）

最后更新：2026-08-10

本文写给一个完全没有此前聊天上下文的新会话。请先完整阅读本文，再查看代码、管理服务器任务或提出新方案。旧版 P0–P3、Semantic ReSA、corrected queue、warm-up 等路线均已退出当前主线。

## 0. 新会话首先要知道的十件事

1. 当前研究范围是 **AimCLR + NTU60 XSub + Joint stream + ST-GCN** 下的二阶段 one-shot 类别语义落地。先锁定 AimCLR，不要立即扩数据集、协议或 backbone。
2. Stage1 是默认 AimCLR 300 epoch，固定 LP Top-1 为 `75.27`。
3. Stage2 固定 100 epoch，只载入 Stage1 online backbone；native ReSA/OSE heads 随机初始化，EMA 分支从 online 精确复制。
4. 当前完整方法是 **Dual-space ReSA+OSE、JMB label-only prototype、Q0 neighbor-free、mixed losses**。
5. Gate 1 已完成 exemplar seed0/1/2。Dual 三 seed 均值最高，但只比 Shared 高 `0.08`，因此是“窄幅通过”，不能宣称稳定或显著优于 Shared。
6. 三个 Dual diagnostics 都正常：没有 collapse、没有持续负梯度冲突；OSE encoder 梯度约为 ReSA 的 23 倍，两个 projector 会稳定分化。
7. 当前最新任务是测试 **同一个 exemplar 的多增强 JMB 集成**。代码已完成并提交，但正式 K=2 训练和 LP 结果尚未返回。
8. 首轮多增强实验固定最差的 exemplar seed1、K=2，只改变增强集成，不同时修改 projector、loss 权重、online/EMA 分支或其他协议。
9. 当前仓库 `main`、`origin/main`、`origin/HEAD` 均为 `9cf5dfcfa39a4f3a40a3bf9081e4eccb8ff2c33e`，提交信息 `multi aug test`。写本文前工作区干净。
10. 不知道服务器是否已经运行 K=2 脚本。新会话不得擅自启动、停止、重启或覆盖任务；必须先询问用户并检查 GPU、进程和输出目录。

## 1. 我们在研究什么

### 1.1 研究问题

AimCLR 的实例对比学习倾向于保留实例可分性；OSE 使用每类一个带标签 exemplar 构造类别 prototype，倾向于把同类样本压向类别中心。早期实验表明，把 AimCLR 与 OSE 直接放在同一训练阶段会退化，因此当前路线是：

```text
Stage1：AimCLR 学习实例级表示
Stage2：ReSA 约束关系结构，OSE 注入 one-shot 类别语义
```

当前方法暂称 **Relational–Semantic Decoupled Grounding（RSDG）**，核心是：

1. **时间解耦**：先实例级自监督预训练，再做 one-shot 类别落地；
2. **几何解耦**：ReSA 与 OSE 使用独立 projector，但共同更新一个 backbone；
3. **结构化 exemplar**：同一带标签骨架样本确定性构造 Joint/Motion/Bone；
4. **neighbor-free grounding**：主方法 Q0 不读取无标签邻居。

正确任务名称：

- one-shot-assisted self-supervised learning；
- label-efficient self-supervised learning；
- post-contrastive one-shot class grounding。

不能称 fully unsupervised，因为每类使用一个带标签 exemplar。

### 1.2 当前论文贡献边界

OSE 和 ReSA 的基础损失均来自已有工作，不能把它们本身写成原创。当前可能形成贡献的是：

- 同阶段实例目标与稀疏类别目标的冲突诊断；
- post-contrastive temporal decoupling；
- relational/semantic projection-space factorization；
- skeleton-specific structure-complete exemplar；
- neighbor-free label-only prototype；
- exemplar 多增强稳定化（当前待验证）。

当前不能把 ReSA 写成已经证明的“Stage1 relational preservation”。ReSA target 来自正在更新的 Stage2 H/EMA H，并非冻结的 Stage1 anchor。更安全的表述是：

> relation-structured auxiliary constraint during semantic grounding

若未来需要更强的方法创新，可单独测试 frozen Stage1 relational anchor，但它不是当前 K=2 任务的一部分。

## 2. 精确模型语义

### 2.1 Stage1 到 Stage2 的迁移

Stage1：

```text
config/ntu60/pretext/pretext_aimclr_xsub_joint.yaml
./data/ntu60_cs/aimclr_joint/pretext/epoch300_model.pt
```

Stage2 只载入 AimCLR `encoder_q` backbone，不载入：

- `encoder_q.fc` / Stage1 projector；
- `encoder_k`；
- AimCLR queue 与 pointer；
- NNM/DDM 状态；
- optimizer/scheduler/RNG。

Stage2 初始化：

- online backbone = Stage1 online backbone；
- EMA backbone = online backbone 精确复制；
- ReSA projector/predictor 随机初始化；
- OSE projector 与 ReSA projector 同构、同初始化，之后独立更新；
- EMA projectors 从各自 online projector 精确复制。

### 2.2 H / Zr / Qr / Zs 不得混用

```text
H   = backbone feature
      用于 ReSA B×B 关系和 Sinkhorn assignment

Zr  = ReSA projector feature
Qr  = ReSA predictor output
      用于 ReSA 跨视图关系预测

Zs  = OSE projector feature
      用于 exemplar、JMB prototype、类别 target、mixed losses 和 OSE queue
```

ReSA 不是向量级 `KL(H || Z)`：

```text
A_H = Sinkhorn(sim(H_online, H_teacher))
P_r = softmax(sim(Qr_online, Zr_teacher) / tau)
L_ReSA = CE(A_H, P_r)
```

### 2.3 固定 Stage2 协议

```text
epochs = 100
batch_size = 128
backbone LR = 0.25 -> 0, per-iteration cosine
head LR = 0.25 -> 0, per-iteration cosine
warm-up = 0
EMA base momentum = 0.996 -> 1 cosine
ReSA weight = 1
Lproto/Lmix-proto/Lmix-ins weights = 1/1/1
checkpoint = fixed epoch100
LP = 200 epochs, fixed protocol, report final Best Top1
```

不要通过多个 Stage2 checkpoint 的 LP 结果挑最好 checkpoint。

## 3. JMB prototype 的精确语义

### 3.1 必须先增强，再派生

正确顺序：

```text
raw exemplar
-> temporal crop / shear / rotation
-> augmented primary skeleton
-> deterministically derive Joint / Motion / Bone
```

同一个增强组中的 J/M/B 必须来自同一增强后的 raw exemplar。不能先派生 J/M/B 再分别独立随机增强，否则 cosine agreement 会混入结构差异和增强差异。

Motion：

```text
motion[t] = joint[t+1] - joint[t]
last frame = 0
```

Bone 使用 NTU 25-joint 固定骨骼边。

### 3.2 Online/EMA 的通用规则

通用规则不是“Joint 永远 online”，而是：

```text
当前训练主 stream -> online backbone/projector
其余结构模态    -> EMA backbone/projector
```

因此：

| 训练 stream | Online anchor | EMA auxiliaries |
|---|---|---|
| Joint | Joint | Motion、Bone |
| Bone | Bone | Joint、Motion |
| Motion | Motion | Joint、Bone |

当前正式实验全部是 Joint stream，所以当前是 Joint-online + Motion/Bone-EMA。

更准确的论文表述是：

> primary-stream-anchored structural consensus

当前 Joint 配置的组内融合：

```text
zJ, zM, zB = normalized embeddings
score_i = cosine(z_i, zJ)
w = softmax(score)
JMB = normalize(sum_i w_i z_i)
```

Joint-online 提供可学习主锚点；Motion/Bone-EMA 提供慢变化结构证据，避免三路高度相关的 OSE 梯度同时支配 backbone。

### 3.3 Q0 的准确措辞

`ose_topk=0` 时 prototype 不读取无标签邻居，因此可称：

> label-only, neighbor-free JMB prototype

当前代码仍分配并更新 OSE queue，因此不能称 queue-free。

## 4. 已有正式结果

所有结果为 NTU60 XSub Joint linear evaluation Top-1。

### 4.1 历史与结构结果

| 实验 | Top-1 | 相对 Stage1 |
|---|---:|---:|
| AimCLR Stage1 | 75.27 | — |
| AimCLR+OSE 同阶段 | 73.79 | -1.48 |
| 旧平滑迁移 epoch50/100 | 76.26 | +0.99 |
| 旧平滑迁移 epoch300 | 75.96 | +0.69 |
| native Stage2 Joint Q4 | 78.29 | +3.02 |
| Shared JMB Q0 | 78.75 | +3.48 |
| Shared JMB Q4 | 78.78 | +3.51 |

安全结论：

- 时间解耦明显优于同阶段耦合；
- Q4 条件下 Joint -> JMB 增加 `0.49`；
- Shared JMB 下 Q4-Q0 只有 `+0.03`，不能证明邻居有效；
- 主方法继续使用更简单的 Q0。

历史上从头训练 ReSA+OSE 曾有 `79.75`，但协议和起点不同，只能作历史参考，不能做严格单因素比较。

### 4.2 A5 seed0 归因

| 方法 | Top-1 | 相对 Stage1 |
|---|---:|---:|
| ReSA-only | 74.62 | -0.65 |
| OSE-only JMB Q0 | 78.48 | +3.21 |
| Shared ReSA+OSE | 78.75 | +3.48 |
| Dual-space ReSA+OSE | 79.15 | +3.88 |

由此可知：

- OSE 是主要准确率来源；
- ReSA-only 不创造类别判别性；
- ReSA 与 OSE 联合时可能提供条件性关系约束；
- seed0 上 Dual-Shared = `+0.40`，需要多 exemplar seed 验证。

### 4.3 Gate 1 exemplar seed0/1/2

训练随机 seed 固定为0，只改变 exemplar seed。因此以下衡量的是 exemplar selection 敏感性，不是完整训练随机性。

| exemplar seed | OSE-only | Shared | Dual |
|---:|---:|---:|---:|
| 0 | 78.48 | 78.75 | 79.15 |
| 1 | 77.93 | 78.77 | 78.26 |
| 2 | 78.08 | 78.66 | 79.00 |
| mean | 78.163 | 78.727 | 78.803 |
| sample std | 0.284 | 0.059 | 0.477 |

配对差值：

| seed | Shared-OSE | Dual-OSE | Dual-Shared |
|---:|---:|---:|---:|
| 0 | +0.27 | +0.67 | +0.40 |
| 1 | +0.84 | +0.33 | -0.51 |
| 2 | +0.58 | +0.92 | +0.34 |
| mean | +0.563 | +0.640 | +0.077 |

Gate 1 决策：

- Dual 三 seed 均值最高；
- Dual 和 Shared 对 OSE-only 都是3/3正收益；
- Dual 只比 Shared 平均高 `0.077`，且 seed1 反转；
- 因此 Dual **窄幅通过**并进入后续验证；
- Shared 必须保留为关键对照；
- 不能写“Dual consistently/significantly outperforms Shared”。

## 5. 三个 Dual diagnostics 的结论

用户回传了 Dual seed0/1/2 的完整 `stage2_diagnostics.csv`。CSV 本身没有 seed 字段，分析按附件顺序映射为 seed0/1/2，对应 LP `79.15/78.26/79.00`。

### 5.1 三个 run 都正常

- 每个100 epoch、每 epoch 312 batches；
- loss 平滑下降；
- 没有异常 NaN；Dual 的 shared-projector cosine 为 NaN 是预期行为；
- encoder/projector/predictor drift 三个 seed 非常接近；
- 无参数爆炸或训练崩坏。

### 5.2 ReSA 关系任务都收敛

大致趋势：

```text
cluster_kl: 1.80-1.95 -> 1.22-1.33
relation target/pred cosine: 0.56 -> 0.66
relation top1 agreement: 0.982 -> 0.9995
```

但 `encoder_resa_relation_cos` 在三个 seed 都从约 `0.10-0.11` 降到 `0.042-0.047`，没有上升。因此不能用 raw H-Z cosine 证明 Stage1 relation 被直接保留。

seed1 的 `cluster_kl` 最低，但其 `cluster_entropy` 最高（后期约 `2.078`，seed0/2约 `1.888/1.910`）。低 KL 部分来自更平坦、更容易匹配的 assignment，不代表下游表示更好。

### 5.3 Dual heads 确实分化

`mean_resa_ose_relation_cos`：

| seed | epoch1 | epoch100 |
|---:|---:|---:|
| 0 | 0.841 | 0.382 |
| 1 | 0.833 | 0.402 |
| 2 | 0.842 | 0.367 |

两个同初始化 projector 稳定分化，说明 Dual 不是形式上的重复 head。seed1 分化略弱，但只有三个点，不能宣称分化程度决定 LP。

### 5.4 不存在持续负梯度冲突

encoder ReSA/OSE 首 batch 梯度 cosine：

| seed | 全程均值 | 负值比例 |
|---:|---:|---:|
| 0 | +0.136 | 5% |
| 1 | +0.150 | 2% |
| 2 | +0.139 | 6% |

负值少且幅度小。正确机制是：

> semantic gradient dominance + projection-space specialization

不能写成：

> persistent negative gradient conflict resolved by Dual

OSE/ReSA 平均梯度比：

```text
encoder:   22.5-24.6x
projector:  7.9-8.2x
```

### 5.5 没有 collapse，但有 semantic contraction

projector：

- feature std约 `0.061-0.063`；
- off-diagonal cosine接近0；
- 没有 collapse。

encoder：

```text
feature std: about 0.037 -> 0.033
offdiag cosine: about 0.59-0.62 -> 0.66-0.68
```

这是 semantic contraction / increased anisotropy，不是 collapse。

seed1 的后期 encoder std最低、offdiag cosine最高，且 relation assignment entropy最高。其低 LP 更像 exemplar 诱导的关系扩散和语义收缩，不是 ReSA 未收敛、参数异常或负梯度冲突。

## 6. 当前最新任务：多增强 JMB 集成

### 6.1 研究动机

Dual 对 exemplar seed 更敏感：

```text
seed range = 78.26-79.15
sample std = 0.477
```

首轮目标是测试同一个标注 exemplar 的多个随机增强能否降低 prototype 方差并恢复最差 seed1。

### 6.2 新 `ose_exemplar_views` 语义

旧实现中 `ose_exemplar_views>1` 只增加独立 Joint EMA view，不是完整多增强 JMB。

提交 `9cf5dfc` 后，其语义改为：

> exemplar_views = 独立的完整结构增强组数量

K=2：

```text
Augment_1(raw) -> Joint_1/Motion_1/Bone_1 -> JMB_1
Augment_2(raw) -> Joint_2/Motion_2/Bone_2 -> JMB_2

prototype = normalize(mean(JMB_1, JMB_2))
```

每个组：

- primary stream（当前 Joint）走 online；
-其余 Motion/Bone 走 EMA；
- 组内使用原 cosine-softmax JMB；
- 组间对归一化 JMB 做均值再归一化。

K=1 仍走原始路径，不进行额外 ensemble，正式旧配置不变。

### 6.3 BN 公平性保护

K=2 会多一次 online primary-stream exemplar forward。为了不把 prototype 集成与 BN running-statistics 改变混在一起：

- 额外 online exemplar 仍参与梯度；
- 其 online backbone/projector BN running mean、variance、num_batches_tracked 在前向后恢复；
- EMA exemplar forward 原本就保存/恢复 BN buffers。

因此 K=1/K=2 的长期 BN 统计差异不会来自额外 exemplar forward。

### 6.4 计算成本

当前 K=1 JMB+M-F 每 iteration backbone forwards：

```text
online view A/B       2
online Joint exemplar 1
online mixed view     1
EMA view A/B          2
EMA Motion/Bone       2
total                 8
```

K=2 额外增加一个 online primary-stream exemplar和两个 EMA structural exemplars，总计11次，backbone forward数增加37.5%。只有准确率或稳定性改善明确才值得保留。

### 6.5 已修改文件

```text
net/ose_resa.py
processor/pretrain_ose_resa.py
tests/test_ose_resa_lmix.py
tests/test_ose_resa_stage2.py
run_stage2_dual_jmb_multiaug.sh
```

新增测试覆盖：

- 两个增强组各自 J/M/B 同源；
- 每个增强组完整 JMB；
- 组间 normalized mean；
- online梯度存在；
- EMA无梯度；
- 额外 online/EMA exemplar forward 不污染 BN buffers。

本地只完成 AST 语法检查和 `git diff --check`。本机没有 PyTorch，也没有 Bash，因此运行时单测尚未在本机执行。新脚本会在服务器正式训练前自动执行相关单元测试。

## 7. 如何运行当前实验

独立脚本：

```bash
bash run_stage2_dual_jmb_multiaug.sh all
```

默认：

```text
SEED=1
AUGMENTATIONS=2
```

脚本顺序：

1. 检查 Stage1 checkpoint；
2. 运行：
   - `tests.test_ose_resa_lmix`
   - `tests.test_ose_resa_stage2`
   - `tests.test_ose_resa_prototypes`
3. 运行100 epoch Dual Stage2；
4. 固定 epoch100 做200 epoch LP；
5. 写出 `lp_best_acc.csv`。

输出目录：

```text
./data/ntu60_cs/aimclr_to_native_ose_resa_a4_dualproj_jmb_q0_mf_ma2_100ep_joint_seed1/
```

脚本会：

- 已有完整 epoch100 时跳过 pretrain；
- 已有完整 epoch200 LP 时复用；
- 遇到非空但不完整的 work_dir 时拒绝覆盖。

开始前仍必须人工检查 GPU、进程和目录。脚本不是任务管理器，也不支持严格 resume。

可选运行：

```bash
SEED=0 AUGMENTATIONS=2 bash run_stage2_dual_jmb_multiaug.sh all
SEED=2 AUGMENTATIONS=2 bash run_stage2_dual_jmb_multiaug.sh all
AUGMENTATIONS=4 bash run_stage2_dual_jmb_multiaug.sh all
```

不要在 seed1 K=2 结果出来前直接全面跑 K=4。

## 8. 预先固定的 K=2 判定规则

seed1 当前 Dual baseline = `78.26`，seed1 Shared = `78.77`。

为了避免看结果后改标准：

- `K2 >= 78.66`：强信号，至少恢复接近/超过 Shared，补 seed0/2；
- `78.36 <= K2 < 78.66`：弱改善，只补一个 seed再判断；
- `K2 <= 78.35`：基本无效，停止多增强路线，不跑 K=4。

如果 K=2 三 seed 有效，报告：

- 每 seed结果；
- mean ± std；
- K2-K1 配对差；
- diagnostics 中 encoder std/offdiag、cluster entropy、head relation cosine；
- 计算成本增加。

## 9. 当前卡在哪里

代码没有已知实现阻塞。当前唯一实质阻塞是：

1. 不知道服务器是否已经启动 `run_stage2_dual_jmb_multiaug.sh`；
2. 尚无 seed1 K=2 epoch100 checkpoint；
3. 尚无 seed1 K=2 200-epoch LP；
4. 尚无 K=2 `stage2_diagnostics.csv`；
5. 本地无 PyTorch/Bash，运行时验证必须在服务器完成。

新会话第一步应询问用户：

- K=2脚本是否已启动；
- 当前跑到单测、pretrain还是LP；
- GPU和进程状态；
- work_dir是否已存在；
- 是否有报错日志。

不要盲目再次运行脚本。

## 10. 下一步计划（严格按顺序）

### 第一步：完成 seed1 K=2

收集：

- 服务器单元测试输出；
- epoch100 checkpoint；
- LP Best Top-1；
- `stage2_diagnostics.csv`；
- 若失败，收集日志尾部和完整异常堆栈。

### 第二步：按预注册规则决定是否补 seed

- 强信号：补 seed0/2 K=2；
- 弱信号：只补一个 seed；
- 无效：停止多增强，不跑 K=4。

多增强实验期间不要同时改变 online/EMA 分支。否则无法判断收益来自增强集成还是分支变化。

### 第三步：完成 Gate 2 结构消融

在最终保留的 Dual K配置下依次做：

```text
Joint Q0
Joint+Motion Q0
Joint+Bone Q0
Joint+Motion+Bone Q0
```

若继续使用 K=1，则已有 JMB Q0 seed0 = `79.15`。

随后做最终架构中的：

```text
Dual JMB Q4 vs Dual JMB Q0
```

Shared 下 Q4-Q0=0.03 不能完全替代 Dual 的邻居消融。

### 第四步：补机制证据

优先级：

1. 获取 Shared seed0/1/2 diagnostics，与 Dual 对比；
2. 必要时做 ReSA backbone-gradient routing；
3. 必要时做 shuffled relation target 负对照；
4. 不要先上 PCGrad/GradNorm，因为当前梯度大多同向。

### 第五步：最后才迁移其他 baseline

AimCLR 的结构、稳定性和归因完成后，再迁移同数据、同协议、同 stream、同 ST-GCN 的第二个自监督 baseline。

首个新 baseline 至少运行：

```text
Stage1
OSE-only
Shared
Dual
```

seed0 有正收益后再补 exemplar seed1/2。

## 11. 绝对不要再踩的坑

1. **不要再提出或实现 warm-up。** 用户已明确不测试 ReSA/OSE warm-up、冻结 encoder warm-up或动态 OSE激活。
2. **不要现在全面迁移其他 baseline/数据集。** 先完成多增强判断和 AimCLR Gate 2。
3. **不要改变 K=2 的其他变量。** 不同时改 projector、loss weight、LR、online/EMA 路径或 queue。
4. **不要先派生 J/M/B 再独立随机增强。** 必须增强 raw exemplar 后再派生同源结构模态。
5. **不要说 Joint 永远 online。** 通用规则是 primary training stream online，其余结构模态 EMA。
6. **不要把 JMB 描述成三模态完全对称。** 当前是 primary-stream-anchored structural consensus。
7. **不要把 Q0 称 queue-free。** 当前仍维护未读取的 OSE queue，只能称 neighbor-free。
8. **不要把 Q4 当主方法。** Shared JMB 下只比 Q0 高0.03。
9. **不要把 ReSA写成向量级 `KL(H || Z)`。** 它对齐的是关系 assignment和Q-Z关系分布。
10. **不要宣称 ReSA 已严格保存 Stage1关系。** 当前 target 随 Stage2 encoder/EMA漂移。
11. **不要宣称 Dual 解决持续负梯度冲突。** 三 seed梯度cosine绝大多数为正。
12. **不要宣称 Dual稳定显著优于 Shared。** 三 seed均值仅高0.077且 seed1反转。
13. **不要把 exemplar seed 当训练随机 seed。** 当前训练随机 seed固定0。
14. **不要只报告最好 seed。** 必须报告每 seed和mean±std。
15. **不要通过反复 LP挑Stage2 checkpoint。** 正式checkpoint固定epoch100。
16. **不要把 smoke、未完成checkpoint或不同cosine总周期混入正式结果。**
17. **不要把 weights+start_epoch当完整resume。** optimizer、scheduler、EMA、queue、pointer和RNG未恢复就不是同一轨迹。
18. **不要迁移AimCLR旧projector、teacher、queue、NNM/DDM。** Stage2只载入online backbone。
19. **不要预填随机projector queue。** `stage2_prefill_queue=False`。
20. **不要在 loss计算完成前 enqueue当前batch。**
21. **不要让额外 exemplar forward污染 EMA BN buffers。** 新代码已有保存/恢复测试。
22. **不要删除额外 online exemplar的BN保护。** 否则K1/K2比较会混入BN统计变化。
23. **不要在 seed1 K=2结果前直接跑K4或所有seed。**
24. **不要盲目重跑脚本。** 先检查GPU、进程和work_dir，避免覆盖结果。
25. **不要删除或回退提交 `9cf5dfc` 的多增强实现。** 除非用户明确要求回退。
26. **不要把旧 handoff 中“A5结果尚未回传”当当前状态。** A5和Gate 1均已完成。

## 12. Git 与文件状态

写本文前：

```text
branch = main
HEAD = 9cf5dfcfa39a4f3a40a3bf9081e4eccb8ff2c33e
origin/main = 9cf5dfcfa39a4f3a40a3bf9081e4eccb8ff2c33e
commit = multi aug test
working tree = clean
```

该提交包含：

- 完整多增强 JMB分组；
- 组间 normalized mean；
- 额外 online exemplar BN buffer保护；
- processor/model单元测试；
- seed1/K=2安全运行脚本。

写入本文后，预期 working tree 只出现 `handoff.md` 修改，除非用户或其他进程随后产生新变化。新会话开始仍需执行：

```bash
git status --short
git diff --check
```

不要覆盖用户后续改动。

## 13. 用户返回 K=2 结果后如何回复

不能只回答“提高了/降低了”。至少输出：

1. K1/K2 seed1 LP及差值；
2. 与 seed1 Shared=78.77 的比较；
3. 是否达到预先固定阈值；
4. K2 `cluster_entropy`、`cluster_kl`、relation target/pred；
5. encoder feature std和offdiag cosine是否缓解；
6. ReSA/OSE head relation cosine；
7. gradient cosine和OSE/ReSA gradient ratio；
8. 是否有collapse、异常drift或BN问题；
9. 是否补seed0/2，或停止多增强；
10. 计算成本是否值得。

当前最简状态：

> Gate 1 已完成，Dual窄幅通过；三个Dual run机制正常但对 exemplar敏感。完整K=2多增强JMB实现已提交，正在等待最差seed1的服务器单测、100-epoch Stage2、200-epoch LP和diagnostics结果。
