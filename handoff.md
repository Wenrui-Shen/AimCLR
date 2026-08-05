# AimCLR Stage2 ReSA/OSE 研究交接（当前唯一有效版本）

最后更新：2026-08-04。

> 2026-08-05 补充：A5 已完成，Shared/ReSA-only/OSE-only/Dual 的 LP 分别为
> `78.75/74.62/78.48/79.15`。`run_stage2_a5_ablation.sh` 保留历史文件名，
> 但当前已改为运行 exemplar seed1/2 × OSE-only/Shared/Dual 的 Gate 1
> 配对验证，并汇总每个 LP 的 Best Top-1。当前迁移与表述规范以
> `BASELINE_MIGRATION_GUIDE.md` 为准。

本文写给一个完全没有此前聊天上下文的新会话。请先完整阅读本文件，再查看代码或提出新方案。本文已经替换旧交接；旧的 P0–P3、Semantic ReSA、corrected queue 等计划均为历史路线，不是当前下一步。

## 0. 新会话首先要知道的五件事

1. 当前只在 **AimCLR + NTU60 XSub Joint + ST-GCN** 上寻找并验证最佳二阶段方案，暂时不要迁移其他 baseline、数据集或协议。
2. 用户明确不准备测试 warm-up；不要再次建议或实现 ReSA-only warm-up、冻结 encoder warm-up 或动态 OSE 激活。
3. 当前主候选是 **AimCLR Stage1 → native ReSA+OSE Stage2，JMB、Q0无邻居、共享projector**，已有 LP `78.75`。
4. 最新任务是 A5：重新运行 Shared、ReSA-only、OSE-only、Dual-projector 四组实验，获得各自 LP 和 `stage2_diagnostics.csv`，判断收益归因及冲突发生在共享 encoder 还是共享 projector。
5. 当前代码和配置已经完成，主要阻塞是 **服务器正式训练结果尚未回传**。服务器是否已经启动脚本、运行到哪里、GPU状态如何均未知；新会话不能擅自重启、覆盖或删除任务。

## 1. 我们在研究什么

### 1.1 研究问题

AimCLR 使用带 memory bank 的实例对比学习，倾向于保持实例可分性；OSE 使用每类一个带标签 exemplar 构造类别 prototype，倾向于将同类样本压缩到类别中心。我们已经观察到，把两类目标直接放进同一阶段会明显下降，因此转向：

```text
Stage1：AimCLR 学习实例级表示
Stage2：ReSA 整理关系结构，OSE 注入 one-shot 类别语义
```

研究目标不是简单拼接 ReSA 和 OSE，而是回答：

1. 实例判别与 one-shot 类别 grounding 的冲突能否通过时间解耦缓解；
2. Stage2 随机初始化的 native ReSA projector 如何适配已经训练好的 AimCLR encoder；
3. ReSA 和 OSE 的贡献分别是多少，是否互补；
4. 两个目标的梯度冲突发生在共享 projector、共享 encoder，还是两者都有；
5. 骨架 exemplar 的 Joint/Motion/Bone 确定性结构视图能否替代 OSE 的无标签邻居检索。

### 1.2 正确的任务名称

每个类别使用一个带标签 exemplar，其余训练样本不使用标签。因此不能称为“完全无监督”，应称为：

- one-shot-assisted self-supervised learning；
- label-efficient self-supervised learning；
- post-contrastive one-shot class grounding。

### 1.3 论文创新的当前出发点

所有基本损失来自 ReSA/OSE，不能把损失本身写成原创。可能形成贡献的是：

1. **Conflict diagnosis**：同阶段 AimCLR+OSE 从 `75.27` 降到 `73.79`；
2. **Post-contrastive temporal decoupling**：先实例学习，再类别语义整理；
3. **Relational handoff**：ReSA 将 Stage1 encoder 中的 batch 关系传递到新初始化的 Stage2 projector/predictor 空间；
4. **Structure-complete exemplar**：同一个带标签骨架样本确定性构造 Joint/Motion/Bone，替代不稳定的无标签邻居扩张；
5. **Neighbor-free grounding**：JMB 下 Q4 相比 Q0 只提高 `0.03`，邻居在当前协议中基本无增益。

注意：第3点目前是待 A5 日志验证的机制解释，不能提前写成已经证明的事实。

## 2. 已有结果与可以安全得出的结论

当前主要结果统一记为 NTU60 XSub Joint 的 linear evaluation Top-1：

| 实验 | LP Acc | 说明 |
|---|---:|---|
| 默认 AimCLR Stage1 | `75.27` | 300 epoch，当前二阶段统一起点 |
| A2：AimCLR+OSE同阶段 | `73.79` | 相对 Stage1 `-1.48`，支持目标冲突现象 |
| A3首版平滑迁移，epoch50/100 | `76.26` | 迁移AimCLR head的旧方案 |
| A3首版平滑迁移，epoch300 | `75.96` | 长训反而下降 |
| native Stage2：Joint + Q4 | `78.29` | 只迁移AimCLR backbone，native ReSA head随机初始化 |
| native Stage2：JMB + Q0 | `78.75` | 当前无邻居主候选，Stage1 `+3.48` |
| native Stage2：JMB + Q4 | `78.78` | 相比Q0仅 `+0.03` |

由此目前可以安全说：

- 同阶段加入 OSE 失败，二阶段明显更好；
- 在 Q4 条件下，JMB 相比仅 Joint 从 `78.29` 提高到 `78.78`，增量 `+0.49`；
- 在 JMB 条件下，Q4 相比 Q0 只有 `+0.03`，未做多 seed 前应视为持平；
- 主方法应优先使用 JMB Q0，Q4只作为邻居消融；
- 当前单次结果不能证明统计显著，也不能证明 ReSA、OSE、JMB 各自的独立贡献，因此必须完成 A5。

历史上从头训练的 ReSA+OSE P1 曾得到 `79.75`，但训练起点、协议和研究问题不同。它只能作为历史上限参考，不能与当前 AimCLR→Stage2 的 `78.75/78.78` 做严格单因素比较。

## 3. 当前二阶段框架的精确语义

### 3.1 Stage1

```text
配置：config/ntu60/pretext/pretext_aimclr_xsub_joint.yaml
训练：完全默认 AimCLR，300 epochs
权重：./data/ntu60_cs/aimclr_joint/pretext/epoch300_model.pt
```

Stage2只读取 AimCLR `encoder_q` 的 backbone 权重，不读取：

- `encoder_q.fc`；
- `encoder_k`；
- AimCLR queue；
- NNM/DDM状态；
- optimizer/scheduler。

### 3.2 Stage2 native ReSA head

Stage2重新初始化：

```text
ReSA projector：256 -> 2048 -> 2048 -> 256
ReSA predictor：native predictor
```

online backbone和随机online head分别复制到EMA分支，因此Stage2起点的online/teacher严格对齐。

正式A5配置固定：

```text
Stage2 epochs = 100
backbone lr = 0.25 -> 0 cosine
head lr = 0.25 -> 0 cosine
resa_warmup_epoch = 0
stage2_load_projector = False
stage2_prefill_queue = False
save_interval = 10
GPU device = [1]
```

虽然optimizer内部仍支持backbone/head分组学习率，但正式配置二者均为 `0.25`。用户已经要求固定同一LR，不要重新引入分离LR作为主线。

### 3.3 H / Z / Q 的语义绝对不能混用

```text
H = encoder/backbone feature
    ReSA用H构造B×B相似关系和Sinkhorn assignment

Z = projector feature
    OSE exemplar、prototype、类别target、mix和邻居queue均在Z中

Q = predictor output
    只用于ReSA online跨视图关系预测
```

ReSA的 `cluster_kl` 不是 encoder feature 与 projector feature 向量之间直接做KL。它是：

```text
A_H = Sinkhorn(sim(H_online, H_teacher))
P_Z = softmax(sim(Q_online, Z_teacher) / tau)
L_ReSA = CE(A_H, P_Z) = H(A_H) + KL(A_H || P_Z)
```

因此论文里正确的说法是“encoder-derived relation 到 projector/predictor relation 的关系蒸馏/交接”，不能写成 `KL(H || Z)`。

### 3.4 JMB label-only prototype

每类只使用同一个增强后的带标签 exemplar，并确定性构造：

- Joint：online encoder/projector；
- Motion：EMA encoder/projector；
- Bone：EMA encoder/projector。

三个归一化embedding以Joint为参考，根据cosine agreement做无额外超参的softmax聚合，得到label-only prototype。

当前主候选 `ose_topk: 0`，因此prototype不使用无标签邻居。Q4版本只作消融。

重要措辞：当前Q0代码仍然分配并更新OSE feature queue，只是 `topk=0` 时完全不读取邻居。因此现在只能称 **neighbor-free**，不能严谨地称 **queue-free**。若论文最终要宣称queue-free，之后必须在Q0下关闭queue分配与enqueue，并验证数值不变。

### 3.5 当前完整损失

Shared和Dual完整模型：

```text
L = L_ReSA
  + L_proto
  + L_mix-proto
  + L_mix-ins
```

四项权重均为1。mixed branch：

- 走online encoder和OSE使用的projector；
- 不走ReSA predictor；
- 不参与Sinkhorn；
- 不进入queue。

## 4. 最新A5实验：已经完成的代码与设计

### 4.1 四组实验

| 名称 | ReSA loss | OSE | Projector | 目的 |
|---|---:|---:|---|---|
| Shared JMB Q0 | 1 | JMB Q0 + M-F | ReSA/OSE共享 | 复现78.75并产生新诊断日志 |
| ReSA-only | 1 | 关闭 | 只有ReSA head | 测Stage2 ReSA自身贡献 |
| OSE-only JMB Q0 | 0 | JMB Q0 + M-F | 当前共享head语义 | 测OSE自身贡献 |
| Dual-projector JMB Q0 | 1 | JMB Q0 + M-F | ReSA/OSE独立heads，共享encoder | 定位共享projector冲突 |

关键公平性设计：

- ReSA-only设置 `ose_match_exemplar_split=True`，排除与OSE实验完全相同的60个exemplar，避免训练数据量差异；
- OSE-only仍计算ReSA forward和关系指标用于诊断，但 `resa_weight=0`，ReSA不产生实际优化梯度；
- Dual中ReSA和OSE各自拥有online/EMA projector；predictor仍只属于ReSA；
- Dual的两个online projector从完全相同的随机权重开始，随后独立更新，避免初始空间差异污染；
- Dual不增加backbone前向，但增加head参数，且head最终在线性评估中全部丢弃。因此Dual只用于定位优化冲突，不能直接宣称更高效率。

### 4.2 已修改的核心代码

```text
net/ose_resa.py
```

- 新增 `ose_separate_projector`；
- OSE可使用独立 `ose_projector_q/k`；
- ReSA始终使用 `projector_q/k + predictor`；
- OSE exemplar/prototype/mix/queue全部进入所选OSE projector；
- 输出encoder/projector关系、feature std、off-diagonal cosine等诊断。

```text
processor/pretrain_ose_resa.py
```

- 新增 `resa_weight`，默认1；
- `resa_weight=0`实现OSE-only优化；
- 新增 `ose_match_exemplar_split`；
- 为Stage2诊断增加不侵入原训练循环的hooks。

```text
processor/pretrain_ose_resa_stage2.py
```

- 每epoch写一行 `stage2_diagnostics.csv`；
- 每epoch第一个batch额外计算ReSA/OSE原始梯度norm与cosine；
- 记录encoder/projector/predictor相对Stage2初始点的参数漂移；
- 诊断梯度会detach并移到CPU，避免长期占用两份GPU梯度；
- 梯度诊断只在首batch执行，开销有限。

### 4.3 单元测试

新增/扩展：

```text
tests/test_ose_resa_lmix.py
tests/test_ose_resa_stage2.py
```

覆盖：

- 双projector同初始化；
- OSE loss只更新OSE projector与共享encoder，不更新ReSA projector/predictor；
- ReSA/OSE在共享encoder上的梯度余弦可计算；
- Dual的两个实际projector都有梯度。

本地Windows Python没有安装PyTorch，所以当前只完成了Python语法检查和 `git diff --check`；未在本机执行运行时单测。服务器正式长训前必须在有torch的环境运行测试。

## 5. A5配置、输出目录和设备

所有A5 pretrain和LP配置当前均使用：

```yaml
device: [1]
```

### 5.1 Shared JMB Q0

```text
Pretrain config:
config/ntu60/pretext/pretext_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml

LP config:
config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_jmb_q0_mf_xsub_joint.yaml

Output root:
./data/ntu60_cs/aimclr_to_native_ose_resa_a4_jmb_q0_mf_100ep_joint/
```

### 5.2 ReSA-only

```text
Pretrain config:
config/ntu60/pretext/pretext_ose_resa_a4_stage2_resa_only_xsub_joint.yaml

LP config:
config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_resa_only_xsub_joint.yaml

Output root:
./data/ntu60_cs/aimclr_to_native_resa_only_100ep_joint/
```

### 5.3 OSE-only JMB Q0

```text
Pretrain config:
config/ntu60/pretext/pretext_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml

LP config:
config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_ose_only_jmb_q0_mf_xsub_joint.yaml

Output root:
./data/ntu60_cs/aimclr_to_native_ose_only_a4_jmb_q0_mf_100ep_joint/
```

### 5.4 Dual-projector JMB Q0

```text
Pretrain config:
config/ntu60/pretext/pretext_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml

LP config:
config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml

Output root:
./data/ntu60_cs/aimclr_to_native_ose_resa_a4_dualproj_jmb_q0_mf_100ep_joint/
```

每个pretext目录最终应包含：

```text
epoch10_model.pt ... epoch100_model.pt
log.txt
config.yaml
stage2_diagnostics.csv
```

每个LP目录的 `log.txt` 用于读取epoch100 Stage2 checkpoint对应的linear evaluation结果。

## 6. 自动运行脚本与命令

自动脚本：

```text
run_stage2_a5_ablation.sh
```

默认顺序：Shared → ReSA-only → OSE-only → Dual，先顺序跑完四组pretrain，再顺序跑四组LP。

```bash
bash run_stage2_a5_ablation.sh
```

分开运行：

```bash
bash run_stage2_a5_ablation.sh pretrain
bash run_stage2_a5_ablation.sh lp
```

指定训练环境Python：

```bash
PYTHON_BIN=/path/to/python bash run_stage2_a5_ablation.sh all
```

脚本使用 `set -euo pipefail`，任何一组失败都会立即停止。开始前检查Stage1 checkpoint：

```text
./data/ntu60_cs/aimclr_joint/pretext/epoch300_model.pt
```

注意：脚本不是任务管理器，不会识别服务器上是否已有同名任务，也不会安全恢复中断训练。运行前必须先检查GPU1、进程和四个work_dir，避免覆盖已有结果。

服务器测试命令：

```bash
python -m unittest \
  tests.test_ose_resa_lmix \
  tests.test_ose_resa_stage2 \
  tests.test_ose_resa_prototypes
```

## 7. `stage2_diagnostics.csv` 如何解释

### 7.1 Projector适配

重点列：

```text
mean_cluster_kl
mean_encoder_resa_relation_cos
mean_relation_target_pred_cos
mean_relation_top1_agreement
```

支持“ReSA作为relational handoff”的理想现象：

- ReSA-only和Shared中 `mean_cluster_kl` 随epoch下降；
- `mean_encoder_resa_relation_cos` 上升；
- relation target/pred cosine和top1 agreement上升；
- OSE-only中的上述变化明显更弱或更不稳定。

如果OSE-only也同样完成关系适配，或者ReSA-only并没有更好的曲线，就不能把ReSA的缓冲作用写成核心贡献。

### 7.2 梯度冲突位置

重点列：

```text
first_batch_encoder_grad_cos
first_batch_shared_projector_grad_cos
first_batch_resa_encoder_grad_norm
first_batch_ose_encoder_grad_norm
first_batch_resa_projector_grad_norm
first_batch_ose_projector_grad_norm
```

这些是每epoch第一个batch上，未乘总loss权重前的ReSA/OSE原始梯度诊断。`actual_*_grad_norm` 才是当前配置总loss反传后的真实梯度norm。

解释：

- shared projector cosine长期为负，而encoder cosine接近0或为正：冲突主要位于head；
- 两个cosine都长期为负：冲突也进入共享encoder，拆head只能部分缓解；
- Shared负、Dual LP明显提高：支持projector解耦；
- Dual与Shared持平/下降：共享projector不是主要瓶颈，或共享空间本身提供有益耦合；
- OSE-only中ReSA梯度只是“如果启用ReSA会怎样”的只读诊断，不参与实际更新。

预期NaN不是错误：

- ReSA-only没有OSE梯度，因此冲突cosine为NaN；
- Dual没有共享projector，因此 `first_batch_shared_projector_grad_cos` 为NaN；
- Shared中OSE与ReSA共用同一projector，`actual_resa_projector_grad_norm` 与 `actual_ose_projector_grad_norm` 指向同一实际head总梯度，不应当相减。

### 7.3 表示坍塌和双head分化

重点列：

```text
mean_encoder_feature_std
mean_resa_projector_feature_std
mean_ose_projector_feature_std
mean_encoder_offdiag_cos
mean_resa_projector_offdiag_cos
mean_ose_projector_offdiag_cos
mean_resa_ose_relation_cos
```

- feature std接近0且off-diagonal cosine接近1：可能表示坍塌；
- Dual中 `mean_resa_ose_relation_cos` 起点应接近1，因为两个head同初始化；
- Dual中关系逐渐分开且LP提高：专业化可能有益；
- 关系分开但LP下降：两个目标可能过度分裂；
- 始终接近1：拆分projector没有实际必要。

### 7.4 参数漂移

```text
encoder_param_drift
resa_projector_param_drift
ose_projector_param_drift
predictor_param_drift
```

这些值相对每个Stage2 run的初始点。若OSE-only造成远大于ReSA-only的encoder drift且LP更差，说明稀疏类别目标可能过度改写Stage1表示。

## 8. 当前卡在哪里

当前没有已知的代码实现阻塞。阻塞是实验状态未知和结果尚未回传：

1. 不知道用户是否已经在服务器运行 `run_stage2_a5_ablation.sh`；
2. 不知道四组pretrain/LP是否有完成、失败或部分结果；
3. 尚未获得四个 `stage2_diagnostics.csv`；
4. 尚未获得四组固定epoch100 checkpoint的LP结果；
5. 本地没有torch，运行时单测仍需服务器验证。

新会话第一句话应先询问用户：脚本是否已经启动、目前跑到哪一组、是否有报错/日志可提供。不要直接重新运行。

## 9. 下一步计划（严格按顺序）

### 第一步：确认服务器状态

确认：

- GPU1是否可用；
- 是否已有 `main.py` 进程；
- 四个输出目录是否存在；
- 是否已有epoch100 checkpoint、LP日志或CSV；
- 是否发生OOM、单测失败或脚本中止。

### 第二步：完成A5四组结果

需要用户回传：

1. Shared、ReSA-only、OSE-only、Dual的epoch100 LP；
2. 四个 `stage2_diagnostics.csv`；
3. 若任一任务失败，额外提供对应 `log.txt` 最后部分和报错堆栈。

### 第三步：做损失贡献归因

记：

```text
S = Stage1 AimCLR = 75.27
R = ReSA-only
O = OSE-only
F = Shared ReSA+OSE
D = Dual-projector ReSA+OSE
```

主要差值：

```text
OSE在ReSA上的增量 = F - R
ReSA在OSE上的增量 = F - O
Stage2完整增量     = F - S
projector解耦增量  = D - F
```

准确率不是可加线性量，不要把 `F-R-O+S` 当严格因果interaction；它最多是描述性指标，必须结合梯度和关系曲线解释。

决策规则：

- 若 `F > R` 且 `F > O`，ReSA与OSE有互补证据；
- 若 `R ≈ F`，当前主要收益可能来自ReSA，OSE/JMB创新性不足；
- 若 `O ≈ F`，ReSA handoff不是主要收益来源；
- 若 `D > F` 且shared head梯度长期为负，Dual可成为最佳候选；
- 若 `D ≤ F`，保留更简单的Shared主方法。

### 第四步：锁定AimCLR最佳配置

用户当前明确要求：先在AimCLR上把最佳配置验证清楚，再迁移其他baseline。因此A5结束后先选择Shared或Dual，不要立即扩数据集或baseline。

### 第五步：补结构原型的必要消融

A5之后、迁移baseline之前，仍建议完成：

```text
Joint Q0
Joint+Motion Q0
Joint+Bone Q0
Joint+Motion+Bone Q0
JMB Q4（已有78.78，可作邻居消融）
```

尤其缺少Joint Q0。没有它，无法严格隔离JMB提升与去邻居效应。

### 第六步：最终多seed

筛选最佳配置阶段可先固定exemplar seed0；最终至少对主方法和关键基线运行多个exemplar seed，报告mean±std。`78.78-78.75=0.03` 未经重复实验不能宣称邻居有效。

### 第七步：最后才迁移其他baseline/协议

只有AimCLR上的归因、结构消融和稳定性完成后，才考虑第二个预训练器、NTU60 XView或NTU120。否则不能强称“plug-and-play”，最多称“AimCLR的二阶段扩展”。

## 10. 绝对不要再踩的坑

1. **不要再提出warm-up。** 用户已经明确不跑warm-up；A5用loss-only和dual-projector直接验证贡献。
2. **不要现在迁移其他baseline。** 先把AimCLR最佳配置和因果归因做完整。
3. **不要重新拆分正式LR。** 当前backbone/head都固定 `0.25`，分组接口仅作历史消融能力保留。
4. **不要把Q4作为主方法。** JMB下Q4只比Q0高0.03；当前主候选是更简单的Q0。
5. **不要把Q0称queue-free。** 当前代码仍维护未使用的OSE queue；只能称neighbor-free。
6. **不要把ReSA写成向量级 `KL(H || Z)`。** 它对齐的是H生成的关系assignment与Q-Z关系分布。
7. **不要在A5结果出来前宣称ReSA已经证明是“缓冲器”。** 这是待验证机制假设。
8. **不要把Dual的提升简单归因于解冲突而忽略额外head参数。** 必须结合gradient cosine和关系曲线；Dual是定位实验。
9. **不要说OSE-only完全删除了ReSA计算。** 它保留ReSA forward用于诊断，只是 `resa_weight=0`、无优化梯度。
10. **不要让ReSA-only使用不同训练样本数。** `ose_match_exemplar_split=True`是刻意设计，不能删除。
11. **不要混用H/Z/Q。** OSE不用predictor输出，ReSA assignment不用projector偷换encoder关系。
12. **不要迁移AimCLR的旧queue、NNM、DDM、encoder_k或随机滞后的teacher。** Stage2只加载online backbone。
13. **不要预填随机projector生成的queue。** `stage2_prefill_queue=False`。
14. **不要提前enqueue当前batch。** 必须先完成prototype、target、logits和loss，再enqueue teacher Z。
15. **不要把旧A2、旧weak+weak、从头ReSA+OSE P1=79.75与当前二阶段结果做严格单因素差值。** 协议和起点不同。
16. **不要只看单次0.1以内差异。** 当前0.03没有统计意义，多seed后再说。
17. **不要通过反复LP测试挑Stage2 checkpoint。** 正式A5固定epoch100；否则会产生测试集选择偏差。
18. **不要把短smoke、未完成checkpoint或不同total-epoch cosine schedule混成正式结果。**
19. **不要把 `weights + start_epoch` 当完整resume。** optimizer、scheduler、EMA、queue、pointer和RNG未恢复就不是同一训练。
20. **不要盲目重复运行自动脚本。** Processor会复用同名work_dir并可能覆盖日志；先检查服务器进程和结果。
21. **不要删除或回退当前A5代码。** 最新代码已提交，当前HEAD和origin/main均为 `476f968`。
22. **不要擅自管理服务器任务。** 任务状态未知时先问用户，不要停止、重启或覆盖。

## 11. 代码与Git状态

本次交接撰写前：

```text
branch = main
HEAD = 476f968
origin/main = 476f968
commit message = devices change
working tree = clean
```

`476f968` 已包含：

- ReSA-only/OSE-only loss开关；
- Dual-projector；
- Stage2 CSV诊断；
- 四组pretrain与四组LP配置；
- `run_stage2_a5_ablation.sh`；
- 所有A5配置切到 `device: [1]`；
- 相关单元测试。

写入本交接文件后，预期working tree只出现 `handoff.md` 修改，除非用户或其他进程随后产生了新变化。新会话开始仍必须执行：

```bash
git status --short
git diff --check
```

不要假设工作区一直干净，也不要覆盖用户在交接后产生的改动。

## 12. 用户回传结果后应如何回复

不要只给“哪个更高”的结论。应至少输出：

1. 四组LP表格和相对Stage1/Shared差值；
2. ReSA与OSE各自贡献；
3. `cluster_kl`和H-Z relation随epoch趋势；
4. encoder与shared projector gradient cosine的均值、早期/后期变化和负值比例；
5. Dual两head关系是否分化以及是否伴随LP改善；
6. 是否出现feature collapse或异常parameter drift；
7. 最终选择Shared还是Dual及其依据；
8. 下一轮只安排必要的Joint/JM/JB/JMB结构消融，不要立即扩baseline。

当前最简结论：**代码已经准备好，等待A5四组正式训练、LP和CSV；在结果回来之前，不再增加新模块。**
