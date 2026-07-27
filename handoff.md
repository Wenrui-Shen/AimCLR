# AimCLR / ReSA / OSE 研究交接

最后更新：2026-07-27。本文面向完全没有上下文的新会话，只保留当前主线和下一步。

## 1. 当前任务

我们在研究 OSE（One-Shot Exemplar）作为骨架自监督预训练的类别空间模块：每类只使用一个固定带标签 exemplar，其余训练样本无标签。

当前论文主线已经收敛为两个问题：

1. **类别空间缺失**：普通自监督不知道动作类别坐标，OSE 用每类一个 exemplar 建立类别语义。
2. **类别信息没有被可靠地产生和利用**：固定 Top-K 原型存在覆盖与污染权衡；同时 ReSA 的 Sinkhorn 关系仍只由实例特征构造，没有使用 OSE 类别关系。

拟议解法：

- **可靠 OSE 原型**：改进邻居分配和聚合，但不使用置信度。
- **OSE-guided ReSA**：用 OSE 的完整软类别关系修正 ReSA Sinkhorn assignment。

不要再把 corrected instance queue 作为论文主线；它是已实现的探索分支，但设计上给 ReSA 人为增加了一个实例队列后再修正它，不够自然。

## 2. 已有结果与结论

统一正式协议：NTU60 xsub joint、ST-GCN、weak+weak、dropout0、batch128、pretext300、LP200、exemplar seed0。

| 方法 | LP Top-1 |
|---|---:|
| ReSA + OSE M-F，Q0 | 77.22 |
| ReSA + OSE M-F，Q4 | **79.98** |
| ReSA + OSE M-F，Q8 | 78.80 |
| ReSA + OSE M-F，MV4 | 79.47 |
| ReSA + OSE Q4 M-F + corrected instance queue | 77.44 |
| AimCLR A0 | 75.33 |
| AimCLR + OSE MV4 M-F | 约 72.56 |

另有 10-layer CTR-GCN（ST-GCN-matched widths）在 pretext epoch190/300
断电截断后的 LP200=76.15。该结果不是完整 backbone 对照，不能与 ST-GCN Q4=79.98
作正式差值结论。

术语：

- Q0：只有一个 online exemplar。
- Q4/Q8：一个 online exemplar 加 4/8 个无标签 OSE queue 邻居。
- MV4：一个 online exemplar 加 4 个同一 exemplar 的 EMA weak views。
- M-F：`Lmix-proto + Lmix-ins`；完整 ReSA Q4 M-F 为 `Lcluster + Lproto + Lmix-proto + Lmix-ins`。

现有证据只能说明存在**覆盖—污染/稀释权衡**：Q0 覆盖不足，Q8 又不如 Q4。不能直接宣称“Q4 原型准确率更高”，还可能是邻居冗余、聚合方式、prototype norm 或 queue 陈旧造成。

原 OSE 邻居分数已经考虑竞争类别：

```text
g[c,j] = alpha * sim(anchor_c, z_j)
       - (1-alpha) * max_{d != c} sim(anchor_d, z_j)
alpha = 0.75
```

它只是逐类 Top-K 排序，不是通过/拒绝阈值。当前聚合权重又只用 `sim(anchor, component)`，没有继续使用 alpha score。

## 3. 当前代码状态

- 基准提交是 `aad4b01 handoff update`；其后的当前工作区包含尚未提交的 P0-P3 实现。
- corrected weak instance queue 已得到 LP=77.44 的负结果，对应配置已按最新实验收敛要求删除；代码仍保留但不进入 P0-P3。
- ReSA/OSE view 已改用专用 `feeder/ose_resa_feeder.py`。默认按顺序遍历
  `temporal_crop -> shear -> rotation`，每项对每个 view 独立以 p=0.5 触发。
  两个无标签 view 和每个 exemplar view 都重新采样。
- P0/P1/P2/P3 已通过 `ose_prototype_stage: 0/1/2/3` 实现；P1 为互斥邻居，
  P2 为 alpha-consistent aggregation，P3 为最终 prototype normalization。
- 当前只保留 P0-P3 各自的 pretext/LP 配置；旧 ReSA-only、CTR、MV4、
  corrected queue 和 OSE+AimCLR 配置已经删除。原始 AimCLR 代码与原始 AimCLR 配置未修改。
- OSE-guided Semantic ReSA 尚未实现。

重要表示空间：

```text
H = encoder feature；ReSA Sinkhorn 在 H 上构造 BxB 关系
Z = projector feature；OSE exemplar/queue/prototype/category target/M-F 在 Z 上
Q = predictor output；用于 ReSA 跨视图预测
```

不得混用 H/Z/Q。

## 4. 下一步：从阶段二开始的逐步消融

所有阶段固定 Q4、M-F 和统一正式协议；每一步只改一个因素。短跑只用于排错，论文结果必须是从头 pretext300 + LP200。

### 阶段二：可靠原型内部消融

#### P0：当前 Q4 基线

```text
每类独立 Top-4
alpha score 只用于选择
raw anchor similarity 用于五个 component 的 softmax 聚合
聚合后不重新归一化
历史 weak+weak 协议 LP = 79.98
```

注意：新 P0 已统一改为三项增强逐项 p=0.5，因此它是新的增强协议基线，必须重新从头
pretext300 + LP200；不能把历史 weak+weak 的 79.98 直接写成新 P0 的结果。

#### P1：互斥邻居分配

对每个 queue 样本先确定唯一类别：

```text
c*(j) = argmax_c g[c,j]
```

类别 c 只在 `c*(j)=c` 的样本中取 Top-4；同一样本不能进入多个类别。候选不足时只用已有候选和 exemplar，不做置信度阈值，也不强行填入其他类样本。其余聚合保持 P0 不变。

比较 `P1-P0`，只隔离互斥分配贡献。必须记录邻居重复率、各类实际 component 数和只读标签纯度。

#### P2：alpha-consistent aggregation

在 P1 上，让选择和聚合使用同一分数。邻居使用 `g[c,j]`；exemplar 自身分数定义为：

```text
g_anchor[c] = alpha * 1
            - (1-alpha) * max_{d != c} sim(anchor_d, anchor_c)
```

把 `[g_anchor, selected_neighbor_scores]` 一起做 softmax（首版温度保持 1，避免再引入超参），再聚合 component。比较 `P2-P1`。

#### P3：最终 prototype 归一化

在 P2 上只增加：

```text
prototype = normalize(weighted_component_sum)
```

比较 `P3-P2`。若下降，说明 prototype norm 可能携带有用的内部一致性信息，不要为了形式强留归一化。

#### P4：统一 teacher 空间（条件实验）

当前 exemplar 是 online Z，queue 是 EMA Z。先只读统计 online/EMA exemplar cosine；只有差异明显才测试全 EMA exemplar + EMA queue 构造 detached teacher prototype。该变化会影响 `disp` 梯度路径，未明确设计前不要直接实现。

从 P0–P3 选出最优且稳定的版本记为 `P*`。不能根据 seed0 的 0.1–0.2 波动下结论。

### 阶段三：OSE-guided ReSA 的 2x2 因果消融

不使用 entropy confidence、JS gate、阈值或 hard pseudo label。

用两个 EMA weak views 的 OSE 软类别分布取平均：

```text
Pbar = (P_teacher_a + P_teacher_b) / 2       # BxC, detach
G = Pbar @ Pbar.T                            # BxB
S_ins = online_H_a.detach() @ teacher_H_a.T
S_sem = S_ins + lambda_r * (G - 1/C)
A_sem = Sinkhorn(S_sem)
```

当类别分布均匀时 `G=1/C`，严格退化为原 ReSA。只增加一个全局关系强度 `lambda_r`，不增加样本置信度、实例 head、实例 queue 或新 loss。需要把 prototype/category target 计算提前到 ReSA assignment 之前，但必须先读旧 queue、最后再 enqueue。

正式 2x2：

| 实验 | Prototype | ReSA assignment | 目的 |
|---|---|---|---|
| T00 | P0 | 原始 ReSA | 已有基线 79.98 |
| T10 | P* | 原始 ReSA | 可靠原型独立贡献 |
| T01 | P0 | Semantic ReSA | OSE 改进 ReSA 的独立贡献 |
| T11 | P* | Semantic ReSA | 完整核心方法 |

必须计算：

```text
Delta_P = T10 - T00
Delta_R_base = T01 - T00
Delta_R_reliable = T11 - T10
Interaction = (T11-T10) - (T01-T00)
```

若 T10/T01 各自提升但 T11 不再提升，两个模块可能重复，不能宣称互补。

`lambda_r` 可先用现有 checkpoint 对 assignment 做只读扫描（如 0.05/0.1/0.25/0.5），正式训练只跑选定值。标签只用于离线诊断，不进入训练。

### 阶段四：可选第三模块

只有 T11 明确有效后，再单独测试 rival-aware separation，替换当前对全部非对角 prototype similarity 求均值的 `disp_loss`：

```text
L_rival = mean_c tau_r * logsumexp_{d != c}(sim(p_c,p_d) / tau_r)
```

它平滑聚焦最混淆类别。不要同时加入 prototype EMA、guided mix、part/temporal 或 cross-stream 模块，否则无法归因。

### 最终稳定性

架构搜索固定 exemplar seed0。最终至少重跑 T00 和 T11 的 exemplar seed 0/1/2，报告 mean±std 和最差 seed。one-shot 方法不能只报 seed0。

## 5. 实现与测试要求

下一会话先做：

1. `git status --short`、`git diff --check`，复核当前未提交实现。
2. 在有 torch 的服务器运行
   `python -m unittest tests.test_ose_resa_prototypes tests.test_ose_resa_lmix tests.test_ose_resa_queue_corr`。
3. 先用 P0/P1/P2/P3 各做相同 seed 的短 smoke，检查 forward/backward/queue、
   component count 和 overlap 日志；短跑只排错。
4. 动态测试通过后，从头运行新增强协议的 P0 pretext300 + LP200，建立新基线；
   再按 P1、P2、P3 顺序运行，不能沿用历史79.98作为新 P0。
5. 未经用户确认 GPU、配置、work_dir 和预计时长，不启动正式训练。

测试至少覆盖：

- P1 同一个 queue sample 不会分给多个类别；每类最多 4 个；候选不足可安全退化。
- P2 exemplar 和邻居都按 alpha score 聚合，禁用时恢复旧权重。
- P3 打开时 prototype 为单位范数。
- Semantic ReSA 在均匀 P 时与原 similarity/Sinkhorn 一致；`P/G` 全部 detach；不使用标签。
- 不增加 backbone forward；当前 batch 在 logits/assignment 计算后才 enqueue。
- state_dict、EMA、queue 和 backward 正常；不开功能时旧路径完全一致。

## 6. 绝对不要再踩的坑

1. 不要把每类一个带标签 exemplar 的设置称为“完全无监督”；应称 one-shot-assisted / label-efficient self-supervised learning。
2. 不要说“Top-K 越小越好”：Q0=77.22 已反证；也不要把 Q4>Q8 直接等同于更高纯度。
3. 用户明确否决置信度设计：不要再加入 entropy confidence、JS confidence、阈值、置信度 queue 或基于置信度的动态 K。
4. 不要把 corrected/raw instance queue 合并回论文主线；其 LP=77.44，且 ReSA 原生 BxB 关系应直接被 OSE 修正。
5. 正式实验不要合并比较邻居分配、聚合、归一化和 Sinkhorn；虽然 P0-P3 已用累计 stage
   实现，结果仍须严格按 P1-P0、P2-P1、P3-P2 和后续 2x2 消融归因。
6. 不要混用 online/EMA、H/Z/Q；P4 的 teacher prototype 和 `disp` 梯度语义没有设计清楚前不要动。
7. 不要复用或覆盖已有 Q4=79.98、Q0、Q8、MV4、AimCLR work_dir；新 P0-P3 已使用独立配置和目录。
8. 不要混比不同 total-epoch cosine schedule 的同名 checkpoint；短跑只能排错，不能当论文结论。
9. 不要把 `weights + start_epoch` 当完整 resume；optimizer、scheduler、EMA、queue 未恢复会改变实验。
10. ground-truth label 只允许 exemplar 选择和离线诊断，绝不能进入 loss、Top-K、matching 或 Sinkhorn 修正。
11. 不要擅自停止、继续、删除或覆盖服务器任务；新会话默认服务器状态未知。

## 7. 当前交接点

研究方案已经确定到“可靠 Q4 原型 + OSE soft relation 改进 ReSA”，且明确不使用置信度。
三项逐项 p=0.5 的 ReSA/OSE 专用增强以及 P0-P3 已实现并配好独立目录，本地静态编译通过；
本机缺少 torch，动态测试仍需在服务器执行。下一步是服务器单元测试和四阶段短 smoke，
然后先从头建立新增强协议的 P0 正式基线。Semantic ReSA 尚未实现。
