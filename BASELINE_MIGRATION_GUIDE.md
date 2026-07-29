# OSE–ReSA 创新迁移指南

最后更新：2026-07-28  
用途：把本项目中围绕 OSE、可靠类别原型和 ReSA 关系学习形成的创新，迁移到另一套骨架自监督 baseline。本文面向不了解当前代码历史的实现者，重点说明问题、方法、证据、接口、迁移步骤和实验边界。

## 1. 一句话概括

本项目使用“每类一个带标签 exemplar”给无标签骨架预训练建立类别坐标，再用无标签 EMA queue 扩展 exemplar，构造类别原型并监督实例表示；当前已经验证的关键改进是：**先让每个 queue 样本只归属于一个最匹配类别，再做每类 Top-K，可以减少跨类重复邻居，并在当前协议下把 P0 的 LP Top-1 78.79 提升到 P1 的 79.75。**

这不是完全无监督方法，应称为：

- one-shot-assisted self-supervised learning；或
- label-efficient self-supervised learning。

当前性能证据来自 ReSA + ST-GCN。迁移到其他 baseline 时，应把“通用 OSE 模块”和“ReSA 专属关系模块”分开，不要把所有代码原样复制后直接比较。

## 2. 我们要解决的核心问题

### 2.1 普通 SSL 缺少明确的类别坐标

常规实例级 SSL 能学习样本不变性和相似关系，但并不知道“哪一个方向对应哪一个动作类别”。OSE 为每个类别固定选择一个带标签 exemplar：

```text
E = {e_1, e_2, ..., e_C}
```

标签只用于确定 exemplar 属于哪个类别。其余预训练样本保持无标签。

### 2.2 单 exemplar 覆盖不足，多邻居又可能污染类别

单个 exemplar 只能覆盖一个类别的一小部分姿态和时序变化。使用 queue 邻居扩展原型可以提高覆盖，但存在两个相反风险：

- 邻居太少：类别变化覆盖不足；
- 邻居太多：混入相邻类别、冗余样本或陈旧特征，稀释类别语义。

历史结果中 Q0=77.22、Q4=79.98、Q8=78.80，只能支持“存在覆盖—污染/稀释权衡”，不能直接推断 Q4 的标签纯度一定最高。

### 2.3 独立 Top-K 会让同一样本服务多个类别

原始 Q4 对每个类别独立排序。同一个 queue 样本可能同时进入多个类别原型，导致：

- 类别原型发生重叠；
- 相似类别共享“捷径样本”；
- 类别空间边界不清晰；
- Top-K 数量看似充足，但有效类别覆盖可能并未提高。

P1 的目的就是隔离并解决这个问题。

### 2.4 OSE 类别信息尚未进入 ReSA 的实例关系

当前 ReSA assignment 仍然由 encoder feature 的实例相似度产生：

```text
S_ins = H_online @ H_teacher.T
A = Sinkhorn(S_ins)
```

OSE 只在 projector 空间中产生类别原型和类别分布，尚未直接修正 ReSA 的 B×B 关系。这是后续 OSE-guided Semantic ReSA 要解决的问题，但目前尚未实现、尚无正式结果。

## 3. 哪些内容可以迁移，哪些内容依赖 ReSA

| 模块 | 可迁移性 | 说明 |
|---|---|---|
| 每类一个固定 exemplar | 通用 | 只要求训练集可按类别选样本 |
| 独立随机增强 | 通用 | 每个 view、每个 exemplar view 都重新采样 |
| EMA feature queue | 高 | 适合已有 momentum teacher 的方法；单 encoder 方法需新增 teacher 或重新定义队列来源 |
| P0–P3 类别原型 | 通用 | 工作在 projector feature `Z` 上 |
| `Lproto` | 高 | 需要 student/teacher 类别分布或可替代的 stop-gradient target |
| `Lmix-proto`、`Lmix-ins` | 中高 | 需要明确 mixed sample 的两个来源和 teacher target |
| ReSA Sinkhorn assignment | ReSA 专属 | 工作在 encoder feature `H` 的 B×B 关系上 |
| OSE-guided Semantic ReSA | 关系型 SSL 可迁移 | 目标 baseline 必须存在可解释的样本关系矩阵或 assignment；当前仍是待验证设计 |
| corrected instance queue | 不建议迁移 | 当前结果为 77.44，且会额外引入一套实例 queue，已退出主线 |

迁移原则：先复现目标 baseline，再以独立 wrapper/module 接入 OSE。尽量不修改 baseline 原始文件，使关闭 OSE 时旧路径数值一致。

## 4. 必须保持的表示空间契约

本项目明确区分三个空间：

```text
H = backbone/encoder feature
Z = projector feature
Q = predictor output
```

用途如下：

| 表示 | 当前用途 | 梯度角色 |
|---|---|---|
| `H` | ReSA 实例相似度与 Sinkhorn assignment | online 侧构造 assignment 时 detach；teacher 无梯度 |
| `Z` | exemplar、OSE queue、prototype、类别 target、M-F | online 分支可训练；teacher/queue target detach |
| `Q` | ReSA online 跨视图预测 | 只在 online ReSA 路径训练 |

不要因为维度相同就混用 `H/Z/Q`。迁移时最常见的隐蔽错误，是用 predictor 输出建立 OSE queue，或用 projector 输出替代 baseline 原本在 encoder 空间计算的关系。

建议目标 baseline 提供以下适配接口：

```python
encode_online(x)   -> H_q
project_online(H)  -> Z_q
predict_online(Z)  -> Q_q       # baseline 不需要 predictor 时可省略

encode_teacher(x)  -> H_k
project_teacher(H) -> Z_k
update_teacher(m)
```

如果 backbone 只有 `forward()`，应新增非侵入式 adapter 暴露分类头之前的 `forward_features()`，不要把分类 logits 当作 `H`。

## 5. 当前完整训练数据流

### 5.1 View 构造

当前默认增强序列为：

```text
temporal_crop -> shear -> rotation
```

对序列中的每个增强独立采样 Bernoulli(p=0.5)，不是随机选择一个增强：

```python
for augmentation in pipeline:
    if random() < 0.5:
        x = augmentation(x)
```

两个无标签 view 和每个 exemplar view 都独立重新采样。因此一个 view 可以同时触发 0–3 个增强；默认配置下完全不触发任何增强的概率为 `0.5^3=0.125`。

迁移时必须确认增强操作适合目标数据格式，并保留“逐项遍历、独立触发、按固定顺序组合”的语义。

### 5.2 Online 与 EMA 前向

当前 Q4 M-F、`ose_exemplar_views=1` 的每次迭代包含：

| 分支 | 输入 | backbone 前向次数 |
|---|---|---:|
| online | view A、view B | 2 |
| online | 每类一个 exemplar batch | 1 |
| online | mixed view | 1 |
| EMA | view A、view B | 2 |
| 合计 |  | 6 |

P0–P3 只改变 prototype 内部算法，不改变前向次数。

原始 ReSA 的两个 view 是 online 2 次、EMA 2 次，共 4 次。OSE 增加 exemplar 前向，M-F 增加 mixed-view 前向。迁移时应复用已有中间特征，避免因代码拆分产生重复 backbone forward。

### 5.3 严格的时序

一次迭代应按以下顺序执行：

1. 独立生成 view A、view B、exemplar view；需要 M-F 时生成 mixed view。
2. 计算 online 两个 view 的 `H/Z/Q`。
3. 计算 online exemplar 的 `Z_e`。
4. EMA 更新 teacher 参数。
5. 在 `no_grad` 下计算 teacher 两个 view 的 `H_k/Z_k`。
6. 计算 baseline/ReSA 主损失。
7. **读取旧 queue**，构造类别原型。
8. 计算 `Lproto`、`Lmix-proto`、`Lmix-ins`。
9. 所有当前 batch 的 logits、target 和 assignment 计算完毕后，才把当前 teacher `Z_k` enqueue。
10. 反向只更新 online encoder/projector/predictor 和相应在线分支。

第 7–9 步不能交换，否则当前样本会先进入自己的检索库，造成同批次泄漏和虚假的近邻质量。

## 6. OSE 类别原型

### 6.1 Exemplar 选择和数据边界

每类使用一个固定 exemplar。推荐做法：

- 用独立 `exemplar_seed` 从每类样本中抽取一个索引；
- 缓存 `class_ids`、`indices`、`seed` 和数据集样本数；
- 加载缓存时校验索引仍属于对应类别；
- 从无标签训练 loader 中排除这些 exemplar，避免把已知标签样本又当作普通无标签样本；
- 不同数据集、划分和 seed 使用不同缓存文件。

ground-truth label 只能用于：

- exemplar 选择；
- 离线 purity 等诊断。

标签不得进入 loss、Top-K 打分、类别 matching 或 Sinkhorn 修正。

### 6.2 Queue

当前 OSE queue 保存归一化后的 EMA projector feature：

```text
queue: D × K
queue_sample_indices: K     # 只用于离线诊断
queue_ptr
queue_filled
```

当前正式配置 `K=8192`。迁移时不要直接复用 baseline 的负样本 queue，除非已经验证两者满足相同的表示空间、归一化、更新时序和语义。更安全的做法是单独维护 OSE queue。

### 6.3 竞争分数

对类别 `c` 和 queue 样本 `j`：

```text
g[c,j] = alpha * sim(anchor_c, z_j)
       - (1-alpha) * max_{d != c} sim(anchor_d, z_j)
```

当前 `alpha=0.75`。第一项要求样本接近目标类别，第二项惩罚它同时接近竞争类别。该分数是排序分数，不是置信度，也没有通过/拒绝阈值。

### 6.4 P0–P3 累计阶段

#### P0：Q4 基线

- 每个类别独立按 `g[c,j]` 选择 Top-4；
- 同一 queue 样本可以被多个类别选择；
- prototype components 为一个 online exemplar 加最多四个 EMA queue 邻居；
- 聚合权重使用 `sim(anchor, component)` 的 softmax；
- 聚合后不重新 L2 归一化。

#### P1：互斥邻居分配

先给每个 queue 样本分配唯一 owner：

```text
owner(j) = argmax_c g[c,j]
```

类别 `c` 只在 `owner(j)=c` 的候选中取 Top-4。候选不足时保留现有候选，不从其他类别强行补齐；prototype 至少仍包含 exemplar。

P1 的目标是消除跨类别邻居重叠。它不使用阈值或置信度。

#### P2：选择与聚合分数一致

P1 负责“选谁”，但仍使用 raw anchor similarity 聚合。P2 改为用同一竞争分数聚合 exemplar 和邻居：

```text
g_anchor[c] = alpha * 1
            - (1-alpha) * max_{d != c} sim(anchor_d, anchor_c)

w = softmax([g_anchor, selected_neighbor_scores])
prototype = sum_i w_i * component_i
```

首版 softmax 温度固定为 1，避免额外增加超参数。

#### P3：最终原型归一化

只在 P2 上增加：

```text
prototype = normalize(prototype)
```

P3 必须与 P2 单独比较。如果结果下降，说明 prototype 范数可能携带内部一致性或有效置信信号，不应为了形式统一强制保留归一化。

## 7. OSE 损失与 M-F

### 7.1 原型对齐 `Lproto`

当前实现使用：

```text
student_logits = Z_online_b @ prototypes.T / tau_s
teacher_logits = Z_teacher_a @ stopgrad(prototypes).T / tau_t
teacher_target = stopgrad(softmax(teacher_logits))

Lalign = CE(student_logits, teacher_target)
Ldisp  = mean(off_diagonal(prototypes @ prototypes.T)) / tau_s
Lproto = Lalign + Ldisp
```

当前 `tau_s=0.1`、`tau_t=0.04`。`Ldisp` 压低不同类别 prototype 的平均相似度，但它对所有非对角类别对一视同仁。只有完整核心方法验证有效后，才考虑用 rival-aware separation 聚焦最混淆类别。

### 7.2 Mixed input

令随机 batch permutation 为 `pi`，`beta ~ Beta(a,a)`：

```text
x_mix = beta * view_b + (1-beta) * view_a[pi]
```

当前 `a=1.0`。

### 7.3 `Lmix-proto`

```text
P_student = stopgrad(softmax(student_logits))
P_mix_target = beta * P_student
             + (1-beta) * teacher_target[pi]

Lmix-proto = CE(logits(x_mix, prototypes), P_mix_target)
```

### 7.4 `Lmix-ins`

mixed feature 与当前 batch 的 teacher view A 做实例匹配：

```text
logits_mix_ins = Z_mix @ stopgrad(Z_teacher_a).T / tau_s

target(i) = beta       on index i
          + (1-beta)   on index pi(i)
```

mixed branch 保持在 encoder-projector 空间：

- 不经过 ReSA predictor；
- 不参与 Sinkhorn；
- 不进入 OSE queue；
- 不更新 EMA 分支。

当前完整损失为：

```text
L = Lbaseline/ReSA
  + lambda_proto * Lproto
  + lambda_mix_proto * Lmix-proto
  + lambda_mix_ins * Lmix-ins
```

当前正式配置四部分权重均为 1。迁移初期应保持这些权重，仅在确认损失尺度明显不兼容时再做独立权重消融。

## 8. OSE-guided Semantic ReSA：待迁移的下一阶段

这一模块尚未实现，不能写成已验证贡献。设计目标是让 OSE 产生的完整软类别关系直接修正 ReSA assignment，而不是再增加实例 queue。

计划形式：

```text
Pbar = (P_teacher_a + P_teacher_b) / 2       # B×C, detach
G = Pbar @ Pbar.T                            # B×B
S_ins = H_online_a.detach() @ H_teacher_a.T
S_sem = S_ins + lambda_r * (G - 1/C)
A_sem = Sinkhorn(S_sem)
```

关键性质：当所有类别分布均匀时 `G=1/C`，严格退化为原始 ReSA，不人为改变 baseline。

迁移到非 ReSA baseline 时：

- 如果 baseline 本身优化 B×B relation/assignment，可把居中的 `G-1/C` 作为关系修正项；
- 如果 baseline 只有 pairwise InfoNCE logits，不能直接声称上述公式等价，需要重新定义正负样本关系并做单独消融；
- 如果 baseline 使用 learnable prototypes，必须区分“无语义聚类 prototype”和“OSE 类别 prototype”，不能直接合并索引含义。

## 9. 不同 baseline 的适配策略

### 9.1 已有 online + EMA teacher 的方法

例如 ReSA、MoCo 类、BYOL/DINO 类方法。这是最容易迁移的情况：

1. 复用 online/teacher encoder；
2. 为 OSE 明确选择 projector 空间 `Z`；
3. 从 teacher `Z` 建立独立 OSE queue；
4. online exemplar 构造类别 anchor；
5. 接入 P0 和 OSE/M-F 损失；
6. 再顺序测试 P1、P2、P3。

注意：不同方法的 predictor 语义不同。OSE 不应默认使用 predictor 输出，除非单独证明该空间适合类别原型。

### 9.2 只有单 encoder 的对比方法

例如标准 SimCLR。必须先明确 queue 的稳定来源：

- 推荐：新增 EMA teacher，仅为 stable target 和 OSE queue 服务；
- 备选：使用 detached online feature queue，但这是新的方法变体，陈旧程度和训练反馈会不同。

新增 EMA 本身可能带来收益，因此必须设置：

```text
baseline
baseline + EMA only
baseline + EMA + OSE
```

否则无法把提升归因给 OSE。

### 9.3 已有负样本 queue 的方法

不要默认把负样本 queue 当作 OSE queue。需要逐项确认：

- 是否来自同一个 `Z` 空间；
- 是否已经 L2-normalize；
- 是否由 EMA teacher 产生；
- 是否在本批 loss 后才更新；
- queue 长度和跨卡同步是否满足 OSE 检索；
- 是否保留样本索引用于只读 purity 诊断。

本项目中把 OSE 直接叠加到 AimCLR 的历史结果约为 72.56，低于 AimCLR A0 的 75.33。这说明“已有 queue”不等于“可以无条件复用其语义”。

### 9.4 无 projector 或特征维度不同

建议新增独立 OSE projector，而不是在 backbone 分类 logits 上构造原型。迁移时保持：

- online/teacher projector 结构一致；
- teacher 初始化为 online 的精确拷贝；
- teacher 参数无梯度；
- projector 输出维度固定；
- queue、exemplar 和 prototype 在同一维度和归一化约定下。

## 10. 推荐迁移实施顺序

### 阶段 A：冻结目标 baseline

1. 在目标仓库完整复现原始 baseline。
2. 固定数据划分、增强、backbone、训练 epoch、batch size、优化器和评估协议。
3. 保存 baseline 的正式结果和独立 work directory。
4. 不覆盖 baseline 原始配置和 checkpoint。

### 阶段 B：建立 adapter，不加新损失

1. 暴露 `H/Z/Q` 接口。
2. 若已有 EMA，验证 online/teacher 初始化和更新。
3. 若新增 EMA，先做 `EMA only` 对照。
4. 确认关闭 OSE 时 loss、前向次数和输出与 baseline 一致。

### 阶段 C：接入数据与状态

1. 实现逐项 p=0.5 的独立增强。
2. 固定并缓存 exemplar seed。
3. 从无标签 loader 排除 exemplar。
4. 新增 OSE queue、pointer、filled count 和只读 sample indices。
5. 验证 queue 在 loss 后更新并正确环绕。

### 阶段 D：建立迁移后的 P0

1. 实现 Q4 P0 prototype。
2. 接入 `Lproto`。
3. 分别接入 `Lmix-proto` 和 `Lmix-ins`，确认可以独立开关。
4. 从头完成正式预训练和 linear evaluation，建立目标 baseline 上的新 P0。

### 阶段 E：可靠原型消融

严格依次运行：

```text
P1 - P0：只改变互斥邻居分配
P2 - P1：只改变聚合分数
P3 - P2：只增加 prototype normalization
```

禁止一次把 P1–P3 全打开后只与 P0 比较，否则无法定位贡献。

### 阶段 F：关系引导模块

先从 P0–P3 选出稳定的 `P*`，再做 2×2：

| 实验 | Prototype | baseline relation | 目的 |
|---|---|---|---|
| T00 | P0 | 原始 | 统一基线 |
| T10 | P* | 原始 | 可靠原型独立贡献 |
| T01 | P0 | OSE-guided | 类别关系独立贡献 |
| T11 | P* | OSE-guided | 完整方法 |

计算：

```text
Delta_P = T10 - T00
Delta_R_base = T01 - T00
Delta_R_reliable = T11 - T10
Interaction = (T11-T10) - (T01-T00)
```

如果 T10、T01 各自提升但 T11 不再提升，应解释为模块作用重叠，不能宣称互补。

## 11. 必须记录的诊断指标

除总 loss 和 downstream accuracy 外，至少记录：

- `queue_fill`；
- 每类实际 prototype component 数；
- queue 邻居重复率；
- 每类和全局 neighbor purity，仅作离线诊断；
- `Lbaseline`、`Lproto`、`Lmix-proto`、`Lmix-ins`；
- teacher target entropy；
- alignment CE 与 target entropy 的差值；
- EMA momentum 和 learning rate；
- linear evaluation 的 early、best、best epoch、last accuracy。

P1 中某类 purity 显示 `n/a`，表示该类没有有效 queue 邻居，分母为 0；不等于 purity=0，也不代表训练出现 NaN。若只发生在队列冷启动阶段属于正常现象；若 queue 已满后仍长期出现，说明互斥分配导致类别饥饿，应结合 component count 检查 exemplar 或类别竞争。

linear evaluation 中“早期 acc 提升、最终 acc 接近”通常说明新表示更容易被线性头优化，但最终线性可分上限接近。必须同时报告 best 和 last，不能只挑一个支持结论。

## 12. 当前实验事实与证据等级

### 12.1 当前统一新增强协议

协议：NTU60 xsub joint、ST-GCN、batch 128、pretext 300、LP 200、exemplar seed0，三个增强逐项独立 p=0.5。

| 方法 | LP Top-1 | 状态 |
|---|---:|---|
| P0：Q4 M-F | 78.79 | 已完成 |
| P1：P0 + 互斥邻居 | **79.75** | 已完成，较 P0 +0.96 |
| P2：P1 + alpha-consistent aggregation | 待运行 | 未验证 |
| P3：P2 + prototype normalization | 待运行 | 未验证 |

P1 的 +0.96 是正向单 seed 信号，但最终结论仍需要多 exemplar seed 验证。

### 12.2 历史协议结果

| 方法 | LP Top-1 | 可得结论 |
|---|---:|---|
| ReSA + OSE M-F，Q0 | 77.22 | 单 exemplar 覆盖不足 |
| ReSA + OSE M-F，Q4 | 79.98 | 历史 weak+weak 下最佳 Q 候选 |
| ReSA + OSE M-F，Q8 | 78.80 | 更多邻居不必然更好 |
| ReSA + OSE M-F，MV4 | 79.47 | 多 exemplar view 未超过 Q4 |
| Q4 M-F + corrected instance queue | 77.44 | corrected queue 不进入主线 |
| AimCLR A0 | 75.33 | AimCLR 历史基线 |
| AimCLR + OSE MV4 M-F | 约 72.56 | 直接移植到 AimCLR 失败 |

历史 Q4=79.98 使用旧 weak+weak 协议，不能与新增强 P0=78.79 直接做单因素差值。

10-layer CTR-GCN 在 epoch190/300 因断电中断，checkpoint 的 LP200=76.15。它不是完整预训练结果，不能作为 ST-GCN 与 CTR-GCN 的正式 backbone 对照。

## 13. 单元测试与迁移验收

当前仓库已有测试覆盖：

- 默认三增强与 p=0.5；
- 两个 view 独立采样；
- P0 数值复现旧 Q4；
- P1 queue 样本互斥、候选不足安全退化、overlap=0；
- P2 选择与聚合使用竞争分数；
- P3 prototype 单位范数；
- M-F 两项可独立开关；
- mixed branch 不进入 queue，不训练 predictor/teacher；
- online/EMA 前向次数；
- queue pointer、state dict 和梯度 detach。

迁移后的最低测试集合应包括：

1. `OSE off` 与原 baseline 数值一致。
2. teacher 初始化等于 online，且所有 teacher 参数无梯度。
3. P0 数值与参考公式一致。
4. P1 同一 queue slot 最多属于一个类别，每类最多 K 个邻居。
5. P1 候选不足时 prototype 仍至少含 exemplar。
6. P2 与 P1 选择相同，只改变聚合权重。
7. P3 输出单位范数。
8. `Lmix-proto/Lmix-ins` 可独立启用。
9. mixed view 不进入 queue、Sinkhorn 或 predictor。
10. 当前 batch 在所有 target/logits 计算之后才 enqueue。
11. 完整 loss backward 后 online 有梯度，teacher/target/queue 无梯度。
12. checkpoint 能恢复 online、EMA、projector、queue、pointer、optimizer 和 scheduler。
13. 单卡 smoke 通过后，再验证多卡 all-gather、queue 一致性和 Sinkhorn 范围。

短 smoke 只用于检查 forward/backward/显存/queue，不得当作论文结果。正式比较必须从头使用完整 schedule。

## 14. 容易忽略的实现问题

### 14.1 EMA 与 BatchNorm

`torch.no_grad()` 只禁止梯度，不会自动冻结 BatchNorm running statistics。迁移时必须明确：

- teacher 是否保持 train mode；
- BN buffers 是通过前向更新、EMA 更新还是直接复制；
- exemplar 额外前向是否会污染 teacher BN statistics。

当前实现对额外 EMA exemplar view 保存并恢复 BN buffers，避免少量固定 exemplar 改写 teacher 统计。目标 baseline 若使用 LayerNorm 则没有该问题，但仍需测试 teacher state。

### 14.2 DDP

当前正式配置是单 GPU。迁移到 DDP 时必须决定：

- Sinkhorn/关系矩阵是在本卡 batch 还是全局 batch 上构造；
- teacher feature 是否 all-gather；
- 每张卡的 OSE queue 是否完全一致；
- exemplar 是否在各卡相同；
- enqueue 顺序和 pointer 是否同步。

未处理这些问题时，多卡结果不应与单卡结果直接比较。

### 14.3 完整恢复训练

`weights + start_epoch` 不是完整 resume。至少需要恢复：

- online/EMA encoder；
- online/EMA projector；
- predictor；
- OSE queue、sample indices、pointer、filled count；
- optimizer；
- LR/momentum scheduler 进度；
- AMP scaler（如有）；
- RNG state（如要求严格复现）。

缺少任何一项都可能改变后续训练轨迹。CTR-GCN epoch190 的中断结果就是不能当作完整训练结论的例子。

### 14.4 Linear evaluation

只加载 online backbone；忽略 EMA、projector、predictor 和 queue。冻结 backbone，仅训练新分类头，并确保所有方法使用同一初始化 seed、LR、epoch、评估间隔和数据预处理。

不要把不同训练协议下的 best 混在一张严格消融表中。

## 15. 明确不迁移的设计

以下内容不属于当前论文主线：

- entropy confidence；
- JS confidence/gate；
- 基于阈值的样本接收或拒绝；
- confidence queue；
- 基于置信度的动态 K；
- corrected/raw instance queue；
- 同时加入 prototype EMA、guided mix、part/temporal 和 cross-stream 等多项扩展。

原因不是这些方向永远无效，而是它们增加额外机制、削弱归因；其中 corrected instance queue 已有 77.44 的负结果。

## 16. 迁移完成判定清单

迁移只有同时满足以下条件才算完成：

- 原 baseline 在新代码结构下可复现；
- 关闭 OSE 时旧路径不变；
- exemplar 缓存可复现且不泄漏到无标签 loader；
- `H/Z/Q` 空间映射清楚；
- online/EMA 参数、buffers 和梯度边界正确；
- queue 使用旧状态计算、最后更新；
- P0、P1、P2、P3 可独立配置并逐级退化；
- loss 各项可单独记录和开关；
- 单元测试和短 smoke 通过；
- 完整预训练和统一 linear evaluation 完成；
- 至少完成 P0/P1，之后按顺序完成 P2/P3；
- 最终候选至少使用 exemplar seed 0/1/2，报告 mean±std 和最差 seed；
- 所有结论严格对应同一数据、增强、epoch、backbone 和评估协议。

## 17. 当前推荐决策

如果现在开始迁移到另一套 baseline，推荐把 **P1 Q4 M-F** 作为第一个“候选创新版本”，因为它在当前新增强协议下优于 P0 0.96 个百分点；但同时必须保留 P0 作为目标 baseline 上的直接对照，并继续完成 P2/P3，不能提前把 P1 写成最终版本。

若目标 baseline 不具备 EMA teacher，应先完成 `baseline + EMA only`，再接 OSE；若目标 baseline 没有 B×B relation/assignment，则先只迁移可靠 prototype 和 M-F，不要立即迁移尚未验证的 Semantic ReSA。

## 18. 当前仓库参考入口

- `handoff.md`：研究状态、实验边界与下一步。
- `feeder/ose_resa_feeder.py`：ReSA/OSE 专用 view 构造。
- `net/ose_resa.py`：online/EMA、P0–P3、Lproto、M-F 与 queue。
- `processor/pretrain_ose_resa.py`：exemplar、训练时序、loss 组合与诊断。
- `tests/test_ose_resa_prototypes.py`：增强和 P0–P3 测试。
- `tests/test_ose_resa_lmix.py`：M-F、梯度和前向次数测试。
- `tests/test_ose_resa_queue_corr.py`：历史 corrected queue 测试；只作代码参考，不属于迁移主线。
- `config/ntu60/pretext/pretext_ose_resa_p{0,1,2,3}_q4_mf_xsub_joint.yaml`：四阶段预训练配置。
- `config/ntu60/linear_eval/linear_eval_ose_resa_p{0,1,2,3}_q4_mf_xsub_joint.yaml`：统一线性评估配置。
