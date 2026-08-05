# Relational–Semantic Decoupled Grounding Stage2 迁移规范

最后更新：2026-08-05

状态：当前 Stage2 跨 baseline 迁移的唯一有效文档

适用范围：首先迁移到与 AimCLR 使用相同数据、协议、输入流和 backbone 的其他自监督 baseline；在该范围验证完成前，不把结论扩展到其他数据集、协议或 backbone。

旧版 P0–P3、Q4/P1、corrected queue 和 OSE-guided Semantic ReSA 路线已退出当前主线，不得继续作为迁移默认方案。

## 1. 迁移目标与当前结论

我们研究的不是“给 baseline 叠加几个损失”，而是一个表示粒度过渡问题：

> 如何把实例判别式自监督表示过渡到 one-shot 类别语义空间，同时避免稀疏 prototype 监督过度改写预训练表示中的关系结构？

当前方法暂称 **Relational–Semantic Decoupled Grounding（RSDG，关系—语义解耦式落地）**，包含两个由上述问题直接推导出的解耦：

1. **时间解耦**：Stage1 先完成实例级自监督预训练，Stage2 再进行 one-shot 类别落地。
2. **几何解耦**：Stage2 中 ReSA 和 OSE 使用独立投影空间，但共同更新同一个 online backbone。

当前 NTU60 XSub Joint + ST-GCN + AimCLR 的固定 epoch100 Stage2 checkpoint 结果如下：

| 实验 | LP Top-1 | 相对 Stage1 75.27 | 作用 |
|---|---:|---:|---|
| AimCLR Stage1 | 75.27 | — | 二阶段统一起点 |
| ReSA-only Stage2 | 74.62 | -0.65 | 关系约束独立贡献对照 |
| OSE-only JMB Q0 Stage2 | 78.48 | +3.21 | one-shot 语义落地贡献 |
| Shared ReSA+OSE JMB Q0 Stage2 | 78.75 | +3.48 | 共享投影空间对照 |
| Dual-space ReSA+OSE JMB Q0 Stage2 | **79.15** | **+3.88** | 当前单 seed 最佳候选 |

由此只能安全得出：

- Stage2 的主要准确率收益来自 OSE；
- Shared 中 OSE 梯度幅值明显大于 ReSA，且 shared projector 的参数轨迹接近 OSE-only；
- Dual 允许 ReSA/OSE head 分别接近各自单目标轨迹，并在当前单 seed 上比 Shared 高 0.40、比 OSE-only 高 0.67；
- 当前 first-batch gradient cosine 大部分为正，不能把 Dual 描述为“解决持续负梯度冲突”；
- 0.40/0.67 仍需多 exemplar seed 验证，因此 Dual 是已选出的迁移候选，不是已经统计锁定的最终结论。

## 2. 论文中的统一表述

### 2.1 问题名称

推荐表述：

- representation granularity transition；
- relational–semantic geometry entanglement；
- post-contrastive one-shot class grounding；
- one-shot-assisted self-supervised learning。

不推荐表述：

- fully unsupervised learning；
- generic gradient-conflict resolution；
- plug-and-play，除非已经在多个 baseline 上完成统一协议验证。

### 2.2 ReSA 的角色

ReSA 的损失和 Sinkhorn 机制来自已有工作，不是本文原创。本文对 ReSA 的定位是：

> relational preservation constraint：在 one-shot 类别语义落地时，保留或约束 Stage1 backbone 中已经形成的样本关系结构。

不能把 ReSA 描述为当前主要性能来源，也不能仅凭 `cluster_kl` 下降宣称已经证明 relational handoff。当前 ReSA-only 低于 Stage1，说明关系约束本身不负责创造类别判别性；它只有在与语义落地共同使用时才可能提供互补作用。

### 2.3 Dual-space 的角色

Dual 不是为了增加推理容量，而是把不同粒度的几何约束放在各自的优化坐标空间：

```text
H   = f_theta(x)       shared backbone representation
Z_r = g_r(H)           ReSA relational space
Q_r = p_r(Z_r)         ReSA online prediction
Z_s = g_s(H)           OSE semantic grounding space
```

- `g_r`、`p_r` 只服务 ReSA；
- `g_s` 只服务 OSE、JMB prototype 和 mixed losses；
- `f_theta` 同时接收两种目标的梯度；
- Stage2 完成后丢弃 `g_r/p_r/g_s`，LP 只使用 `f_theta`。

推荐机制表述：

> Dual-space factorization prevents the stronger semantic objective from monopolizing a shared projection geometry, while allowing relational preservation and class grounding to co-regularize one backbone.

不得写成：

> The two losses have persistent negative gradients, and Dual resolves this conflict.

现有诊断支持的是 **semantic gradient dominance / geometry monopolization / head specialization**，不是持续负 cosine。

## 3. 当前固定的 Stage2 主配置

迁移时首先逐项复现以下协议，不得同时修改多个超参数。

| 项目 | 固定值 |
|---|---|
| Stage2 架构 | shared online/EMA backbone + independent ReSA/OSE online/EMA projectors |
| Stage1 迁移内容 | 仅最终 online backbone |
| 不迁移 | Stage1 classifier/projector、predictor、teacher、queue、optimizer、scheduler、RNG |
| ReSA projector | `D_H -> 2048 -> 2048 -> 256`，每个隐藏线性层后 BN+ReLU，输出层后无 affine BN |
| ReSA predictor | `256 -> 2048 -> 256`，中间 BN+ReLU |
| OSE projector | 与 ReSA projector 完全同构、完全同初始化，但之后独立更新 |
| online/EMA 初始化 | online backbone 载入 Stage1；EMA backbone 精确复制 online；每个 EMA projector 精确复制其 online projector |
| Stage2 epochs | 100 |
| batch size | 128，`drop_last=True` |
| optimizer | SGD，momentum 0.9，Nesterov False，weight decay `1e-5` |
| backbone LR | `0.25 -> 0`，逐 iteration cosine |
| head LR | `0.25 -> 0`，逐 iteration cosine |
| warm-up | 0；不要重新引入 warm-up |
| EMA base momentum | 0.996，按 cosine 逐步升到 1 |
| ReSA weight | 1.0 |
| OSE weights | `Lproto=1`、`Lmix-proto=1`、`Lmix-ins=1` |
| exemplar | 每类一个，`exemplar_seed=0` 为首个正式筛选 seed |
| exemplar split | exemplar 从 Stage2 无标签 loader 中排除 |
| prototype | Joint+Motion+Bone，Q0，无无标签邻居 |
| mixed input | `Beta(1,1)` |
| queue prefill | False |
| checkpoint | 每 10 epoch 保存，正式 LP 固定使用 epoch100 |
| device | 当前正式 AimCLR 实验为单 GPU；迁移时先单 GPU 验证 |

当前主配置的关键开关：

```yaml
stage2_load_projector: false
stage2_prefill_queue: false
stage2_head_lr: 0.25
stage2_head_final_lr: 0.0
resa_warmup_epoch: 0
resa_final_lr: 0.0
resa_momentum: 0.996
resa_weight: 1.0

model_args:
  feature_dim: 256
  projector_type: resa
  projector_hidden_dim: 2048
  projector_layers: 3
  use_predictor: true
  ose_separate_projector: true
  queue_size: 8192
  cluster_temperature: 0.4
  sinkhorn_temperature: 0.05
  sinkhorn_iterations: 3

ose_enabled: true
ose_exemplar_seed: 0
ose_exclude_exemplars: true
ose_topk: 0
ose_prototype_stage: 1
ose_exemplar_views: 1
ose_exemplar_modalities: [joint, motion, bone]
ose_alpha: 0.75
ose_tau_s: 0.1
ose_tau_t: 0.04
ose_lambda: 1.0
ose_mix_proto_weight: 1.0
ose_mix_ins_weight: 1.0
ose_mix_alpha: 1.0
queue_contrast_weight: 0.0
```

`ose_alpha`、`ose_prototype_stage` 和 `queue_size` 在 Q0 下不影响 prototype 的邻居选择，但为与当前代码和配置完全一致仍保留。不要利用这些“当前无效参数”暗中改变其他实现语义。

## 4. Stage1 checkpoint 到 Stage2 的迁移契约

### 4.1 只迁移 online backbone

对任意目标 baseline，必须先确定其正式 Stage1 checkpoint 中哪个模块代表最终 online/student backbone。迁移逻辑必须：

1. 解包常见 `state_dict/model_state_dict/model` 容器并删除 `module.` 前缀；
2. 显式列出 Stage2 online backbone 需要的每个 tensor；
3. 从 Stage1 online backbone 逐个按名称映射并检查 shape；
4. 排除 Stage1 分类头或 projector；
5. 严格报告成功迁移的 tensor 数量；
6. 若有缺失或 shape 不一致，立即报错，不允许用静默 `strict=False` 继续正式训练。

AimCLR 当前映射为：

```text
source: encoder_q.* excluding encoder_q.fc.*
target: encoder_q.* excluding encoder_q.fc.*
```

Stage1 的以下状态不得导入 Stage2：

```text
encoder_q.fc / Stage1 projector
encoder_k / momentum teacher
AimCLR queue and queue pointer
NNM/DDM state
optimizer and LR scheduler
```

### 4.2 迁移后重新建立 Stage2 状态

完成 online backbone 载入后：

1. 随机初始化 native ReSA projector 和 predictor；
2. 建立同构 OSE projector；
3. 把 ReSA online projector 参数精确复制到 OSE online projector，保证两个空间起点一致；
4. 把 online backbone/projectors 精确复制到各自 EMA 分支；
5. 所有 EMA 参数 `requires_grad=False`；
6. OSE queue 为空，pointer 和 filled count 为 0；
7. 保存 epoch0 equality 检查结果。

不能载入 baseline 的滞后 teacher，因为它属于旧目标的历史状态；Stage2 边界必须从严格对齐的 online/EMA 状态开始。

### 4.3 Backbone adapter 最低接口

目标 backbone 必须暴露分类头之前的全局特征：

```python
encode_online(x)  -> H_q
encode_teacher(x) -> H_k
```

当前 wrapper 通过 `base_encoder.forward_features(x)` 获得 `H`。其他 baseline 如果只有 `forward()`，应增加非侵入式 adapter；不能把分类 logits 或 Stage1 projector 输出冒充 `H`。

若目标 backbone 输出维度不是 256，仅修改 projector 的输入维度 `D_H`；ReSA/OSE 输出维度仍先保持 256。任何 projector 宽度、深度或输出维度变化都必须作为显式迁移偏差记录。

## 5. H、Z、Q 的表示空间契约

三个空间不得混用：

| 表示 | 来源 | 当前用途 |
|---|---|---|
| `H` | backbone feature | ReSA B×B 相似关系与 Sinkhorn assignment |
| `Z_r` | ReSA projector | ReSA teacher relation prediction目标 |
| `Q_r` | ReSA predictor | ReSA online 跨视图 relation prediction |
| `Z_s` | OSE projector | exemplar、JMB prototype、类别 target、mixed losses、OSE queue |

ReSA 不是向量级 `KL(H || Z)`。当前形式为：

```text
A_H = Sinkhorn(H_q^a @ (H_k^a)^T)
P_r = softmax(Q_r^a @ (Z_r,k^b)^T / tau_cluster)
L_ReSA = CE(A_H, P_r)
```

另一个方向 `b -> a` 同样计算，两个跨视图项取平均。assignment 构造中的 online `H` 必须 detach，teacher 分支始终无梯度。

固定 ReSA 参数：

```text
tau_cluster = 0.4
tau_sinkhorn = 0.05
sinkhorn_iterations = 3
```

OSE 不使用 predictor，mixed branch 不进入 predictor 或 Sinkhorn。

## 6. Stage2 数据、增强和 one-shot split

### 6.1 两个无标签 view

当前增强序列固定为：

```text
temporal_crop -> shear -> rotation
```

每个 view 独立遍历该序列，每个增强独立以 `p=0.5` 触发。一个 view 可以触发 0–3 个增强，不是从三个增强中随机选一个。

当前参数：

```text
temporal padding ratio = 6
shear amplitude = 0.5
rotation = current feeder.tools.random_rotate
```

迁移到同一数据格式时必须复用完全相同的 view 语义。若目标 baseline 原生增强不同，Stage1 保持其原生增强；进入统一 Stage2 后使用本文固定增强，不能让每个 baseline 的 Stage2 继续使用不同增强，否则无法归因迁移效果。

### 6.2 Exemplar 选择与缓存

每类从训练集选择一个带标签 exemplar：

```text
sorted class IDs
NumPy RandomState(exemplar_seed)
one random index per class
```

缓存至少包含：

```text
class_ids
indices
seed
num_samples
dataset/protocol identifier in the path
```

加载缓存时必须检查：seed、类别列表、数据集长度、索引范围以及索引真实标签。每个数据集、协议和 seed 必须使用不同缓存路径。

这些 exemplar 从 Stage2 无标签 sampler 中排除。所有 OSE-only、Shared、Dual 必须使用同一组 exemplar 和同一个无标签 split；ReSA-only 对照也必须通过 `ose_match_exemplar_split=True` 排除完全相同的 exemplar。

标签只能用于 exemplar 选择及离线诊断，不能进入其他无标签样本的 loss、matching 或 Sinkhorn。

### 6.3 JMB label-only prototype

对每个 iteration，每类固定 exemplar 会重新进行一次随机增强。Joint、Motion、Bone 必须由 **同一个增强后的原始 exemplar** 确定性构造，不能分别增强三次。

当前分支：

```text
Joint  -> online backbone + online OSE projector
Motion -> EMA backbone + EMA OSE projector
Bone   -> EMA backbone + EMA OSE projector
```

Motion：

```text
motion[t] = joint[t+1] - joint[t]
last frame = 0
```

Bone 使用 NTU 25-joint 固定骨骼边。迁移到其他数据集时必须替换为该数据集的真实拓扑并单独验证，不能继续使用 NTU 边表。

JMB 融合不增加可调超参数：

```text
z_j, z_m, z_b = L2-normalized embeddings
s_i = cosine(z_i, z_j)
w = softmax([s_j, s_m, s_b])
p_c = normalize(sum_i w_i z_i)
```

Joint 是参考分支。当前 `ose_exemplar_views=1` 表示不再增加额外的同流随机 exemplar view；Motion/Bone 来自 `ose_exemplar_modalities`，不是该参数计数中的独立随机 view。

## 7. OSE 和 mixed losses 的精确形式

### 7.1 Neighbor-free prototype

当前主方法 `ose_topk=0`，prototype 只包含带标签 JMB exemplar，不读取任何无标签 queue 邻居。因此准确称呼是：

> label-only, neighbor-free JMB prototype

当前代码仍然分配并更新 OSE queue，所以不能称为 queue-free。

### 7.2 Prototype alignment

以 `view_b` 的 online OSE embedding 作为 student，以 `view_a` 的 EMA OSE embedding作为 teacher：

```text
student_logits = Z_s,q^b @ P^T / tau_s
teacher_logits = Z_s,k^a @ stopgrad(P)^T / tau_t
teacher_target = stopgrad(softmax(teacher_logits))

L_align = CE(student_logits, teacher_target)
L_disp  = mean(offdiag(P @ P^T)) / tau_s
L_proto = L_align + L_disp
```

固定温度：

```text
tau_s = 0.1
tau_t = 0.04
```

`L_disp` 可以为负，这不是数值错误。

### 7.3 Mixed input

```text
pi ~ random permutation of the batch
beta ~ Beta(1, 1)
x_mix = beta * view_b + (1-beta) * view_a[pi]
```

Mixed prototype target：

```text
P_student = stopgrad(softmax(student_logits))
P_mix = beta * P_student + (1-beta) * teacher_target[pi]
L_mix-proto = CE(logits(Z_s,q(x_mix), P), P_mix)
```

Mixed instance target：

```text
instance_logits = Z_s,q(x_mix) @ stopgrad(Z_s,k^a)^T / tau_s
target row i has beta on i and (1-beta) on pi(i)
L_mix-ins = corresponding two-point soft cross entropy
```

Mixed branch：

- 使用 online backbone 和 online OSE projector；
- 不使用 ReSA projector/predictor；
- 不参与 Sinkhorn；
- 不进入 OSE queue；
- 不更新任何 EMA 参数。

### 7.4 完整目标

当前 Dual 主方法：

```text
L = L_ReSA + L_proto + L_mix-proto + L_mix-ins
```

四项权重均为 1。不要因为 OSE 梯度幅值较大就默认做 GradNorm、loss normalization 或动态权重；这些属于新的方法变体，必须单独命名和消融。

## 8. 一次 iteration 的严格时序

必须遵守以下顺序：

1. 从同一样本独立生成 `view_a`、`view_b`。
2. 构造 permutation、`beta` 和 `x_mix`。
3. 生成每类一个增强后的 raw exemplar，并由它确定性构造 Joint/Motion/Bone。
4. online backbone 计算 `view_a/view_b` 的 `H`。
5. ReSA online projector/predictor 计算 `Z_r/Q_r`；OSE online projector 计算 `Z_s`。
6. online backbone + OSE projector 计算 Joint exemplar。
7. EMA 更新 backbone、ReSA projector、OSE projector。
8. 在 `no_grad` 下计算 teacher `view_a/view_b` 的 `H/Z_r/Z_s`。
9. 在 `no_grad` 下分别计算 Motion/Bone exemplar；额外 exemplar 前向前后恢复 EMA BN buffers，避免固定 exemplar 污染 teacher BN 统计。
10. 构造 ReSA assignment 和跨视图 relation loss。
11. 构造 JMB prototype、`L_proto` 和 teacher target。
12. online backbone + OSE projector 计算 mixed view，得到两项 mixed loss。
13. 所有 logits、target、assignment 和 loss 计算完成后，才 enqueue 当前 teacher `Z_s^a`。
14. 计算总 loss、反向、optimizer step。

当前 JMB+M-F 每次 iteration 的 backbone forward 数：

| 分支 | 输入 | 次数 |
|---|---|---:|
| online | view A、view B | 2 |
| online | Joint exemplar | 1 |
| online | mixed view | 1 |
| EMA | view A、view B | 2 |
| EMA | Motion、Bone exemplar | 2 |
| 合计 |  | 8 |

Dual 相比 Shared 只增加 projector 计算和参数，不增加上述 backbone forward 数。

## 9. LR、EMA 与参数更新边界

### 9.1 LR schedule

backbone 和所有 online head 放在两个 optimizer parameter groups，但当前初始/最终 LR 相同：

```text
backbone: 0.25 -> 0
heads:    0.25 -> 0
```

两者按 iteration progress 使用 100-epoch cosine，无 warm-up。分组接口只是保留实现能力，不能把“分离 LR”写成当前方法。

### 9.2 EMA schedule

基础 momentum 为 0.996，随训练进度升到 1：

```text
m(t) = 1 - (1 - 0.996) * (cos(pi * t/T) + 1) / 2
theta_k = m(t) * theta_k + (1-m(t)) * theta_q
```

### 9.3 梯度边界

必须满足：

- `L_ReSA` 更新 shared online backbone、ReSA online projector 和 predictor；
- OSE 三项 loss 更新 shared online backbone 和 OSE online projector；
- OSE loss 不更新 ReSA projector/predictor；
- ReSA loss 不更新 OSE projector；
- 所有 teacher、prototype target、queue feature 无梯度；
- backbone classifier `fc` 在 Stage2 中被 bypass 且明确冻结。

## 10. Q0 queue 的精确语义

当前实现为了兼容旧 Q4 路径仍：

- 分配 `D x 8192` OSE queue；
- 保存 sample indices、pointer、filled count；
- 在每个 iteration 末尾写入 teacher `Z_s^a`；
- `stage2_prefill_queue=False`，从空 queue 开始。

但 `ose_topk=0` 时 prototype 不读取 queue。因此：

- 可以称 neighbor-free；
- 不可以称 queue-free；
- 跨 baseline 首次迁移应保留该状态更新以匹配当前实现；
- 只有在 AimCLR 上关闭 queue 分配/enqueue 并验证数值与 LP 不变后，才能把真正 queue-free 作为代码简化同步迁移。

不要复用目标 baseline 自带的负样本 queue。

## 11. 跨 baseline 实施步骤

### 阶段 A：冻结并验证 Stage1

1. 使用目标 baseline 原始训练入口复现其正式 checkpoint。
2. 固定数据、协议、stream、backbone、Stage1 epoch、增强和优化器。
3. 用统一 LP 协议得到 `S_b`。
4. 保存 checkpoint key 清单并确认 online backbone 来源。
5. 不覆盖 Stage1 work directory。

### 阶段 B：实现 checkpoint adapter

1. 为目标 baseline 编写独立 `transfer_<baseline>_stage1()`。
2. 只迁移 online backbone，严格 shape check。
3. 输出 loaded/missing/unexpected tensor 报告。
4. 迁移后验证 Stage2 online backbone tensor 与 Stage1 source 逐项完全相等。
5. 验证 EMA backbone 等于 online backbone。

不要在一个通用函数中依赖模糊字符串替换来兼容所有 baseline；每个 baseline 使用显式 key map 更容易审计。

### 阶段 C：建立无损 Stage2 wrapper

1. 暴露 `forward_features()` 或等价 adapter。
2. 接入 native ReSA projector/predictor。
3. 接入独立 OSE projector。
4. 验证两个 online projector 在 optimizer step 之前完全相等。
5. 验证关闭某项 loss 时对应 head 无梯度。

### 阶段 D：接入统一 Stage2 数据

1. 使用固定两-view 增强。
2. 生成并验证 exemplar cache。
3. 从无标签 loader 排除 exemplar。
4. 构造同增强来源的 JMB。
5. 接入 mixed view。
6. 保留当前 Q0 queue 更新时序。

### 阶段 E：先 smoke，再正式训练

Smoke 只验证：

- checkpoint 迁移；
- forward/backward；
- 梯度边界；
- GPU memory；
- checkpoint 保存/加载；
- epoch CSV 正常写入。

Smoke 不能作为论文结果，也不能从 smoke checkpoint 继续拼接正式 cosine schedule。正式 Stage2 必须从同一个 Stage1 checkpoint 重新开始完整 100 epochs。

## 12. Linear probing 迁移规范

### 12.1 加载边界

LP 固定使用 Stage2 `epoch100_model.pt`。只加载：

```text
Stage2 encoder_q backbone parameters and buffers
```

必须忽略：

```text
encoder_q.fc
encoder_k
projector_q / projector_k
predictor
ose_projector_q / ose_projector_k
queue / queue_ptr / queue_filled / queue_sample_indices
```

新建一个与目标 backbone 匹配的 `num_class=C` 分类头。LP 中 backbone 所有参数冻结，只有 classifier weight/bias 可训练。当前 processor 在训练 classifier 时保持整个 model 为 `eval()`，因此 backbone BN running statistics 和 dropout 状态也被冻结。

### 12.2 当前统一 LP 配置

| 项目 | 固定值 |
|---|---|
| input stream | joint |
| train/test preprocessing | clean single view；shear/padding 关闭 |
| epochs | 200 |
| optimizer | SGD，momentum 0.9 |
| LR | 3.0 |
| step | `[80]`，按当前 processor 衰减 0.1 |
| Nesterov | False |
| weight decay | 0 |
| batch size | 128 |
| test batch size | 128 |
| eval interval | 5 epochs |
| classifier initialization | current `weights_init` under process seed 0 |

每个方法必须使用同一个 LP 实现、初始化 seed、训练 epoch 和评估频率。正式表格至少记录 best Top-1、last Top-1 和 best epoch；所有方法的主比较必须使用同一种报告规则。

禁止通过对多个 Stage2 checkpoint 反复 LP 后挑最好结果。当前统一 checkpoint 是 epoch100。

### 12.3 目标 baseline 的 LP model

目标 baseline 的原始 pretraining wrapper 不一定适合 LP。只要满足以下条件，可以使用独立 LP wrapper：

- state dict 中的 backbone key 能从 `encoder_q.*` 精确载入；
- forward 只执行 backbone + 新 classifier；
- 可列出且验证恰好只有 classifier 两个 tensor 可训练；
- backbone 结构、输入 stream 和 BN 语义与 Stage2 一致。

## 13. 每个 baseline 的最小实验矩阵

AimCLR 已完成完整四组归因。迁移到新的 baseline 时，最小正式矩阵为：

| 实验 | 必须程度 | 目的 |
|---|---|---|
| Stage1 baseline LP | 必须 | 得到迁移起点 `S_b` |
| OSE-only JMB Q0 Stage2 | 必须 | 判断时间解耦的语义落地是否可迁移 |
| Shared ReSA+OSE JMB Q0 | 推荐且首个新 baseline 必须 | 验证 Dual 是否仍有必要 |
| Dual-space ReSA+OSE JMB Q0 | 必须 | 完整方法 |
| ReSA-only Stage2 | 首个新 baseline 推荐；后续可选 | 检查关系约束独立行为 |

差值统一报告：

```text
semantic grounding gain = OSE-only - Stage1
ReSA-on-OSE gain        = Dual - OSE-only
space factorization     = Dual - Shared
full Stage2 gain        = Dual - Stage1
```

accuracy 不是可加线性量，不要把四项差值写成严格因果分解。

若新的 baseline 上 `Dual <= OSE-only`，不能只报告完整方法并隐藏 OSE-only；这意味着 ReSA/space factorization 的可迁移性没有得到支持。

## 14. 从现在开始的实验顺序

### Gate 1：先验证 AimCLR 的 Dual 稳定性

当前不是立刻把所有消融全面多 seed，而是先做最小决策集：

```text
exemplar seed 1: OSE-only, Shared, Dual
exemplar seed 2: OSE-only, Shared, Dual
```

已有 seed0，因此新增 6 组 Stage2+LP。该阶段保持训练随机 seed 为当前固定 0，只改变 exemplar seed，并确保同一 exemplar seed 下三组使用同一个 exemplar cache 和无标签 split。

当前仓库可直接运行：

```bash
bash run_stage2_a5_ablation.sh all
```

该脚本保留历史文件名，但当前行为已经切换为上述 Gate 1 六组配对实验；每个 LP 完成后会把最终 `Best Top1` 汇总到：

```text
./data/ntu60_cs/aimclr_stage2_gate1_lp_best_acc.log
```

暂不重复 ReSA-only；其与 Stage1/Shared 的差距足够大，不影响 Shared/Dual 的当前选择。若最终论文需要 ReSA-only mean±std，再在最后补齐。

判断规则：

- 若 Dual 的三 seed 均值高于 Shared 和 OSE-only，且提升不是由单个异常 seed 独占，则锁定 Dual；
- 若 Dual 与 OSE-only 持平或次序频繁反转，保留更简单的 OSE-only/Shared，不能为叙事强行保留 Dual；
- 报告 mean±std 和每个 seed，不只报告最好 seed。

### Gate 2：锁定结构完备 prototype

Dual 通过 Gate 1 后，在 AimCLR 上先用 seed0 完成：

```text
Joint Q0
Joint+Motion Q0
Joint+Bone Q0
Joint+Motion+Bone Q0   # 已有 79.15
```

这一步回答 Motion/Bone 是否分别必要。若 JMB 与次优结构差距很小，再对关键结构补 seed1/2。

为在最终 Dual 架构下支持 neighbor-free 选择，可再做：

```text
Dual JMB Q4 vs Dual JMB Q0
```

已有 Shared JMB Q4/Q0 的 0.03 差距只能作为 Shared 下的证据，不能完全替代最终 Dual 架构中的严格邻居消融。

### Gate 3：再迁移其他 baseline

迁移文档和 adapter 可以现在准备，但正式大规模迁移在 Gate 1 后启动。每个新 baseline：

1. seed0 跑最小矩阵；
2. 完整方法有正收益后，补 OSE-only 和 Dual 的 exemplar seed1/2；
3. 若论文要声称 Dual 在不同 baseline 上普遍必要，则 Shared 也补相同 seeds；
4. 不在迁移初期同时扩展数据集、协议和 backbone。

## 15. Stage2 diagnostics 与验收

每个正式 run 必须输出 `stage2_diagnostics.csv`，每 epoch 一行。至少保留：

```text
mean_cluster_kl
mean_relation_target_pred_cos
mean_relation_top1_agreement
mean_encoder_feature_std
mean_resa_projector_feature_std
mean_ose_projector_feature_std
mean_encoder_offdiag_cos
mean_resa_projector_offdiag_cos
mean_ose_projector_offdiag_cos
mean_resa_ose_relation_cos
first_batch_encoder_grad_cos
first_batch_shared_projector_grad_cos
first_batch_resa/ose_*_grad_norm
encoder/projector/predictor_param_drift
```

解释边界：

- gradient diagnostics 只来自每 epoch 第一个 batch，不能证明所有 batch 的全局冲突状态；
- Dual 没有 shared projector，因此对应 cosine 为 NaN；
- ReSA-only 没有 OSE 梯度，因此冲突 cosine 为 NaN；
- Dual 两个 projector 在任何 optimizer step 前应完全相同；epoch1 CSV 是整轮均值，低于 1 不代表初始化失败；
- `relation_top1_agreement` 在当前 ReSA-enabled run 中接近饱和，不能单独解释 LP；
- 当前 raw `encoder_resa_relation_cos` 并未随训练上升，不能作为 relational handoff 的单独证据；
- feature std 接近 0 且 off-diagonal cosine 接近 1 才是明显 collapse 信号；较高但远低于 1 的 encoder cosine 更适合描述为 semantic contraction/anisotropy。

## 16. 最低单元测试与迁移验收清单

### 16.1 模型与迁移

- Stage1 online backbone 每个目标 tensor 都被加载且数值完全相等；
- Stage1 teacher/projector/queue 没有进入 Stage2；
- Stage2 online/EMA backbone 初始化完全相等；
- Dual 的 ReSA/OSE online projector 初始化完全相等；
- 每个 EMA projector 等于对应 online projector且无梯度；
- backbone classifier 在 Stage2 中冻结且不被 `forward_features()` 使用。

### 16.2 数据与 prototype

- 两个无标签 view 独立抽取每项增强；
- exemplar cache seed、类别、索引和数据集长度校验通过；
- exemplar 从无标签 sampler 排除；
- J/M/B 使用同一个增强后的 raw exemplar；
- Q0 的 neighbor index shape 为 `[C, 0]`；
- JMB prototype 有限且 L2 norm 为 1；
- extra EMA exemplar forward 不改变 teacher BN buffers。

### 16.3 梯度与时序

- ReSA loss 更新 backbone、ReSA projector、predictor，不更新 OSE projector；
- OSE loss 更新 backbone、OSE projector，不更新 ReSA projector/predictor；
- teacher、target、queue 均无梯度；
- mixed branch 不进入 predictor、Sinkhorn 或 queue；
- 当前 batch 在所有 loss 计算之后才 enqueue；
- Dual 与 Shared backbone forward 次数一致。

### 16.4 LP

- 只加载 Stage2 online backbone；
- 所有非 classifier 参数 `requires_grad=False`；
- optimizer 中恰好只有 classifier weight/bias；
- frozen backbone 在 LP train 时保持 eval mode；
- epoch100 checkpoint、200-epoch LP 和统一 seed 可复现。

## 17. 运行、恢复与目录安全

每个 baseline、variant、dataset/protocol 和 exemplar seed 必须使用独立 `work_dir`。开始前检查：

- GPU 与已有进程；
- output directory 是否已存在；
- 是否已有 epoch100 checkpoint、LP log 或 diagnostics CSV；
- config 中 Stage1 checkpoint 和 exemplar cache 是否指向正确 baseline/seed。

当前训练框架中的 `weights + start_epoch` 不等于严格 resume。完整恢复至少需要：

```text
online/EMA backbone
all online/EMA projectors
predictor
queue state and pointers
optimizer
LR/EMA scheduler progress
AMP scaler if used
RNG state
```

缺少任一项就应作为新轨迹，不得与原 run 拼接成一个正式实验。不要盲目重跑自动脚本或覆盖同名目录。

## 18. 明确不进入当前迁移主线的内容

- ReSA/OSE warm-up、冻结 encoder warm-up 或动态 OSE 激活；
- Stage1 projector/head 迁移；
- Stage1 queue、teacher、NNM/DDM 迁移；
- Q4/P1/P2/P3 作为主方法；
- OSE-guided Semantic ReSA；
- corrected/category confidence instance queue；
- queue prefill；
- 动态 loss weighting、PCGrad/GradNorm；
- 同时迁移多个数据集、协议、stream 或 backbone；
- 通过反复 LP 选择 Stage2 checkpoint。

这些方向不是永远无效，而是尚未属于当前已经归因清楚的 RSDG Stage2 方法。

## 19. 当前仓库参考入口

- `handoff.md`：AimCLR 当前研究状态、A5 结果解释和实验边界。
- `net/ose_resa.py`：Dual projector、ReSA、JMB prototype、OSE/M-F 和 queue 时序。
- `processor/pretrain_ose_resa.py`：exemplar split、JMB stream、完整 loss 和训练循环。
- `processor/pretrain_ose_resa_stage2.py`：AimCLR checkpoint 迁移、Stage2 optimizer/schedule 和 diagnostics。
- `feeder/ose_resa_feeder.py`：统一两-view 增强。
- `processor/linear_evaluation.py`：当前 LP 冻结与优化协议。
- `config/ntu60/pretext/pretext_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml`：当前主候选 Stage2 配置。
- `config/ntu60/linear_eval/linear_eval_ose_resa_a4_stage2_dualproj_jmb_q0_mf_xsub_joint.yaml`：当前主候选 LP 配置。
- `run_stage2_a5_ablation.sh`：当前 Gate 1 seed1/2 配对预训练、LP 与 Best Top-1 汇总脚本；文件名因兼容历史命令而保留。
- `tests/test_ose_resa_stage2.py`：checkpoint transfer、JMB 同增强和 Stage2 LR 测试。
- `tests/test_ose_resa_lmix.py`：mixed loss、Dual 初始化与梯度隔离测试。

## 20. 迁移完成的定义

只有同时满足以下条件，才算某个 baseline 已准确迁移：

1. Stage1 原结果和统一 LP 起点已记录；
2. checkpoint adapter 完整、显式且经过 tensor equality 检查；
3. Stage2 使用 native random Dual heads、JMB Q0、M-F、100 epochs 和固定 schedule；
4. exemplar cache 与无标签 split 在所有对照间完全一致；
5. 单元测试、单卡 smoke、epoch100 checkpoint 和 diagnostics CSV 完整；
6. LP 只加载 online backbone，并按统一 200-epoch 协议执行；
7. 至少完成 Stage1、OSE-only、Dual；首个新 baseline 还必须完成 Shared；
8. seed0 有正向结果后补 exemplar seed1/2，并报告 mean±std；
9. 结论使用本文第2节的表示边界，不把结果夸大为完全无监督、负梯度冲突已解决或已证明 plug-and-play。
