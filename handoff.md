# AimCLR / ReSA / OSE 骨架迁移实验交接

> 写给一个完全没有上下文的新会话。最后更新：2026-07-17。
>
> 本文以最新实验为准。旧交接里“ReSA-only 只有约 4%、已经发生不可恢复坍塌”的判断已经被完整 LP 结果推翻，绝对不要继续引用。

## 1. 我们在做什么

项目目录：`D:\Program\codex\program\AimCLR`

服务器目录：`/home/user9/public3/swr/AimCLR`

服务器 Conda 环境：`swr_aimclr`

任务是在 NTU60 `xsub/joint` 骨架数据上，将 ICLR 2026 论文 **One-Shot Exemplars for Class Grounding in Self-Supervised Learning**（OSE）迁移到 AimCLR/ST-GCN，并与纯 ReSA 做公平比较。

本地有两篇论文 PDF：

- `Cui 等 - 2026 - ONE-SHOT EXEMPLARS FOR CLASS GROUNDING IN SELF-SUPERVISED LEARNING.pdf`
- `Weng 等 - 2025 - Clustering Properties of Self-Supervised Learning.pdf`

当前实现严格说仍是：

```text
ReSA + Lproto

L = Lcluster + lambda * Lproto
Lproto = Lalign + Ldisp
```

论文完整 OSE 还包含：

```text
L = Lcluster + lambda * Lproto + mu * Lmix
Lmix = Lmix-proto + Lmix-ins
```

当前尚未实现 `Lmix`，所以任何结果都应称为 **ReSA+Lproto** 或“当前 OSE 迁移版”，不能称为完整 OSE 复现。

## 2. 当前最重要结论

### 2.1 新版 projector-space prototype 修复有效

旧实现错误地使用不同空间构造原型：

```text
exemplar: online encoder -> projector -> predictor -> Q
queue:    EMA encoder    -> projector              -> Zk
```

随后直接用 `Q_exemplar` 与 `Zk_queue` 做近邻检索并混合成 prototype。论文 Eq. (2)-(4) 要求 exemplar、memory 和 prototype 在同一个 encoder+projector embedding space。

现已修改为：

```text
ReSA Lcluster:
online encoder -> projector -> predictor -> Q
EMA encoder    -> projector              -> Zk
用 Q 对齐 Zk relation

OSE Lproto:
online encoder   -> projector -> Z
exemplar encoder -> projector -> Zl
EMA queue                    -> Zk
用 Zl 检索 Zk queue，用 Z 对齐 prototype
```

修复后，OSE 的 LP 从旧版约 72% 提升到约 75.9%。这是目前最关键、已经得到下游结果验证的代码修复。

### 2.2 纯 ReSA 并没有完全坍塌

`dropout=0, weak+weak` 的纯 ReSA checkpoint 做完整 200-epoch LP 后：

```text
pretext 30  -> LP200 Top-1 48.97
pretext 50  -> LP200 Top-1 50.74
pretext 120 -> LP200 Top-1 54.13
```

表征随 pretext 持续改善。旧日志中高 assignment entropy 或低 KL 不能单独证明 backbone 完全坍塌。旧交接中的“只有 4%”和“应该停止训练”已经失效。

### 2.3 weak+standard(rotation) 严重伤害纯 ReSA

当前 weak augmentation 是：

```text
temporal crop + shear(amplitude=0.5)
```

曾将第二视图改为：

```text
standard = temporal crop + shear + random rotation
```

rotation 与 AimCLR 原实现数值等价：随机主轴 `0~30°`，另外两轴 `0~1°`，每次必定执行，仅采样正角度。

结果：

```text
ReSA weak+standard pretext200 -> LP约40
ReSA weak+standard pretext300 -> LP约38
```

显著低于 weak+weak 的 54.13。因此 ReSA 和 OSE 当前均已恢复 `weak+weak`。rotation 结果只作为负面消融保留，不要再默认它是“官方 standard 所以一定更好”。图像 standard augmentation 与骨架三维旋转不是等价迁移。

### 2.4 purity 不是 LP 的单调代理

新版 prototype 的整体 neighbor purity：

```text
epoch 1   0.090
epoch 10  0.268
epoch 23  0.2966（峰值）
epoch 30  0.296
epoch 60  0.265
epoch 100 0.253
epoch 120 0.244
epoch 143 0.252
```

但 LP：

```text
pretext30  -> 69.73
pretext120 -> 74.77
```

purity 下降时 LP 仍提高 5.04 点。因此不能以“purity 后期回落”作为提前停止或改温度的充分理由。

purity 的确存在类别两极化：少数类别长期接近 `8/8`，一些类别长期接近 `0/8`；这可能影响部分类别和性能上限，但尚未证明它伤害整体 LP。下一步应计算每类 LP accuracy 与每类 purity 的相关性，而不是只优化平均 purity。

### 2.5 当前 `Lcluster+Lproto` 已进入约 75.9% 平台

新版、weak+weak、dropout0、总计划 500-epoch 的探索运行：

```text
pretext30  -> LP200 69.73
pretext120 -> LP200 74.77
pretext175 -> LP200 75.03
pretext200 -> LP200 75.73
pretext250 -> LP200 75.77
pretext300 -> LP200 75.94（该 run 最好）
pretext500 -> LP200 75.86（最终）
```

200 到 500 仅变化约 0.13，300 到 500 下降 0.08，说明继续延长 pretext 已无明显收益。下一步应增加新学习信号（优先 `Lmix`），而不是继续延长 epoch。

## 3. 完整实验结果表

下表所有 checkpoint 数字均指 **pretext epoch**；所有 LP 都完整训练 **200 epoch**。

| 方法 | 原型实现 | 增强 | Dropout | Pretext 计划 | Checkpoint | LP Top-1 |
|---|---|---|---:|---:|---:|---:|
| ReSA-only | 无 OSE | weak+weak | 0 | 300 | 30 | 48.97 |
| ReSA-only | 无 OSE | weak+weak | 0 | 300 | 50 | 50.74 |
| ReSA-only | 无 OSE | weak+weak | 0 | 300 | 120 | 54.13 |
| ReSA-only | 无 OSE | weak+standard(rotation) | 0 | 300 | 200 | 约40 |
| ReSA-only | 无 OSE | weak+standard(rotation) | 0 | 300 | 300 | 约38 |
| ReSA+Lproto | 旧版，Q/Z 空间错配 | weak+weak | 0.5 | 300 | 300 | 71.90 |
| ReSA+Lproto | 旧版，Q/Z 空间错配 | weak+standard(rotation) | 0 | 300 | 300 | 72.39 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 30 | 69.73 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 120 | 74.77 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 175 | 75.03 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 200 | 75.73 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 250 | 75.77 |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 300 | **75.94** |
| ReSA+Lproto | 新版，projector-space | weak+weak | 0 | 500 | 500 | 75.86 |

注意：500-epoch run 的 checkpoint300 使用的是“总计划500”的 cosine LR，不能与总计划300的 checkpoint300 严格等价。

## 4. 当前正在运行什么

用户现在正在服务器上重新训练一条 **总计划只有300 epoch** 的公平新版 OSE：

```text
新版 projector-space prototype
weak+weak
dropout=0
pretext total=300
single GPU
LP total=200
```

目的：与旧 OSE 的 `300/300` 训练计划做严格对比，并验证 75.9% 是否在300 schedule下仍成立。

接手后第一件事应确认服务器启动日志中确实是：

```text
num_epoch: 300
second_view: weak
dropout: 0.0
device: [1]
ose_enabled: True
```

只需要对这条公平 run 做：

```text
checkpoint120 -> LP200（可选，用于早期效率曲线）
checkpoint300 -> LP200（正式结果）
```

不要再为每25或50个epoch密集跑LP。

### 非常重要：work_dir 混用风险

当前本地配置仍写：

```text
./data/ntu60_cs/ose_resa_weak_dropout0_joint/pretext
```

如果服务器300-run也复用这个目录，它会覆盖旧500-run的 `epoch5~300`，但旧 `epoch305~500` 仍可能残留，最终同一目录混合两条run。应优先改成独立目录，例如：

```text
./data/ntu60_cs/ose_resa_weak_dropout0_300ep_joint/pretext
./data/ntu60_cs/ose_resa_weak_dropout0_300ep_joint/linear_eval
```

如果当前运行已经无法重启，至少在记录里明确：`<=300` 是新run，`>300` 是旧run残留，绝对不要混用。

## 5. 当前代码状态与关键实现

写交接前仓库状态是 clean，最新提交：

```text
a98e33d OSEchange
9725427 dropout change and augmentation change
d622e0b resa
```

写入本交接文档后，`handoff.md` 会成为新的未提交修改；不要误删。

### 5.1 `net/ose_resa.py`

当前 `_online_embeddings` 返回三组表示：

```text
online_h：encoder H，构造 ReSA assignment
online_z：projector Z，参与 OSE Lproto
online_q：predictor Q，参与 ReSA Lcluster
```

关键路径：

```python
online_h, online_z, online_q = self._online_embeddings(view_a, view_b)
```

ReSA：

```text
A_H = Sinkhorn(normalize(Hq_weak1) @ normalize(Hk_weak1).T)
Lcluster = row-wise soft CE(Q1 @ Zk2.T, A_H)
         + row-wise soft CE(Q2 @ Zk1.T, A_H)
```

OSE：

```text
exemplar_z = normalize(projector_q(encoder_q(exemplar)))  # 不经过 predictor
queue = teacher_z[0]                                      # EMA projector
student prototype logits = online_z[1] @ prototypes.T
teacher target logits = teacher_z[0] @ prototypes.T
```

不要把 exemplar 再改回 predictor space。

### 5.2 ReSA assignment 和 relation CE

已经核对 ReSA 官方仓库：

- assignment 使用 encoder 特征 `H`，不是 projector/predictor；
- online 分支可选 predictor，官方 ImageNet 默认启用；
- EMA teacher 没有 predictor；
- relation soft CE 是沿每一行计算；
- 双向来自两个跨视图方向，不是再额外加矩阵转置 CE。

当前 row-wise CE：

```python
-(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
```

不要擅自改成：

```text
CE(S, A) + CE(S.T, A.T)
```

那不是官方仓库默认实现，只能作为单独消融。

### 5.3 prototype 构造

当前论文 Eq. (2) discriminative score：

```text
s_c(j) = alpha * sim(exemplar_c, memory_j)
         - (1-alpha) * max_{c' != c} sim(exemplar_c', memory_j)
```

当前：

```text
alpha = 0.75
topk = 8
queue size = 8192
feature dim = 256
```

`alpha` 已经惩罚“同时接近其他类别 exemplar”的样本，但只改变候选排序；固定 top-k 仍会为每类强制选满8个。暂时不要因为平均 purity 低就直接引入阈值。先做因果消融：`k=0/4/8`，看 LP 是否随 purity 改善。

### 5.4 四个温度

```text
sinkhorn_temperature = 0.05  # H relation target 进入 Sinkhorn 前的温度
cluster_temperature  = 0.4   # ReSA Q-Z relation prediction
ose_tau_s            = 0.1   # OSE student prototype distribution，同时缩放 Ldisp
ose_tau_t            = 0.04  # OSE teacher prototype target
```

neighbor top-k 检索本身不直接使用这些温度。温度通过改变训练后的 embedding 间接影响未来 neighbor。

日志中 `target_h` 后期很低、teacher target 接近 one-hot，但 LP 从30到120仍明显提高，因此目前没有证据支持立即修改 `tau_t`。不要仅凭 entropy 修改温度。

### 5.5 数据和 exemplar protocol

- 当前 ReSA/OSE 都使用 `Feeder_double`，`second_view: weak`；
- 两个视图分别独立做 temporal crop + shear；
- OSE 固定 seed0，每类一个 exemplar，共60个；
- exemplar 属于 `D_l`，必须从无标签 `D_u` sampler 排除；
- queue 写入 EMA teacher projector 特征；
- 标签只允许在 top-k 完成后做 purity 诊断，不能参与模型或近邻选择。

启动日志必须看到：

```text
OSE unlabeled split | 40031 samples | excluded 60 exemplars
```

### 5.6 LP 中的 dropout

LP 配置仍可能写 `dropout: 0.5`，但 `processor/linear_evaluation.py` 在训练线性层时执行 `self.model.eval()`，并冻结除 `fc.weight/fc.bias` 外的 backbone 参数。因此 LP 的 `nn.Dropout` 实际不生效；pretext 的 dropout 才会影响特征学习。

## 6. 当前配置摘要

### ReSA-only pretext

`config/ntu60/pretext/pretext_resa_xsub_joint.yaml`

```text
ose_enabled: False
second_view: weak
dropout: 0.0
device: [1]
num_epoch: 300
batch_size: 128
```

### ReSA+Lproto pretext

`config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml`

```text
ose_enabled: True
second_view: weak
dropout: 0.0
device: [1]
num_epoch: 300
batch_size: 128
feature_dim: 256
projector_hidden_dim: 2048
projector_layers: 3
use_predictor: True
queue_size: 8192
ose_topk: 8
ose_alpha: 0.75
ose_tau_s: 0.1
ose_tau_t: 0.04
ose_lambda: 1.0
```

### LP

两份 LP config 当前均：

```text
num_epoch: 200
device: [0]
```

每个 checkpoint 应使用独立 `--work_dir`，否则 LP 日志互相覆盖。

## 7. 常用命令

服务器环境：

```bash
cd /home/user9/public3/swr/AimCLR
conda activate swr_aimclr
```

ReSA-only pretext：

```bash
python main.py pretrain_resa --config config/ntu60/pretext/pretext_resa_xsub_joint.yaml
```

ReSA+Lproto pretext：

```bash
python main.py pretrain_ose_resa --config config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml
```

OSE 指定 checkpoint 做 LP：

```bash
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_ose_resa_xsub_joint.yaml --weights ./data/ntu60_cs/ose_resa_weak_dropout0_joint/pretext/epoch300_model.pt --work_dir ./data/ntu60_cs/ose_resa_weak_dropout0_joint/linear_eval_epoch300
```

ReSA 指定 checkpoint 做 LP：

```bash
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_resa_xsub_joint.yaml --weights ./data/ntu60_cs/resa_only_weak_dropout0_joint/pretext/epoch120_model.pt --work_dir ./data/ntu60_cs/resa_only_weak_dropout0_joint/linear_eval_epoch120
```

命令建议写成单行。之前曾把 `\` 写成独立参数或在它前后放错空格，导致 argparse 报 `unrecognized arguments`。

## 8. 当前卡在哪里

当前不是卡在明显代码 bug，而是卡在两个研究决策：

1. 等待总计划300 epoch的公平新版 OSE 结果，确认当前 75.9% 是否可在与旧版相同 schedule 下复现；
2. 当前 `Lcluster+Lproto` 已进入约75.9%平台，下一步需要实现新的学习信号，最明确的是论文缺失的 `Lmix`。

purity 类别两极化是潜在次级问题，但现有证据显示平均 purity 回落时 LP 仍增长，不能把它当作当前首要瓶颈。

## 9. 下一步计划（按顺序）

### Step 1：完成公平300-run

1. 确认服务器 `num_epoch=300` 且 work_dir 独立；
2. 跑完 pretext300；
3. 对 checkpoint300 完整跑 LP200；
4. 与旧版71.90/72.39和500-run结果75.94/75.86比较；
5. 固定这条结果为新的 `Lcluster+Lproto` baseline。

正式报告不要通过测试集 LP 反复选择“最佳 checkpoint”。探索阶段可画曲线，但正式实验应预先固定300 epoch或使用独立 validation 做选择。

### Step 2：实现完整 OSE 的 `Lmix`

必须做成配置可开关、可单独加权，不能破坏现有 baseline：

```text
baseline: Lcluster + Lproto
ablation A: + Lmix-proto
ablation B: + Lmix-proto + Lmix-ins
```

第一版先按论文实现线性输入 mix，作为复现基线。之后再研究更适合骨架的：

- embedding mix；
- temporal segment mix；
- body-part mix；
- 根节点/骨长对齐后的坐标 mix。

每个新方法先跑到 pretext120 + LP200 做筛选；只有明确超过同 schedule baseline 才继续到300。

### Step 3：判断 neighbor purity 是否真是性能瓶颈

先做：

```text
每类 LP accuracy vs 每类 neighbor purity 的相关性
```

若相关性明显，再做严格单变量：

```text
topk=0（exemplar-only）
topk=4
topk=8（当前基线）
```

只有 purity 和 LP 同时提高，才继续尝试：

- `alpha=0.5/0.6/0.75`；
- exemplar EMA descriptor（EMA只用于检索，当前online exemplar仍用于梯度）；
- mutual-best / own-vs-other margin；
- 自适应 neighbor 数量；
- prototype 中显式 exemplar anchor weight。

### Step 4：骨架域特化

最有研究价值的方向：

1. joint/bone/motion 多流 neighbor score；
2. temporal segment 和 body-part 局部 descriptor；
3. subject/view/骨长归一化，减少 nuisance 相似度；
4. 更强 backbone（CTR-GCN、2s-AGCN等）；
5. 最佳方案跑多个 exemplar seed 和训练 seed。

### Step 5：最后再做容量和系统扩展

- feature dim 256 -> 512；
- queue 8192 -> 16384/更大；
- 多卡 all-gather Sinkhorn 和同步 queue；
- LR、EMA momentum、temperature schedule。

这些都放在 `Lmix` 和主要骨架适配之后，不要现在同时展开。

## 10. 绝对不要再踩的坑

1. **不要再引用“ReSA只有4%/已经完全坍塌”。** 完整LP已经推翻。
2. **不要用 `ose_lambda=0` 冒充纯 ReSA。** 必须 `ose_enabled: False`，彻底跳过 exemplar/queue/prototype/BN 副作用。
3. **不要把 ReSA assignment 改到 projector/predictor。** 官方用 encoder `H` 构造 `A_H`。
4. **不要把 relation CE 擅自改成矩阵转置对称CE。** 官方是 row-wise CE；双向来自两个视图方向。
5. **不要让 OSE exemplar 再经过 predictor。** prototype、memory、OSE student 必须在 projector space；predictor只服务ReSA在线relation预测。
6. **不要把 queue source 改成 online。** 当前使用 EMA teacher projector，用户之前已明确取消online queue方案。
7. **不要让 exemplar 留在无标签 `D_u` 或进入queue。** 会产生self-neighbor并虚高purity。
8. **不要把训练标签传入模型、score或top-k。** 标签只允许在检索完成后做purity诊断。
9. **不要默认 weak+standard(rotation) 更正确。** 它让纯ReSA掉约14~16点；当前主协议是 weak+weak。
10. **不要只看平均purity决定模型好坏。** purity下降时LP仍从69.73升到74.77；还要看LP和per-class。
11. **不要只看 `target_h` 很低就立即改 `tau_t`。** 现有下游结果没有证明尖锐teacher target伤害整体性能。
12. **不要同时改 alpha、k、temperature、lambda、queue和维度。** 每次只改一个变量。
13. **不要把 `alpha` 误解为prototype中exemplar占75%。** 它只用于neighbor discriminative ranking。
14. **不要认为给neighbor score除温度会改变top-k。** 正常数缩放不改变排序。
15. **不要混淆 pretext epoch 和 LP epoch。** 所有已报告LP均完整跑200；30/50/120等是pretext checkpoint。
16. **不要比较不同总epoch schedule下同名checkpoint而不注明。** cosine LR依赖总epoch；`300/500`不等于`300/300`。
17. **不要复用work_dir导致不同run checkpoint混在一起。** 特别是300重跑会与旧500-run残留混合。
18. **不要为每个checkpoint复用同一个LP work_dir。** 日志和best结果会覆盖。
19. **不要把LP config中的dropout=0.5当成实际启用。** LP执行`model.eval()`，dropout关闭。
20. **不要直接用多卡DataParallel。** 当前Sinkhorn和queue没有跨卡all-gather/sync；单卡结果才可信。
21. **不要把 `weights + start_epoch` 当完整resume。** checkpoint未必保存optimizer/scheduler完整状态。
22. **不要在协议或loss改变后续训旧checkpoint。** 应从epoch0重新跑。
23. **不要用测试集反复选择正式checkpoint。** 探索曲线与正式报告要区分。
24. **不要忘记当前仍缺 `Lmix`。** 不能把当前75.9称为完整OSE。

## 11. 新会话接手后的第一组操作

先检查本地：

```bash
git status --short
git log -3 --oneline
git diff --check
```

再检查配置：

```bash
rg -n "work_dir|second_view|dropout:|device:|num_epoch:|ose_topk|ose_alpha|ose_tau" config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml config/ntu60/pretext/pretext_resa_xsub_joint.yaml
```

然后向用户确认/读取服务器300-run的：

```text
启动参数
当前epoch
work_dir
checkpoint300路径
LP300结果
```

如果公平300-run尚未结束，不要中途改代码影响当前实验；可以先设计 `Lmix` 的可开关实现方案。公平run完成并锁定baseline后，下一项代码任务就是实现 `Lmix-proto`，再实现 `Lmix-ins`。
