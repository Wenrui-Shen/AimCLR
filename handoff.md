# AimCLR / ReSA / OSE 骨架迁移任务交接

> 写给完全没有上下文的新会话。最后更新：2026-07-15。

## 1. 我们在做什么

项目目录：`D:\Program\codex\program\AimCLR`

服务器目录：`/home/user9/public3/swr/AimCLR`

服务器 Conda 环境：`swr_aimclr`

目标是在 NTU60 `xsub/joint` 骨架数据上，把 ICLR 2026 论文 **One-Shot Exemplars for Class Grounding in Self-Supervised Learning**（OSE）迁移到 AimCLR/ST-GCN，并与原始 AimCLR、真正的纯 ReSA 做公平比较。

- OSE 页面：https://openreview.net/forum?id=Anv4gdNFaL
- ReSA 官方仓库：https://github.com/winci-ai/resa
- 数据：NTU60 xsub joint
- 数据路径：`../data/pstl/xsub/`
- 当前固定 exemplar seed：0
- 当前首先研究的是 `ReSA + Lproto`，不是完整 OSE。

完整 OSE 目标为：

```text
L = Lcluster + lambda * Lproto + mu * Lmix
Lproto = Lalign + Ldisp
```

当前实现没有 `Lmix`。因此任何实验和论文描述都必须称为 `ReSA + Lproto`，不能称为完整 OSE 复现。

## 2. 当前最重要的结论

### 2.1 真正的 ReSA-only 已经实现，而且 LP 仍只有约 4%

过去把 `ose_lambda=0` 当纯 ReSA 是错误的：exemplar 仍会 forward，并污染 online encoder/projector 的 BN running stats，OSE queue/prototype 也仍会运行。

现在已经实现明确的 `ose_enabled: False`：

- 不选 exemplar；
- 不 forward exemplar；
- 不创建/更新 OSE queue；
- 不计算 prototype、purity、`Lalign`、`Ldisp`；
- 只计算 ReSA `Lcluster`；
- 训练日志会显示：

```text
Training mode | ReSA-only | OSE disabled
```

这个真正的 ReSA-only 完整跑过后，linear probe 仍然只有约 4%。所以此前低性能不是 OSE 残余路径导致的。

### 2.2 dropout 不是根因

用户随后在服务器把 ReSA-only 的 ST-GCN `dropout` 从 `0.5` 改成 `0.0` 重新跑。启动日志确认：

```text
Training mode | ReSA-only | OSE disabled
dropout: 0.0
weights: None
start_epoch: 0
batch_size: 128
```

早期日志大致为：

```text
iter 0   loss 4.8696 | cluster_h 4.7052 | cluster_kl 0.1643 | lr 0.000399
iter 100 loss 4.6295 | cluster_h 4.4787 | cluster_kl 0.1508 | lr 0.040335
iter 200 loss 4.4314 | cluster_h 4.3203 | cluster_kl 0.1111 | lr 0.080272
iter 300 loss 4.4101 | cluster_h 4.2965 | cluster_kl 0.1137 | lr 0.120208
iter 400 loss 4.4431 | cluster_h 4.1746 | cluster_kl 0.2686 | lr 0.160144
iter 500 loss 4.3973 | cluster_h 4.3658 | cluster_kl 0.0315 | lr 0.200080
iter 600 loss 4.3300 | cluster_h 4.3095 | cluster_kl 0.0205 | lr 0.240016
iter 700 loss 4.3687 | cluster_h 4.3413 | cluster_kl 0.0274 | lr 0.25
iter 800 loss 4.3834 | cluster_h 4.3751 | cluster_kl 0.0084 | lr 0.25
```

epoch 3 到至少 epoch 22，mean loss 一直约 `4.37`，`cluster_h` 约 `4.30-4.45`，`cluster_kl` 多数只有 `0.006-0.025`，没有恢复迹象。

结论：去掉 dropout 后仍进入 ReSA assignment collapse，可以停止这条 300 epoch run。dropout 只影响噪声和一致性，无法给近均匀 target 凭空创造语义结构。

### 2.3 当前 ReSA 的直接失败机制

当前代码的 ReSA target 是：

```python
assignment = sinkhorn(online_h[0].detach() @ teacher_h[0].T)
```

随后让跨视图 projector/predictor 特征拟合这张 batch 内 `B x B` assignment。

`B=128` 时均匀分布的最大熵为：

```text
ln(128) = 4.852
```

当前 `cluster_h=4.30-4.45`，相当于每个样本的 assignment 有约 `exp(H)=74-86` 个有效邻居；初始 `4.705` 相当于约 110 个有效邻居。target 非常弥散。

同时 `cluster_kl≈0.01` 不是好现象，它只表示 student 已经很好地复现了这个弥散 target。当前稳定解是：

```text
ST-GCN backbone 特征高度同向/公共方向占主导
    -> batch 相似度矩阵接近常数
    -> Sinkhorn 给出高熵 assignment
    -> student 学会输出同样弥散的关系
    -> KL 很低、loss 很稳，但没有动作类别结构
```

这属于 **assignment/self-training collapse**。目前还不能仅凭 loss 断言 backbone 表征完全常数坍塌；需要补 feature similarity、std 和 effective-rank 诊断。

### 2.4 为什么骨架/ST-GCN 比 ReSA 官方 ImageNet/ResNet 更容易出现此问题

ReSA 是正反馈 SSL。官方方法明确依赖 encoder 输出已经具有稳定、非平凡、语义相关的聚类性质，再用在线自聚类强化它。它不是一个能凭空制造语义结构的强 anti-collapse objective。

当前 ST-GCN 的 `forward_features`：

1. 最后一层输出经过 ReLU，所有维度非负；
2. 对时间、关节和人体做全局平均池化；
3. 再做 L2 normalize 后构建 batch relation。

在训练初期，这容易形成共同正方向/窄锥：不同骨架样本 cosine 都很高，动作的局部时空差异被池化弱化。ReSA target 又完全由自身 encoder 产生，因此没有外部信号打破这个均匀固定点。

此外当前峰值 LR `0.25` 对 ST-GCN 偏激进。日志显示 LR 在 `0.08-0.16` 时 KL 曾达到 `0.11-0.27`、entropy 也有所下降；LR 升至 `0.20-0.25` 后 KL 很快降到约 `0.01` 并锁死。当前判断是：

- backbone 初始关系质量差是根因；
- 过高 LR 加速系统进入均匀固定点；
- dropout 不是根因。

## 3. 已完成的代码工作

当前 Git 状态在写交接前是干净的，最新提交：

```text
d622e0b resa
```

不要再假设以下改动是未提交状态；先以实际 `git status` 和 `git log` 为准。

### 3.1 `net/ose_resa.py`

- 增加构造参数 `ose_enabled=True`；
- 只有启用 OSE 时才注册 OSE queue buffers；
- 将 online 双视图路径与 exemplar 路径拆开；
- ReSA-only 只执行 online/EMA 双视图和 `Lcluster`；
- OSE 启用时保留原先 exemplar forward 顺序；
- ReSA assignment 仍遵循官方结构：backbone relation 做 Sinkhorn，projector relation 做跨视图 soft CE。

### 3.2 `processor/pretrain_ose_resa.py`

- 增加 `--ose_enabled`；
- 将开关注入 model args；
- 启动时明确打印 ReSA-only 或 ReSA+Lproto；
- 只有 OSE 启用时才选 exemplar、排除 exemplar、统计 purity；
- ReSA-only batch 只传两个视图，loss 仅为 `cluster`；
- exemplar cache 增加 seed、class IDs、indices、样本标签、数据集大小等严格校验；
- OSE 模式强制 `return_index: True`，避免 purity 静默错误；
- 支持 `ose_exclude_exemplars: True`，从 `D_u` sampler 排除 60 个 `D_l` exemplar。

### 3.3 `main.py`

增加：

```text
pretrain_resa -> processor.pretrain_ose_resa
```

这只是复用 processor 的入口，真正分支由 `ose_enabled` 控制。

### 3.4 配置文件

- `config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml`
  - `ose_enabled: True`
  - `return_index: True`
  - `ose_exclude_exemplars: True`
- `config/ntu60/pretext/pretext_resa_xsub_joint.yaml`
  - `ose_enabled: False`
  - `return_index: False`
- `config/ntu60/linear_eval/linear_eval_resa_xsub_joint.yaml`
  - 指向 ReSA-only checkpoint。

重要状态差异：当前本地 `pretext_resa_xsub_joint.yaml` 仍写着 `dropout: 0.5`、`base_lr: 0.25`、`resa_final_lr: 0.025`。服务器正在跑的 dropout=0 实验是用户在服务器上改过的版本。本地配置并未同步成 dropout=0。下一会话不要混淆两者。

### 3.5 已完成的静态验证

- Python `py_compile` 通过；
- `git diff --check` 通过；
- 本地没有 NTU 数据，也没有可用的 torch/PyYAML 训练环境，因此没有在 Windows 本地做 forward/runtime smoke test；
- 实际运行验证来自服务器。

## 4. 当前两种训练模式与命令

服务器单卡 GPU 1。使用 `CUDA_VISIBLE_DEVICES=1` 后，配置里必须是逻辑设备 `device: [0]`。

### 4.1 真正的 ReSA-only

```bash
cd /home/user9/public3/swr/AimCLR
conda activate swr_aimclr
CUDA_VISIBLE_DEVICES=1 python main.py pretrain_resa \
  --config config/ntu60/pretext/pretext_resa_xsub_joint.yaml
```

必须在日志看到：

```text
Training mode | ReSA-only | OSE disabled
```

LP：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_resa_xsub_joint.yaml
```

如需测其他 checkpoint，可用 `--weights /absolute/path/epochXXX_model.pt` 覆盖配置。

### 4.2 ReSA + Lproto

```bash
CUDA_VISIBLE_DEVICES=1 python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml
```

LP：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_resa_xsub_joint.yaml
```

## 5. ReSA + Lproto 当前实现细节

### 5.1 维度和训练设置

```text
ST-GCN feature: 256
projector: 256 -> 2048 -> 2048 -> 256
predictor: 256 -> 2048 -> 256
OSE queue: [256, 8192]
batch size: 128
weak + weak views
single GPU
```

当前 256 维、queue 8192、weak+weak 都是迁移设计，不等同于 OSE 图像论文原设置。

### 5.2 OSE prototype

邻居选择 Eq. (2)：

```text
s_c(j) = alpha * sim(exemplar_c, m_j)
         - (1-alpha) * max_{c' != c} sim(exemplar_c', m_j)
```

当前 `alpha=0.75`、`topk=8`。每类 prototype 由 exemplar 和 top-k queue 邻居加权得到。加权后的 prototype 按论文没有再次归一化，当前代码也没有归一化。

### 5.3 温度方向

正确方向：

```text
student tau_s = 0.1
teacher tau_t = 0.04
```

此前曾经写反，已经修复。绝对不要再次颠倒。

### 5.4 queue source

OSE queue 当前写入 `teacher_z[0]`，也就是 EMA encoder + EMA projector 特征。用户曾讨论改成 online feature，随后明确取消。不要擅自把 queue source 改成 online。

### 5.5 `D_l` / `D_u`

60 个固定 exemplar 属于 `D_l`，当前代码会从无标签训练 mini-batch `D_u` 中排除它们。否则 exemplar 可能进入 queue，甚至被选成自己的邻居，虚高 purity。

启动时应看到类似：

```text
OSE unlabeled split | ... samples | excluded 60 exemplars
```

## 6. prototype 分支已经观察到的问题

### 6.1 purity 很低且类别选择性坍缩

随机 purity 是：

```text
1/60 = 1.67%
```

每类 top-8 在随机情况下的期望正确数是：

```text
8/60 = 0.133/8
```

所以每类只有 `0.x/8` 基本仍很差：

- `0.1/8` 接近随机；
- `0.3/8` 仍然很差；
- `0.8/8` 也只有 10% purity。

旧 run 中整体 purity 曾在约 10%-23% 周期跳变，但类别中位数会从 `0.075/8` 突然跳到 `1.08/8`；少数类能达到 `7-8/8`，大量类长期 `0/8`。这是 prototype 周期性重组和类别选择性坍缩，不是稳定语义改善。

这些旧日志来自尚未把 exemplar 从 `D_u` 排除的旧 protocol，只能诊断，不能作为最终结果。

### 6.2 teacher target 过尖锐

旧 ReSA+Lproto run 中 `target_h` 很快降到约 `0.1-0.2`，几乎 one-hot，但许多类别邻居仍是 `0/8`。这是“非常自信但语义错误”的 confirmation bias。

### 6.3 `Ldisp` 为什么会很快到 0 或负数，却没有阻止 collision

当前 `Ldisp` 是所有不同类别 prototype 点积的 off-diagonal 平均，再除以 `tau_s`。它可以正常为负数。

它不能保证没有 prototype/neighbor collision，原因包括：

- 正负 pair 平均时会相互抵消；
- prototype 没有再归一化，范数变小也能让点积降低；
- 少数严重 collision 会被 60 类的 3540 个有序 off-diagonal pair 稀释；
- top-k 索引和 queue 是 detached/discrete 的，`Ldisp` 不直接约束多个类别复用同一批 queue index；
- 它约束的是 prototype 向量平均点积，不是邻居集合的唯一性。

因此 `disp≈0` 或 `<0` 绝不等于“prototype 已经分得很好”。以后应额外记录：

- prototype norm mean/min/max；
- 归一化 prototype cosine 的 max、top-5% mean；
- neighbor unique ratio；
- queue sample reuse histogram/max reuse；
- 类间 neighbor-set Jaccard。

## 7. 如何判断 ReSA 是否坍塌

需要区分三种现象：

1. backbone representation collapse/common-direction concentration；
2. projector/predictor collapse；
3. assignment/self-training collapse。

当前日志已经足以判断第 3 种，但尚不足以严格证明第 1 种。

建议在 `net/ose_resa.py` 给 ReSA 分支加入以下 detached 诊断，不参与 loss：

```python
B = online_h[0].size(0)
eye = torch.eye(B, dtype=torch.bool, device=online_h[0].device)

relation_same = online_h[0].detach() @ teacher_h[0].T
same_pos = relation_same.diag().mean()
same_neg = relation_same[~eye].mean()
same_margin = same_pos - same_neg
same_top1 = (
    relation_same.argmax(dim=1) == torch.arange(B, device=relation_same.device)
).float().mean()

relation_cross = online_h[0].detach() @ teacher_h[1].T
cross_pos = relation_cross.diag().mean()
cross_neg = relation_cross[~eye].mean()
cross_margin = cross_pos - cross_neg
cross_top1 = (
    relation_cross.argmax(dim=1) == torch.arange(B, device=relation_cross.device)
).float().mean()

assignment_entropy_norm = cluster_entropy / math.log(B)
assignment_max_prob = assignment.max(dim=1).values.mean()
h_std = online_h[0].std(dim=0).mean()
z_std = online_z[0].std(dim=0).mean()
h_offdiag_cos = (online_h[0] @ online_h[0].T)[~eye].mean()
```

还应记录 off-diagonal cosine 的 p50/p95/max，以及 centered feature effective rank：

```python
centered = online_h[0] - online_h[0].mean(dim=0, keepdim=True)
s = torch.linalg.svdvals(centered)
p = s / s.sum().clamp_min(1e-12)
effective_rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
```

解释：

- dropout=0 时 `same_pos` 应接近 1；如果 `same_neg` 也接近 1、margin 接近 0，说明 backbone 特征挤在共同方向；
- `h_std -> 0`、offdiag cosine -> 1、effective rank -> 1：backbone 表征坍塌；
- backbone `h` 正常但 `z_std` 很低：projector/predictor 坍塌；
- 高 normalized entropy + 低 KL：assignment collapse；
- Sinkhorn 强制边际平衡，所以只看 row/column marginal 是否均匀不能检测坍塌；
- `cross_top1` 的随机水平是 `1/B=0.78%`（B=128），应显著高于随机并持续提高。

建议 stop rule：连续 3-5 epoch 满足以下条件即可停止，不必跑到 300：

```text
cluster_h / ln(B) > 0.88-0.90
cluster_kl < 0.03
loss 基本不变
relation margin / cross_top1 没有改善
```

当前 dropout=0 run 已满足前 3 项；加入 relation diagnostics 后可完成最后确认。

## 8. 当前真正卡在哪里

不是代码还没有 pure ReSA，而是 **ReSA 直接从随机 ST-GCN 启动时，batch relation target 没有足够的非平凡结构，正反馈机制进入近均匀固定点**。

尚未确定的关键分叉是：

1. 初始 ST-GCN backbone feature 是否已经严重共同方向化；
2. 还是初始 relation 尚可，但 `LR=0.25` 在 warmup 后把它推入坍塌；
3. 或两者同时存在（当前最可能）。

在没有新增 relation/std/rank 诊断前，不要继续凭总 loss 猜测。

## 9. 下一步计划（按顺序，每次只改一个变量）

### Step 1：先加 ReSA collapse diagnostics

优先在 `net/ose_resa.py` 和 processor 日志中加入第 7 节的 detached 指标。不要改变 loss。先对随机初始化和前 3-5 epoch 做检查。

最关键的是：

```text
h offdiag cosine p50/p95
same/cross margin
same/cross top1
h_std / z_std
effective rank
assignment entropy normalized
assignment max probability
```

### Step 2：做唯一变量的低 LR ReSA-only

保持：

```text
dropout = 0.0
batch = 128
weak + weak
feature/projector dimension 不变
```

只改：

```yaml
base_lr: 0.10
resa_final_lr: 0.0
```

与当前 `0.25 -> 0.025` 对照。不要同时改温度、batch、augmentation 或 projector。

如果前 3-5 epoch 仍满足 stop rule，就停止；不要再盲跑 300 epoch。

### Step 3：根据诊断决定 ReSA 的适配方向

如果 epoch 0 就出现 `offdiag cosine≈1`、低 std、低 effective rank：

- 先考虑 relation feature 做 batch centering/去公共方向，再 normalize；
- 或修改用于构建 relation 的 backbone readout，保留更多时空/关节结构；
- 这些都属于骨架域适配，不再是完全原样 ReSA baseline，实验命名必须注明。

如果 epoch 0 relation 有结构，只在 LR 升高后消失：

- 优先保留低 LR；
- 再单独研究更长 warmup 或 momentum schedule。

### Step 4：若 pure ReSA 仍无法从头稳定，采用两阶段方案

用户目前更认可“方案 3”：

1. Stage 1 用可靠的骨架 SSL（优先 AimCLR/InfoNCE）学出非坍塌、语义较好的空间；
2. Stage 2 再用 exemplar 做类别空间 grounding，而不是让单个 exemplar 从随机空间直接定义整个类别。

推荐 Stage 2 不要把每类强行压成单一 cluster：

- Stage 1 先 over-cluster，`K > 60`；
- exemplar 只给高置信 cluster 命名；
- 允许一个类别对应多个 cluster，以容纳视角、主体和动作速度等多模态；
- 低置信样本保留为 reject/dustbin，不强制伪标签；
- 可再做 graph propagation；
- 保留 Stage 1 SSL loss，防止 exemplar grounding 把原空间拉坏；
- 至少用多个 exemplar seed 报均值/方差。

### Step 5：如果继续研究 Lproto，再处理 prototype confirmation bias

必须建立在非坍塌 ReSA/AimCLR 空间上。按单变量顺序考虑：

1. Lproto warmup/ramp，而不是 epoch 0 满权重；
2. top-k schedule，例如 `2 -> 4 -> 8`；
3. teacher temperature warmup或提高 `tau_t`，但必须标记为迁移改进；
4. 降低 `ose_lambda`；
5. 增加 neighbor reuse/collision 诊断；
6. 最后再考虑 strong augmentation、512 维和 `Lmix`。

## 10. exemplar 的其他用法与单样本偏差

曾讨论过“exemplar-relative soft distribution”：对每个样本构造 60 维分布：

```text
r(c | x) = softmax(sim(z_x, z_exemplar_c) / tau)
```

再做跨视图一致性和类别边际平衡，不做 hard top-k。这比当前 hard neighbor mining 平滑，但单 exemplar 仍存在实例偏差：

```text
e_c = class_center_c + instance_bias_c + augmentation_noise_c
```

对同一个 exemplar 做多次增强只能减弱 augmentation noise，无法消除 instance bias。因此如果采用该方案，必须：

- 低权重或后期启用；
- 用 uncertainty/reject；
- 多 seed 评估；
- 不把单 exemplar 当作真实 class center。

用户更倾向两阶段方案，因为它让 exemplar 只负责命名/grounding 已经形成的结构，而不是负责创造整个类别空间。

## 11. 已确认可用的其他基线

原始 AimCLR 的 LP 流程表现正常，旧日志约为：

```text
epoch 5  Top1 70.56%
epoch 10 Top1 72.75%
```

这说明数据、ST-GCN backbone 和 linear evaluation 主流程基本可用。ReSA 的约 4% 不是整个数据/LP pipeline 都坏了。

早期还实现过 AimCLR+OSE（独立 OSE projector/queue、prototype、mixup、ramp），LP 只有约 17%-18%，各 loss 长期接近随机。当前研究方向已从该版本切换到 ReSA+Lproto；不要回头混用旧实现的 queue 或 loss。

## 12. 绝对不要再踩的坑

1. **不要用 `ose_lambda=0` 声称 pure ReSA。** 必须是 `ose_enabled: False`，彻底跳过 exemplar/queue/prototype/BN 副作用。
2. **不要只看 total loss 或低 KL。** 高 target entropy + 低 KL 可以是稳定的均匀坍塌。
3. **不要把 `cluster_h` 高解释成“类别利用均衡”。** `B=128` 时接近 `4.852` 表示 assignment 接近均匀、缺乏区分性。
4. **不要认为 dropout=0 已解决问题。** 最新实验已经否定这一点。
5. **不要继续跑明显满足 stop rule 的 300 epoch 实验。** 先看前 3-5 epoch 的关系诊断。
6. **不要同时改 LR、dropout、augmentation、温度、维度和 loss。** 每次只改一个变量，否则无法归因。
7. **不要把官方 ImageNet/ResNet LR 直接当成 ST-GCN 的正确 LR。** 当前 `0.25` 很可能过高。
8. **不要再写反 OSE 温度。** student `tau_s=0.1`，teacher `tau_t=0.04`。
9. **不要只凭 `Ldisp<=0` 判断 prototype 已分离。** 它只是未归一化 prototype 点积的全局平均，不能约束 neighbor-index collision。
10. **不要把标签传进模型或参与 top-k。** 标签只允许在 forward/top-k 完成后做离线 purity 诊断。
11. **不要让 `D_l` exemplar 留在 `D_u` 和 queue。** 会出现 self-neighbor 并虚高 purity。
12. **不要跨 seed 静默复用 exemplar cache。** cache 必须校验 seed、class、index、标签和数据集大小。
13. **不要擅自把 OSE queue source 改为 online。** 用户已经取消该方案，当前使用 EMA teacher feature。
14. **不要把当前 ReSA+Lproto 称为完整 OSE。** 当前没有 `Lmix`，且 weak+weak、256 维、queue8192 都与论文设置不同。
15. **不要用多卡 DataParallel 跑当前实现。** 本地 ReSA Sinkhorn 和 OSE queue 没有跨卡 all-gather/sync，只有单卡结果可信。
16. **不要把 `start_epoch + weights` 当完整 resume。** checkpoint 只保存 model state，不包含 optimizer/scheduler 完整状态。
17. **protocol、温度、`D_l/D_u` 或 loss 改动后不要续训旧 checkpoint。** 必须从 epoch 0 重跑。
18. **不要混淆本地和服务器配置。** 本地目前仍 `dropout=0.5`；最新 server run 是手动改成 `0.0`。
19. **不要提交论文 PDF、`.Identifier` 或 `Zone.Identifier` 文件。** 修改后检查工作区。
20. **不要把少数类别 purity 很高当整体成功。** 必须看 per-class median、bottom classes、reuse/collision 和 LP。

## 13. 新会话接手后的第一组操作

先确认仓库状态和本地配置：

```bash
git status --short
git log -1 --oneline
git diff --check
```

然后检查：

```bash
grep -n "ose_enabled\|dropout:\|base_lr\|resa_final_lr" \
  config/ntu60/pretext/pretext_resa_xsub_joint.yaml
```

第一项代码任务不是再改 OSE，而是给 pure ReSA 加第 7 节的 collapse diagnostics。完成后：

1. 在服务器同步代码；
2. 建一个明确命名的新配置，固定 `dropout=0.0`、`base_lr=0.10`、`resa_final_lr=0.0`；
3. 从 epoch 0 跑；
4. 观察前 3-5 epoch；
5. 根据 relation/std/rank 指标决定继续、停止或做骨架 relation adaptation；
6. 不要直接再开一个没有诊断的 300 epoch run。
