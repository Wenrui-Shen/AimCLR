# AimCLR / ReSA + OSE 迁移任务交接

## 1. 任务目标

项目目录：`D:\Program\codex\program\AimCLR`

目标是在 NTU60 `xsub/joint` 骨架数据上，将 ICLR 2026 论文 **One-Shot Exemplars for Class Grounding in Self-Supervised Learning**（下文简称 OSE）迁移到 AimCLR/ST-GCN 代码库，并与 AimCLR、纯 ReSA 做公平比较。

- 论文页面：https://openreview.net/forum?id=Anv4gdNFaL
- 本地 PDF：`Cui 等 - 2026 - ONE-SHOT EXEMPLARS FOR CLASS GROUNDING IN SELF-SUPERVISED LEARNING.pdf`
- ReSA 官方仓库：https://github.com/winci-ai/resa
- 目标数据集：NTU60 xsub joint
- 数据路径：`../data/pstl/xsub/`
- exemplar 必须使用固定随机种子，当前为 seed 0，便于消融。

用户当前要研究的是 **ReSA + Lproto**，不是完整 OSE。完整论文目标为：

```text
L = Lcluster + lambda * Lproto + mu * Lmix
Lproto = Lalign + Ldisp
```

当前实现没有 `Lmix`，这是此前主动决定，不是遗漏。用户最后提出过“prototype 不用 EMA 特征”，随后明确说“算了”，因此不要擅自把 queue source 改成 online。

## 2. 数据与运行环境

服务器项目路径：

```text
/home/user9/public3/swr/AimCLR
```

Python 环境示例：`swr_aimclr`。

数据配置已改为：

```text
../data/pstl/xsub/train_position.npy
../data/pstl/xsub/train_label.pkl
../data/pstl/xsub/val_position.npy
../data/pstl/xsub/val_label.pkl
```

单卡 GPU 1 的正确运行方式是：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py pretrain_ose_resa \
  --config config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml
```

使用 `CUDA_VISIBLE_DEVICES=1` 后，配置中的逻辑设备应为 `device: [0]`。不要同时把配置写成物理 GPU 1，避免设备映射混乱。

LP：

```bash
CUDA_VISIBLE_DEVICES=1 python main.py linear_evaluation \
  --config config/ntu60/linear_eval/linear_eval_ose_resa_xsub_joint.yaml
```

做不同 checkpoint 的 LP 前，修改线性评估配置中的 `weights`。

## 3. 已确认的论文公式

### 3.1 邻居选择 Eq. (2)

对类别 `c` 和 queue 样本 `j`：

```text
s_c(j) = alpha * sim(exemplar_c, m_j)
         - (1-alpha) * max_{c' != c} sim(exemplar_c', m_j)
```

当前 `net/ose_resa.py::_class_prototypes` 与此一致。

### 3.2 prototype 构造 Eq. (3)-(4)

每类由 exemplar 加 top-k 邻居组成，当前 `k=8`。组件权重为：

```text
pi(c,j) = softmax_j(<exemplar_c, q(c,j)>)
prototype_c = sum_j pi(c,j) * q(c,j)
```

论文没有要求对加权后的 prototype 再归一化；当前实现也没有再归一化，这是正确的。

### 3.3 Lalign Eq. (5)-(8)

必须注意温度方向：

```text
student tau_s = 0.1
teacher tau_t = 0.04
```

此前曾经写反，现已修复。当前 student 使用 `online_z[1] / tau_s`，teacher target 使用 `teacher_z[0] / tau_t` 并 stop-gradient。

### 3.4 Ldisp Eq. (9)

当前为不同类别 prototype 两两点积的 off-diagonal 平均，再除以 `tau_s`。`disp` 可以为负数；负数本身不是 bug，代表平均余弦相似度小于 0。

### 3.5 Lmix

论文的 `Lmix` 同时包含：

- prototype perspective：混合样本的 prototype 分布匹配按同一 beta 混合的目标分布；
- instance perspective：混合特征与 key/instance 特征的一致性。

之前 AimCLR+OSE 版本实现过相关逻辑，但当前 ReSA+Lproto 版本刻意不包含 `Lmix`。不要把“当前 ReSA+Lproto”称为完整 OSE 复现。

## 4. 当前模型流程

关键文件：

- `net/ose_resa.py`
- `processor/pretrain_ose_resa.py`
- `feeder/ntu_feeder.py`
- `config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml`
- `config/ntu60/linear_eval/linear_eval_ose_resa_xsub_joint.yaml`

### 4.1 维度

这是用户明确要求的骨架版本配置，不是论文原始 512 维配置：

```text
ST-GCN feature: 256
projector: 256 -> 2048 -> 2048 -> 256
predictor: 256 -> 2048 -> 256
OSE queue: [256, 8192]
```

论文图像实验为 projector output 512；ImageNet queue 65536、CIFAR queue 4096。当前 256/8192 是主动迁移设计。

### 4.2 ReSA Lcluster

当前实现遵循官方 ReSA 代码结构：

1. 两个视图进入 online encoder/projector/predictor。
2. EMA encoder/projector 产生 teacher 特征，不经过 predictor。
3. 使用 backbone 特征关系 `online_h[0] @ teacher_h[0].T` 做 Sinkhorn assignment。
4. 使用跨视图 online embedding 与 teacher embedding 做 soft cross entropy。
5. ReSA 本身不使用 OSE 的 memory queue；它的 assignment 是 batch 内 `B x B`。

### 4.3 OSE Lproto

1. 60 个固定 exemplar 每次做 weak augmentation。
2. exemplar 经 online encoder/projector/predictor 得到 `exemplar_z`。
3. OSE queue 当前写入 `teacher_z[0]`，即 EMA encoder + EMA projector 特征。
4. 按 Eq. (2) 选每类 top-8，按 Eq. (4) 加权构造 prototype。
5. `online_z[1]` 对 prototype 得到 student distribution。
6. `teacher_z[0]` 对 detached prototype 得到 teacher target。
7. 总损失当前为：

```text
loss = Lcluster + ose_lambda * (Lalign + Ldisp)
```

不要在没有新确认的情况下把第 3 步改成 online queue。用户讨论过该消融，但最终取消。

### 4.4 数据增强

当前使用 `Feeder_double`，产生两个独立 weak views；没有 strong view。用户明确要求过 data2、data3 不用强增强。

论文原始设置是 unlabeled weak + strong，labeled exemplar 只 weak。因此当前 weak + weak 是又一项主动偏离论文的迁移设置。

## 5. 当前配置

`config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml` 当前重要参数：

```yaml
feature_dim: 256
projector_hidden_dim: 2048
projector_layers: 3
queue_size: 8192
cluster_temperature: 0.4
sinkhorn_temperature: 0.05
sinkhorn_iterations: 3
dropout: 0.5

base_lr: 0.25
resa_final_lr: 0.025
resa_warmup_epoch: 2
resa_momentum: 0.996
weight_decay: 1e-5
batch_size: 128
num_epoch: 300

ose_exemplar_seed: 0
ose_topk: 8
ose_alpha: 0.75
ose_tau_s: 0.1
ose_tau_t: 0.04
ose_lambda: 1.0
```

LR 为 2 epoch 线性 warmup，然后 cosine `0.25 -> 0.025`；EMA momentum cosine `0.996 -> 1`。

这个 LR 是按 ReSA 图像代码的 batch scaling 推导出来的，但在 ST-GCN 上偏激进。不要把“官方图像 LR”直接当成“骨架上必然正确”。

## 6. 已完成内容

### 6.1 AimCLR 基线

原版 AimCLR 数据路径已改到 `../data/pstl/xsub`。AimCLR LP 日志表现正常：

```text
epoch 5  Top1 70.56%
epoch 10 Top1 72.75%
```

这证明数据、ST-GCN、LP 主流程基本可用。

### 6.2 早期 AimCLR + OSE

曾实现 AimCLR+OSE，包括独立 OSE projector/queue、prototype、mixup 和 ramp。该版本最终 LP 只有约 17%-18%，并且各 loss 长期接近随机附近。当前研究方向已切换到 ReSA+Lproto。

AimCLR+OSE 中 AimCLR queue 和 OSE queue 是两条独立 queue，不是共用一条。

### 6.3 ReSA + Lproto

已实现：

- 3 层 projector 与 predictor；
- EMA encoder/projector；
- ReSA Sinkhorn batch assignment；
- OSE Eq. (2)-(9)；
- 256 维、8192 queue；
- 固定 seed exemplar；
- weak + weak feeder；
- cosine LR / EMA momentum；
- LP 配置；
- 每个 exemplar top-8 邻居的离线真实标签纯度诊断。

### 6.4 离线邻居诊断

训练 feeder 返回样本 index，queue 同时保存 feature 与 sample index。模型完成 top-k 后返回选中的 sample indices，processor 再用 `dataset.label` 统计：

- batch `nn_purity`；
- epoch overall purity；
- 每个类别平均 top-8 中正确邻居数量。

标签只在 forward 完成后用于 CPU 统计，不参与 similarity、top-k、prototype、loss 或梯度。不要为了方便把 label 传入模型。

随机 purity 基线为 `1/60 = 0.0167`；随机情况下每类 top-8 正确邻居期望为 `8/60 = 0.133/8`。

## 7. 当前本地未提交改动

执行交接时 `git status --short` 为：

```text
 M config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml
 M processor/pretrain_ose_resa.py
```

这些改动尚未提交，也很可能尚未同步到服务器：

1. exemplar cache 严格校验 seed、class IDs、indices、样本标签和数据集大小；
2. cache 新增 `num_samples` 元数据；
3. 强制 `return_index: True`，避免纯度诊断静默输出 0；
4. 新增 `ose_exclude_exemplars: True`；
5. 选出 60 个 `D_l` exemplar 后，用 `SubsetRandomSampler` 将它们从 ReSA 的 unlabeled `D_u` mini-batch 中排除。

第 5 点很重要：论文明确将 `D_l` 和 `D_u` 定义为两个集合。旧代码会让 exemplar 自己进入 queue、甚至被选成自己的邻居，虚高 purity。

新版本启动时应看到：

```text
OSE unlabeled split | ... samples | excluded 60 exemplars
```

如果日志参数里没有 `ose_exclude_exemplars`，或者没有上述日志，说明跑的仍是旧代码。

## 8. 当前实验现象

### 8.1 温度修复后的 epoch 1-10

旧服务器代码（尚未排除 exemplar）在正确温度下：

```text
train mean loss: 8.00 -> 5.55
neighbor purity: 5.0% -> 10.05%
cluster_h: ~4.84 -> ~4.2
target_h: ~3.3 -> ~0.5
```

这说明训练不再完全停在随机解，但 teacher target 很快变得极尖锐。

### 8.2 epoch 59-95

同一旧版本后续日志暴露了类别选择性坍缩和周期性重组：

```text
epoch 59 purity 19.76%
epoch 69 purity 11.58%
epoch 70 purity 19.99%  # 突然跳高
epoch 91 purity 10.95%
epoch 92 purity 22.83%  # 再次突然跳高
epoch 95 purity 16.25%
```

更关键的是类别分布：

```text
epoch 91: median 0.075/8，32 个类别低于随机
epoch 92: median 1.080/8，只有 6 个类别低于随机
```

loss 下降时 purity 经常下降；purity 跳高时 proto loss 和 `disp` 会突然增大。模型在反复重组 prototype，而不是稳定地改善语义空间。

长期 `target_h ~= 0.1-0.2`，等价于 teacher distribution 几乎 one-hot，但大量类别邻居仍为 `0/8`。这是“非常自信但语义错误”的 confirmation bias。

`cluster` 大致稳定在 `4.1-4.3`，`cluster_kl` 约 `0.2-0.3`。目前不稳定主要来自 prototype 分支，而不是 ReSA batch assignment 数值爆炸。

注意：上述日志来自旧代码，exemplar 尚未从 `D_u` 排除，因此只能作为诊断，不能作为最终实验结果。

## 9. 当前真正卡住的地方

1. **没有干净的纯 ReSA baseline。** 当前即使设 `ose_lambda=0`，仍会 forward exemplar，并更新 online encoder/projector 的 BN running stats，所以不等价于纯 ReSA。
2. **prototype 语义质量不稳定。** 少数类别可达到 7-8/8，但大量类别长期为 0/8，且 purity 会周期性升降。
3. **teacher target 过度尖锐。** `target_h` 很早接近 0，错误伪标签被快速固化。
4. **当前 LR 可能过高。** epoch 60-95 时 LR 仍约 0.23-0.20，可能加剧 prototype 重组。
5. **dropout 风险。** `dropout: 0.5` 作用在 ST-GCN 多个残差块，不只是分类头；online 与 EMA 分支使用独立随机 mask。ReSA 原始 ResNet 没有这种结构性 dropout。
6. **还没有可靠 LP 对照。** 目前主要看 pretrain loss/purity，最终仍必须用 LP 判断 representation。

## 10. 下一步计划

按以下顺序推进，不要同时改变多个变量：

### Step 1：同步并验证本地最新修复

- 将当前两个未提交改动同步到服务器；
- 从 epoch 0 重跑，不能续训旧 checkpoint；
- 确认日志包含 `ose_exclude_exemplars` 和 `excluded 60 exemplars`；
- 确认 seed0 exemplar indices 与 cache 一致。

### Step 2：补一个真正的 ReSA-only 模式

应增加明确开关，例如 `ose_enabled: False`，做到：

- 不选择/forward exemplar；
- 不构造 OSE queue/prototype；
- 不更新 exemplar 导致的 BN stats；
- 只计算 ReSA `Lcluster`。

不要用简单的 `ose_lambda=0` 冒充纯 ReSA。

### Step 3：做最小公平消融

固定 seed、数据、增强、维度、batch 和 LR：

```text
A. pure ReSA
B. pure ReSA + Lproto
```

先证明 `Lproto` 相对 pure ReSA 有增益，再讨论 Lmix、strong augmentation、512 维等。

### Step 4：学习率消融

建议优先比较：

```text
0.25 -> 0.025  # 当前配置
0.10 -> 0.00   # 更接近 AimCLR/ST-GCN 已验证范围
```

其余设置完全不变。当前日志不能单凭 loss 判断 0.25 好坏，要同时看 purity 分布和 LP。

### Step 5：做 checkpoint LP

旧 run 已保存 epoch 60、70、90 等 checkpoint。可以做诊断性 LP，检查 purity 峰值是否对应更高 top-1，但不要把旧 run 当最终结果。

新 run 至少对相同 epoch 的 pure ReSA 与 ReSA+Lproto 做 LP。

### Step 6：如果 Lproto 仍坍缩

按单变量消融顺序考虑：

1. `dropout: 0.5 -> 0.0`；
2. teacher temperature warmup 或适当提高 `tau_t`，但这会偏离论文固定 `0.04`，必须标成迁移改进；
3. 降低 `ose_lambda`；
4. 监控 prototype 使用率/预测类别直方图，而不仅是 entropy；
5. 最后再考虑 weak+strong、512 维、Lmix。

用户已取消“prototype queue 改成 online feature”的想法，不要把它放在近期默认计划中。

## 11. 绝对不要再踩的坑

1. **不要再写反温度。** student `tau_s=0.1`，teacher `tau_t=0.04`。
2. **不要只看总 loss。** 随机上限：`cluster_h ~= ln(128)=4.852`；`target_h` 接近 0 可能是错误过度自信，不代表学得好。
3. **不要把标签送入模型。** 标签只能在 forward/top-k 完成后做离线 purity 分析。
4. **不要跨 seed 静默复用 exemplar cache。** cache path 要跟 seed/数据集绑定；最新本地代码已增加校验。
5. **不要让 `D_l` exemplar 留在 `D_u` queue。** 会选到自己并虚高 purity；最新本地代码已排除。
6. **不要用 `ose_lambda=0` 声称 pure ReSA。** exemplar forward 仍会污染 BN stats。
7. **不要同时改 LR、dropout、augmentation、维度和 loss。** 每次只做一个变量，否则无法归因。
8. **不要把当前版本叫完整论文复现。** 当前无 Lmix、weak+weak、256 维、8192 queue，均与论文不同。
9. **不要用多卡 DataParallel 跑当前 ReSA/OSE。** Sinkhorn 和 queue 没有跨卡 all-gather/sync，只有单卡结果可信。
10. **不要把模型 checkpoint 当完整 resume。** 当前只保存 model state，不保存 optimizer；`start_epoch + weights` 不等价于连续训练。
11. **不要续训旧 protocol checkpoint。** 温度、诊断和 `D_l/D_u` 划分已经改变，应从 epoch 0 重跑。
12. **不要看到 `disp < 0` 就当异常。** 它可以正常为负。
13. **不要再次增加 `.Identifier` / `Zone.Identifier` 文件。** 每次修改后检查并删除。
14. **不要提交论文 PDF。** `.gitignore` 已包含 `*.pdf` 和 `*Zone.Identifier`。
15. **不要擅自改 prototype queue 为 online。** 用户讨论后明确取消。

## 12. 工程限制与验证状态

- 当前配置为单 GPU，符合 queue/Sinkhorn 实现限制。
- LP 使用 `net.aimclr.AimCLR(pretrain=False)`，只加载相同命名的 `encoder_q` 骨干，分类头重新初始化；ReSA projector/teacher/queue 均由 ignore prefix 过滤，键匹配已检查。
- 当前 checkpoint 只保存模型 state dict，不保存 optimizer。
- 本地 Windows 工作区没有 `../data/pstl/xsub` 数据，因此无法在本地做训练 smoke test。
- 最近一次静态检查：Python 语法通过，`git diff --check` 通过。
- 最近一次检查没有发现 `.Identifier` / `Zone.Identifier` 文件。

## 13. 接手后第一件事

先执行：

```bash
git status --short
git diff -- processor/pretrain_ose_resa.py \
  config/ntu60/pretext/pretext_ose_resa_xsub_joint.yaml
```

确认本交接中提到的两处未提交修复仍在。随后决定如何同步服务器，并优先实现真正的 ReSA-only 开关，而不是继续盲跑当前 ReSA+Lproto 300 epoch。
