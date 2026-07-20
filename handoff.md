# AimCLR / ReSA / OSE / CTR-GCN 实验交接

> 面向完全没有上下文的新会话。最后更新：2026-07-19。
>
> 当前没有修改训练代码；本轮只检查了 handoff、本地实现以及 CTR-GCN、
> SCD-Net、HiCo 的 GitHub 代码。不要未经用户确认直接启动长期训练或改变实验协议。

## 1. 我们在做什么

项目位置：

```text
本地：D:\Program\codex\program\AimCLR
服务器：/home/user9/public3/swr/AimCLR
Conda：swr_aimclr
```

实验协议：

```text
Dataset: NTU60
Protocol: xsub
Stream: joint
Input: [N, 3, 50, 25, 2]
Pretext: 300 epochs
Linear evaluation: 200 epochs
Single GPU
```

目标：

1. 将 OSE 的 `Lproto`、`Lmix-proto`、`Lmix-ins` 迁移到 ReSA/AimCLR。
2. 在相同 fullmix 目标下比较 ST-GCN 与 CTR-GCN。

统一命名，禁止再使用含糊的“proto-only”：

```text
B0  = Lcluster + Lproto
M-P = B0 + Lmix-proto
M-F = B0 + Lmix-proto + Lmix-ins
```

## 2. 已完成结果

### 2.1 ST-GCN 正式结果

以下结果使用 NTU60 xsub joint、weak+weak、dropout0、batch128、LP200：

| 版本 | pretext checkpoint | OSE top-k | LP Top-1 |
|---|---:|---:|---:|
| B0 | 120 | 8 | 74.39 |
| B0 | 300 | 8 | 75.95 |
| M-P | 300 | 8 | 78.20 |
| M-F | 300 | 8 | 78.80 |

在 top-k=8、相同300-epoch协议下：

```text
M-P 相对 B0：+2.25
M-F 相对 B0：+2.85
Lmix-ins 在 M-P 上再贡献：+0.60
```

### 2.2 最新最佳结果

用户最新报告：

```text
ST-GCN，OSE top-k=4，LP Top-1 = 79.98
```

相对 top-k=8 的 M-F 78.80 提升 `+1.18`。这是当前最佳结果，但新会话应先向
用户确认它是否确实使用同一个 M-F checkpoint300、weak+weak、dropout0、LP200、
相同 LP seed/配置。确认前标为“用户报告的最佳单次结果”，不要擅自混入正式消融表。

历史500-epoch B0 schedule 的 74.77/75.94/75.86 不再展开；它与正式300-epoch
schedule 的同名 checkpoint 不可严格比较，因为 cosine LR 依赖总 epoch。

## 3. 已完成实现

关键文件：

```text
net/ose_resa.py
processor/pretrain_ose_resa.py
net/ctrgcn.py
tests/test_ose_resa_lmix.py
tests/test_ctrgcn.py
```

Lmix 已完整实现：

```python
mixed_view = beta * view_b + (1.0 - beta) * view_a[mix_index]
```

`mixed_view` 只走：

```text
encoder_q -> projector_q -> normalized mixed_z
```

它不走 predictor、Sinkhorn、queue 或 teacher。`beta` 是输入混合权重；target
只是 loss 中的概率监督，且 target 分支 detach。exemplar、queue、prototype 必须
统一在 projector `Z` 空间，不能放回 predictor `Q` 空间。

最近相关提交：

```text
93438d2 ctrgcn layer
9bcbf9d ctrgcn
676f680 Lmix
a98e33d OSEchange
9725427 dropout change and augmentation change
```

本地 Python 没有可用的 torch/yaml 动态环境；动态测试需在服务器执行：

```bash
python -m unittest tests.test_ctrgcn tests.test_ose_resa_lmix
```

## 4. CTR-GCN 检查结论

### 4.1 Backbone 本身

本地10层 CTR-GCN 的核心结构与官方实现基本一致：

```text
64,64,64,64,128,128,128,256,256,256
stride 在第5、8层为2
```

没有发现导致显存异常的明显重复计算或邻接矩阵错误。真正问题是动态图 activation
和 fullmix 调用图，而不是参数量。

当前8层版是官方前8层：

```text
64,64,64,64,128,128,128,256
```

当前3层版是人为定义的：

```text
64,128,256
stride = 1,2,2
```

它既不是官方 CTR-GCN，也不是 SCD-Net 的三层 CTR。SCD-Net 实际使用
`64 -> 256 -> 64`、全部 stride1，并返回时空特征图给 Transformer。因此绝对不要
再用当前3层版解释或复现 SCD-Net 的显存/性能。

HiCo 官方仓库根本没有 CTR-GCN；它使用 GRU/LSTM/Transformer 双分支序列编码器。
SCD-Net 和 HiCo 的正式命令也都是 batch64，不能与我们的 batch128 fullmix 直接比较。

### 4.2 为什么比 ST-GCN 慢且占显存

我们的 ST-GCN 与 CTR-GCN 并不同宽：

```text
ST-GCN:
16,16,16,16,32,32,32,64,64,256

CTR-GCN:
64,64,64,64,128,128,128,256,256,256
```

ST-GCN 使用固定 `[3,25,25]` adjacency；CTR 每层3个 subset 都动态构造：

```text
[N*M, Cout, 25, 25]
```

batch128、M=2、Cout=256、FP32 时，一个 relation tensor 就是156.25 MiB，
与 OOM 日志中申请158 MiB吻合。CTR 还有多分支 TCN 和 `torch.cat`，因此参数量
接近不代表 activation、FLOPs 或速度接近。完整模型参数量又被大型
projector/predictor 掩盖，不能据此判断 backbone 成本。

M-F 每步有6次 backbone forward：

```text
online view_a       batch128，有梯度
online view_b       batch128，有梯度
online exemplar     batch60，有梯度
teacher view_a/b    batch128×2，no_grad
online mixed_view   batch128，有梯度
```

统一 backward 前同时保留的有梯度样本规模为 `444`。这正是 CTR 比 ST-GCN
更容易 OOM、三层仍很慢的主要原因。

## 5. 当前卡点

24GB GPU、FP32、batch128、M-F：

```text
10层：view_b online forward 时 OOM
8层：view_a/view_b 完成，exemplar forward 时 OOM
3层：约20GB，但第一个完整 iteration 极慢，实际不可运行300 epoch
```

代码只在 forward、backward、optimizer.step 全部结束后打印 `Iter 0 Done`，
所以“停在 Iter 0”不等于死锁；没有分支计时前不要下死锁结论。

用户已经明确：

```text
不使用 AMP
不接受 hidden_channels=16 + 额外64->256投影的 Tiny CTR
不希望继续靠盲目减层解决
目标仍是 fullmix CTR 与 ST-GCN 的比较
```

当前没有成功的 CTR pretext checkpoint，也没有用户确认的下一训练配置。

## 6. 下一步计划

### 优先级1：确认并固化 top-k=4 结果

向用户确认79.98对应的：

```text
pretext版本（预计 M-F）
checkpoint epoch
LP epoch、seed和配置
weights路径与独立work_dir
```

确认后把它加入正式结果表；必要时再决定是否复跑验证。不要擅自启动复跑。

### 优先级2：CTR 先诊断，不启动长期 run

若用户继续 CTR，先确认旧的3层进程已经停止；不要擅自杀进程。然后做默认关闭、
只运行一个 iteration 的分支计时/显存诊断：

```text
view_a / view_b / exemplar / teacher / mixed
loss / backward / optimizer.step
```

每段用 `torch.cuda.synchronize()` 后记录 wall time、allocated、reserved、
max_memory_allocated。诊断同步不能代表正式训练速度。

### 优先级3：若用户授权改代码，首选分阶段反传

候选方案尚未实现：

1. teacher a/b 先 `no_grad` 计算并缓存小型 embedding。
2. online view_a 算完对应 cluster 项后立即 backward 并释放图。
3. online view_b + exemplar 计算 cluster/proto 后 backward。
4. 为 mixed 分支重算一次 batch60 exemplar，再算 Lmix 并 backward。
5. 所有梯度累积完后只执行一次 optimizer.step 和一次 queue 更新。

这样同时存活的有梯度 backbone 规模可从444降到约188，只多一次 batch60
exemplar 的前向/反向，比全模型 checkpoint 更有希望兼顾速度。实现时必须处理
第二次 exemplar 前向造成的 BN running-stat 重复更新，否则协议会变化。

可同时优化 `_class_prototypes` 中约 `[60,60,8192]`（约112.5 MiB）的临时 clone，
改用 top-2 求“排除自身后的最大值”。`zero_grad(set_to_none=True)`也可作为小优化。

若仍不够，严格保持10层、FP32、batch128、完整M-F且不降速，只能考虑更大显存、
正确的多卡/分支并行，或自定义融合 `conv4 + adjacency + einsum` kernel；不存在
无代价的单卡开关。

## 7. 绝对不要再踩的坑

1. 不要再说 Lmix 未实现，也不要混淆 B0、M-P、M-F。
2. 不要让 mixed branch 进入 predictor、Sinkhorn、queue 或 teacher。
3. 不要把 target 当输入；输入混合权重是 `beta`。
4. 不要把 exemplar 改回 Q 空间、放回无标签 sampler，或送入 queue。
5. 标签只能用于挑选固定 exemplar 和检索后的 purity 诊断，不能参与模型、queue、top-k。
6. 不要恢复 weak+standard(rotation)；正式协议是 weak+weak。
7. 不要只凭 purity/entropy 调参；它们不是 LP 的可靠单调代理。
8. 不要混淆不同总 schedule 的 checkpoint，不要复用 work_dir，也不要把
   `weights + start_epoch` 当完整 resume。
9. 不要把当前3层 CTR 称为 SCD-style；结构和输出接口都不同。
10. 不要默认建议 AMP、Tiny CTR、继续减层、盲目降 batch、CPU offload 或全量 checkpoint。
11. 不要直接使用 DataParallel；Sinkhorn/queue 没有正确的跨卡同步。
12. 不要把 allocator 参数当根治；现有日志显示是真实容量不足。
13. 不要提交新的 `__pycache__`/`.pyc`；提交93438d2已经误带过缓存。
14. 不要未经用户授权删除、覆盖或终止服务器任务。

## 8. 新会话接手动作

先读本文，然后只做只读检查：

```bash
git status --short
git log -5 --oneline
git diff --check
```

第一件事不是改 CTR，而是确认 top-k=4 的79.98完整实验身份。若用户转回 CTR，
再确认服务器任务状态并征得同意后做单 iteration 诊断或分阶段反传实现。
