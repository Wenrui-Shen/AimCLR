# 渐进式递归聚类（PRC）论文写法草稿

## 方法概述

本文提出渐进式递归聚类（Progressive Recursive Clustering, PRC），用于骨架动作识别的自监督表征学习。与 DeepCluster 类方法将所有样本一次性划分为固定数量 \(K\) 个扁平伪类别不同，PRC 将伪标签组织为一棵动态聚类树。每个样本的监督信号不再是单一 cluster id，而是一条从根节点到当前叶节点的路径标签。模型需要逐层预测样本在当前父簇下属于哪个子簇，从而把聚类式自监督学习转化为层级路径预测任务。

PRC 的核心思想是先学习粗粒度可分结构，再在表征逐渐稳定后递归产生细粒度伪标签。每个 epoch 开始时，PRC 不会推翻已有层级做全局重新聚类，而是在已有树上执行自顶向下的软路由纠错：对每个已有父节点，根据样本与各子节点中心的相似度得到 soft routing 概率，再选择概率最大的子节点形成 hard 路径伪标签。每个叶节点是否继续分裂由局部 BIC 模型选择决定：若二分 K-means 给出的两簇模型比单簇父模型具有更高 BIC，则接受该 split。因此最终簇数由数据和表征质量自适应产生，而不是由人工预设的全局 \(K\) 决定。

## 层级路径伪标签

给定无标签骨架序列 \(x_i\)，编码器 \(f_\theta\) 输出归一化表征

\[
z_i=\frac{f_\theta(x_i)}{\|f_\theta(x_i)\|_2}.
\]

动态聚类树记为 \(T=(V,E)\)，根节点 \(r\) 包含全部训练样本。若节点 \(u\) 被分裂，则为其分配一个二分类预测头 \(h_u\)，用于预测样本在父节点 \(u\) 下进入哪个子节点：

\[
p_\theta(b\mid u,x_i)=\mathrm{softmax}(h_u(z_i))_b,\quad b\in\{0,1\}.
\]

设样本 \(x_i\) 的当前路径为

\[
r=u_i^0\rightarrow u_i^1\rightarrow \cdots \rightarrow u_i^{d_i},
\]

其中 \(d_i\) 是样本当前路径深度。第 \(\ell\) 层伪标签定义为

\[
y_i^\ell=\mathrm{child}(u_i^{\ell-1},u_i^\ell)\in\{0,1\}.
\]

PRC 的主损失为路径伪标签交叉熵：

\[
\mathcal{L}_i
=\sum_{\ell=1}^{d_i}\alpha_\ell
\mathrm{CE}\left(h_{u_i^{\ell-1}}(z_i),y_i^\ell\right),
\]

其中 \(\alpha_\ell\) 是层级权重。默认可令 \(\alpha_\ell=1\)，也可以使用 \(\alpha_\ell=\gamma^{\ell-1}\) 降低深层伪标签的权重。batch 损失为

\[
\mathcal{L}_{\mathrm{PRC}}
=\frac{1}{|B|}\sum_{i\in B}\mathcal{L}_i.
\]

该目标只依赖聚类伪标签本身，不引入 contrastive loss、masked reconstruction loss 或 prototype contrastive loss。

## 边缘熵防塌缩

借鉴 latent distribution matching 的观点，自监督学习可以理解为让样本表征分布匹配某种潜在目标分布，同时避免表征或 assignment 发生塌缩。在 PRC 中，熵正则可以用于防止所有样本被预测到同一个子簇，但熵的作用对象必须谨慎选择。

我们不最大化单个样本的 assignment entropy：

\[
H(p_\theta(\cdot\mid u,x_i)).
\]

因为这会鼓励每个样本在子簇之间保持不确定，与伪标签交叉熵希望样本进入明确路径的目标相冲突。相反，我们最大化父节点下 batch 或 dataset 层面的边缘分配熵。对内部节点 \(u\)，记当前 batch 中路径经过该节点的样本集合为

\[
B_u=\{i\in B\mid u\in \mathrm{path}(x_i)\}.
\]

该节点的边缘预测分布为

\[
\bar p_u=\frac{1}{|B_u|}\sum_{i\in B_u}p_\theta(\cdot\mid u,x_i).
\]

防塌缩项定义为低熵 hinge 惩罚：

\[
\mathcal{L}_{\mathrm{ent}}
=\lambda_H\sum_{u\in \mathcal{I}(T)}
\left[\tau_H-\frac{H(\bar p_u)}{\log |\mathrm{ch}(u)|}\right]_+.
\]

为了进一步避免一个叶节点内部的样本表征塌缩到同一点，PRC 对软叶分配下的簇内方差加入反比惩罚。记 \(q_{i\ell}\) 为样本 \(i\) 到达叶节点 \(\ell\) 的软路径概率，则

\[
\mu_\ell=\frac{\sum_i q_{i\ell}z_i}{\sum_i q_{i\ell}},
\quad
\sigma_{\ell j}^2=
\frac{\sum_i q_{i\ell}(z_{ij}-\mu_{\ell j})^2}{\sum_i q_{i\ell}}.
\]

簇内方差项定义为

\[
\mathcal{L}_{\mathrm{var}}
=\lambda_V\sum_\ell
\frac{\sum_i q_{i\ell}}{|B|}
\frac{1}{d}\sum_{j=1}^{d}
\frac{1}{\sigma_{\ell j}^2+\epsilon}.
\]

该项不使用方差阈值；当某个叶节点内部方差接近 0 时，惩罚会快速增大。

最终训练目标为

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{PRC}}+\mathcal{L}_{\mathrm{ent}}+\mathcal{L}_{\mathrm{var}}.
\]

这里使用 hinge 而不是直接最大化到 \(\log |\mathrm{ch}(u)|\)，原因是骨架动作数据通常存在长尾分布，真实动作类别或语义簇的样本数可能天然不均衡。PRC 只需要避免所有样本进入同一分支的极端退化，而不应该强迫每次递归分裂都接近 50/50。因此，\(\tau_H\) 应设置为低于最大熵的阈值，使边缘分配达到最低多样性后不再继续施加均衡压力。

## 递归分裂准则

每个 stage 开始时，使用当前编码器提取全训练集特征。对当前叶节点 \(u\) 的样本集合 \(S_u\)，在节点内部执行局部二分 K-means，得到两个候选子簇 \(S_{u,0}\) 和 \(S_{u,1}\)。父簇 SSE 为

\[
J(u)=\sum_{i\in S_u}\|z_i-\mu_u\|_2^2,
\]

候选子簇 SSE 为

\[
J(u\rightarrow 0,1)
=\sum_{b\in\{0,1\}}\sum_{i\in S_{u,b}}\|z_i-\mu_{u,b}\|_2^2.
\]

PRC 将父节点看作一个球形高斯模型，将候选 split 看作共享方差的二分量球形高斯混合模型。对局部模型 \(M_K\)，设簇数为 \(K\)，样本数为 \(n\)，特征维度为 \(d\)，各簇样本数为 \(n_k\)，共享方差为

\[
\sigma^2=\frac{\mathrm{SSE}}{nd}.
\]

BIC 写为

\[
\mathrm{BIC}(M_K)
=\log p(S_u\mid\theta_K)-\frac{p_K}{2}\log n,
\]

其中 \(p_K=Kd+1+(K-1)\)，分别对应 \(K\) 个中心、一个共享方差和 \(K-1\) 个混合权重。节点 \(u\) 被接受分裂当且仅当

\[
\Delta\mathrm{BIC}
=\mathrm{BIC}(M_2)-\mathrm{BIC}(M_1)>0.
\]

实践中可允许根节点在第一次 stage 被强制分裂一次，用于启动路径伪标签训练；后续节点严格遵循上述准则。

## 软路由纠错与结构复杂度

每个 epoch 的树更新首先执行软路由纠错。假设父节点 \(u\) 已经有若干子节点，PRC 先计算样本到每个子节点中心的相似度，并通过 softmax 得到样本进入各子节点的概率。这个概率只用于估计不确定性和纠错，最终路径标签仍然取最大概率的子节点，因此训练阶段保持 hard CE。若某个样本的最大路由概率过低，可以保留它上一轮在该父节点下的分配，避免伪标签在相邻 epoch 之间频繁抖动。

完成已有树上的路径纠正后，PRC 只检查当前叶节点是否值得新增子节点。每个叶节点先用二分 K-means 产生一个候选二叉划分，然后用局部 BIC 比较“一簇父模型”和“两簇子模型”。只有当两簇模型的 BIC 更高，即 \(\Delta\mathrm{BIC}>0\) 时，才接受该 split。这样不需要固定最大深度、最小样本数、分离度或稳定性阈值；模型复杂度由 BIC 的参数惩罚项自动控制，已经足够紧凑或样本不足的节点通常不会继续分裂。

## 可写入论文的贡献点

1. 提出一种以路径伪标签为主任务的骨架自监督学习框架，将扁平聚类分类扩展为层级路径预测。
2. 提出基于局部 BIC 的递归分裂机制，用模型选择准则决定是否扩展节点，从而避免固定 \(K\) 和手工 split 阈值。
3. 通过动态树结构自适应控制伪标签粒度，使最终簇数由训练过程自动产生，而非依赖人工预设的固定 \(K\)。
4. 引入父节点边缘分配熵作为防塌缩机制，只避免极端单簇退化，不强制簇大小完全均衡，因此适合长尾动作数据。
5. 该方法不依赖对比学习、重建或 masked prediction，可作为纯聚类伪标签驱动的 SSL pretext task。

## 论文段落示例

Existing clustering-based self-supervised methods usually generate flat pseudo labels with a pre-defined number of clusters. However, the optimal granularity of skeleton action semantics is difficult to determine before training, and fine-grained pseudo labels are particularly noisy when the representation is still immature. To address these issues, we propose Progressive Recursive Clustering (PRC), which organizes pseudo labels as paths on a dynamically growing clustering tree. Each internal node defines a local classification problem, and the model predicts which child branch a sample should follow under its current parent node. In this way, PRC preserves the simplicity of pseudo-label cross entropy while avoiding a fixed global cluster number.

During training, PRC periodically extracts features for all training samples and first performs soft top-down routing on the existing hierarchy. For each internal node, samples are softly assigned to its children according to feature-to-centroid similarity, and the most probable route is committed as a hard path pseudo-label. Low-confidence assignments can keep their previous sibling labels to reduce pseudo-label jitter. After this tree-constrained reassignment, PRC examines each current leaf node. A leaf is split only when its local binary partition satisfies minimum size, balance, separation, confidence, objective gain, and stability constraints after paying a structural complexity cost. The parent node remains active after splitting, and each sample is supervised by all decisions along its selected root-to-leaf path.

To avoid assignment collapse, we regularize the marginal branch distribution under each internal node rather than the entropy of each individual sample. Specifically, PRC computes the average predicted assignment over samples whose paths pass through the same parent node and applies a hinge penalty only when its normalized entropy is below a small threshold. This design discourages degenerate one-branch solutions while preserving naturally imbalanced and long-tailed action distributions.
