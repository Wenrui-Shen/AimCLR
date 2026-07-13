# Progressive Recursive Clustering for Skeleton SSL

## Motivation

DeepCluster-style methods convert unlabeled samples into flat pseudo classes and train the encoder with cross entropy. This is effective, but it couples self-supervision to a fixed global cluster number \(K\). In early training, the feature space is unstable, so a large fixed \(K\) can create noisy fine labels; in late training, a small fixed \(K\) can under-partition action semantics.

Progressive Recursive Clustering (PRC) keeps the pseudo-label CE objective, but replaces the flat label \(y_i \in \{1,\ldots,K\}\) with a path label on a dynamic clustering tree:

\[
\mathrm{path}(x_i):\quad r \rightarrow c_i^1 \rightarrow c_i^2 \rightarrow \cdots \rightarrow c_i^{d_i}.
\]

Each split is local and binary by default. A split is accepted only when a local two-cluster model improves BIC over the one-cluster parent model. The final number of leaves is therefore produced by the data and training dynamics rather than prescribed as a single \(K\).

## Model

Given a skeleton sequence \(x_i\), an ST-GCN encoder produces a normalized representation

\[
z_i = \frac{f_\theta(x_i)}{\|f_\theta(x_i)\|_2}.
\]

The clustering tree is \(T=(V,E)\). The root node contains every training sample. Every internal node \(u\) owns a binary classifier \(h_u\), which predicts the child branch under that parent:

\[
p_\theta(v \mid u, x_i)=\mathrm{softmax}(h_u(z_i))_v,\quad v\in\{0,1\}.
\]

For a sample with path \(r=u_i^0 \rightarrow u_i^1 \rightarrow \cdots \rightarrow u_i^{d_i}\), the target at level \(\ell\) is the child index

\[
y_i^\ell = \mathrm{child\_id}(u_i^{\ell-1}, u_i^\ell) \in \{0,1\}.
\]

The main objective is hierarchical pseudo-label cross entropy:

\[
\mathcal{L}_{\mathrm{PRC}}(x_i)
= \sum_{\ell=1}^{d_i} \alpha_{\ell}
\,\mathrm{CE}\left(h_{u_i^{\ell-1}}(z_i), y_i^\ell\right),
\]

where \(\alpha_\ell\) can be set to \(1\), or decayed for deeper, noisier labels. For a mini-batch \(B\),

\[
\mathcal{L}_{\mathrm{PRC}}(B)
= \frac{1}{|B|}
\sum_{i\in B}
\sum_{\ell=1}^{d_i}
\alpha_\ell
\,\mathrm{CE}\left(h_{u_i^{\ell-1}}(f_\theta(x_i)), y_i^\ell\right).
\]

No contrastive loss, reconstruction loss, masked prediction loss, or prototype contrast is required. Clustering pseudo labels are the supervision signal.

## Marginal Entropy Anti-Collapse

Following the latent distribution matching view of SSL, PRC can add entropy terms to prevent trivial assignment collapse. The entropy should not be applied to each individual prediction \(p_\theta(\cdot\mid u,x_i)\), because maximizing per-sample assignment entropy would make every sample uncertain. Instead, PRC regularizes marginal assignment distributions over a batch.

For an internal node \(u\), let

\[
B_u=\{i\in B \mid u \in \mathrm{path}(x_i)\}
\]

be the mini-batch samples whose current path passes through \(u\). The node-level marginal prediction is

\[
\bar p_u=\frac{1}{|B_u|}\sum_{i\in B_u}p_\theta(\cdot\mid u,x_i).
\]

The node-level anti-collapse term is a weak entropy floor:

\[
\mathcal{L}_{\mathrm{ent}}(B)
=\lambda_H\sum_{u\in \mathcal{I}(T)}
\left[\tau_H-\frac{H(\bar p_u)}{\log |\mathrm{ch}(u)|}\right]_+,
\]

where \(\mathcal{I}(T)\) is the set of internal nodes. The hinge form is important: once the marginal entropy is above the floor \(\tau_H\), there is no pressure to become perfectly uniform. Therefore, the term prevents all samples from taking the same branch while still allowing naturally long-tailed child sizes.

To target the stronger collapse mode where samples inside one leaf all map to nearly the same representation, PRC also adds a within-leaf inverse-variance penalty. Let \(q_{i\ell}\) be the soft probability that sample \(i\) reaches leaf \(\ell\), obtained by multiplying routing probabilities along the root-to-leaf path. The weighted leaf mean and variance are

\[
\mu_\ell=\frac{\sum_{i\in B}q_{i\ell}z_i}{\sum_{i\in B}q_{i\ell}},
\quad
\sigma_{\ell j}^2
=\frac{\sum_{i\in B}q_{i\ell}(z_{ij}-\mu_{\ell j})^2}{\sum_{i\in B}q_{i\ell}}.
\]

The penalty is

\[
\mathcal{L}_{\mathrm{var}}(B)
=\lambda_V
\sum_\ell
\frac{\sum_{i\in B}q_{i\ell}}{|B|}
\frac{1}{d}\sum_{j=1}^{d}
\frac{1}{\sigma_{\ell j}^2+\epsilon}.
\]

This has no hand-set variance floor: if a leaf collapses and its variance approaches zero, the penalty grows sharply; if the leaf already has spread-out features, the penalty becomes small.

The training loss becomes

\[
\mathcal{L}(B)
=\mathcal{L}_{\mathrm{PRC}}(B)
+\mathcal{L}_{\mathrm{ent}}(B)
+\mathcal{L}_{\mathrm{var}}(B).
\]

## Recursive Split Rule

At the beginning of each stage, PRC extracts features for the full training set using the current encoder:

\[
Z=\{z_i\}_{i=1}^{N}.
\]

For each current leaf node \(u\) with sample set \(S_u\), PRC considers a local binary K-means split:

\[
S_u \rightarrow S_{u,0}\cup S_{u,1},\quad S_{u,0}\cap S_{u,1}=\varnothing.
\]

Let the parent within-cluster sum of squares be

\[
J(u)=\sum_{i\in S_u}\|z_i-\mu_u\|_2^2,
\]

and the child objective be

\[
J(u\rightarrow 0,1)
= \sum_{b\in\{0,1\}}\sum_{i\in S_{u,b}}\|z_i-\mu_{u,b}\|_2^2.
\]

PRC treats the parent as one spherical Gaussian and the child proposal as a two-component spherical Gaussian mixture. For a local model with \(K\) clusters, sample counts \(n_k\), total sample count \(n\), feature dimension \(d\), and shared variance

\[
\sigma^2=\frac{\mathrm{SSE}}{nd},
\]

the BIC score is

\[
\mathrm{BIC}
=\log p(S_u\mid \theta)
-\frac{p}{2}\log n,
\]

where \(p=Kd+1+(K-1)\) counts cluster centers, one shared variance, and mixture weights. PRC accepts the split when

\[
\Delta \mathrm{BIC}
=\mathrm{BIC}(S_{u,0},S_{u,1})-\mathrm{BIC}(S_u)>0.
\]

The root can be force-split once to bootstrap training, matching the practical role of the first DeepCluster assignment.

## Algorithm

1. Initialize tree \(T\) with a root containing all samples.
2. At each stage, extract full-dataset features using the current encoder.
3. Softly route samples from the root to existing children. For each internal node, compute probabilities over its current children from feature-to-centroid similarity.
4. Commit the most probable child at each internal node to obtain a hard root-to-leaf path pseudo-label. Low-confidence samples can keep their previous sibling assignment to reduce label jitter.
5. Visit current leaves. For each leaf, test a binary K-means split with local BIC model selection.
6. Accepted leaves are split into two child nodes. Parent nodes remain in the tree and continue to provide a CE term.
7. Train the encoder and all active node heads with hard path CE until the next stage.

## Paper Wording

Progressive Recursive Clustering converts clustering-based self-supervised learning from fixed-\(K\) flat classification into dynamic path prediction. Unlike methods that treat clusters as auxiliary positives, triplet samplers, or masked prediction targets, PRC uses hierarchical pseudo labels as the sole pretext objective. At the beginning of each epoch, PRC performs soft top-down routing on the existing tree to correct pseudo labels while preserving the hierarchy. The most probable route is committed as a hard path label for CE training. A leaf node is expanded only when a local two-cluster model has higher BIC than the one-cluster parent model. Thus, coarse labels are learned first, fine labels emerge only when useful, and the final pseudo classes are determined by the recursive tree rather than a predefined flat \(K\).

## Implementation Notes in This Repository

- `net/prc.py` defines `PRC`, an ST-GCN encoder plus one binary head per internal tree node.
- `prc/tree.py` defines the dynamic tree, soft top-down reassignment, local binary K-means, split acceptance, and path-target lookup.
- `processor/pretrain_prc.py` performs full-dataset feature extraction, tree updates, dynamic head registration, hierarchical CE training, and optional marginal entropy anti-collapse.
- `feeder/ntu_feeder.py` now supports `return_index: True`, which is required because path labels are indexed by sample id.
- The default config uses `prc_mode: hard`. A leaf split is accepted by local X-means-style BIC model selection:

\[
\Delta\mathrm{BIC}(u)=
\mathrm{BIC}(S_{u,0}, S_{u,1})-\mathrm{BIC}(S_u).
\]

The parent model is one spherical Gaussian over the leaf's samples. The child model is a two-component spherical Gaussian mixture initialized by binary K-means. PRC accepts the split when `delta_BIC > 0`. If binary K-means or BIC is not numerically defined for a tiny leaf, that candidate naturally returns no split.
- Example command:

```bash
python main.py pretrain_prc --config config/ntu60/pretext/pretext_prc_xsub_joint.yaml
```

## Optional Differentiable Soft-Tree Variant

The code also keeps an experimental `prc_mode: soft`. In this mode PRC does not run K-means, does not accept or reject splits with hand-tuned thresholds, and does not need hard path pseudo labels. Instead, `net/prc.py` starts from a root router with two leaves and grows the tree during training. Every internal node has a learned routing head

\[
p_\theta(b\mid u,x)=\mathrm{softmax}(h_u(z)/\tau)_b,\quad b\in\{0,1\}.
\]

The probability of reaching a leaf is the product of the routing probabilities along its path:

\[
q_\theta(\ell\mid x)=\prod_{(u,b)\in \mathrm{path}(\ell)}p_\theta(b\mid u,x).
\]

Each leaf owns a learned normalized prototype \(c_\ell\). The soft-tree objective is

\[
\mathcal{L}_{soft}
=\lambda_c\sum_i\sum_\ell q_{i\ell}\|z_i-c_\ell\|_2^2
+\lambda_g \mathrm{KL}(\bar q\|U)
+\lambda_n\sum_u \mathrm{KL}(\bar p_u\|U_2)
+\lambda_s\sum_{\ell\ne m}[\delta-\|c_\ell-c_m\|_2^2]_+
+\lambda_h H(q_i).
\]

The terms correspond to compactness, global leaf balance, per-node branch balance, prototype separation, and assignment confidence. These are the differentiable counterparts of the previous hard split constraints such as gain, balance, separation, and confidence.

Growth is dynamic rather than fixed-depth. At the end of each epoch, the processor records each active leaf's soft compactness cost

\[
s_\ell=\bar q_\ell
\frac{\sum_i q_{i\ell}\|z_i-c_\ell\|_2^2}{\sum_i q_{i\ell}+\epsilon}.
\]

At a growth step, the highest-scoring leaves are expanded: the selected leaf becomes a new internal routing node, and its prototype is copied into two perturbed child prototypes. The new routing head and child prototypes are added to the optimizer immediately. This keeps tree growth data-driven without fixed depth or split-acceptance thresholds. The hard-tree implementation remains available with `prc_mode: hard` for comparison.
