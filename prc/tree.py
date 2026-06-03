import math
import pickle

import numpy as np


class PRCNode(object):
    def __init__(self, node_id, parent_id, depth, samples, child_pos=None):
        self.node_id = int(node_id)
        self.parent_id = parent_id
        self.depth = int(depth)
        self.samples = np.asarray(samples, dtype=np.int64)
        self.child_pos = child_pos
        self.children = []
        self.split_epoch = None
        self.split_stats = {}
        self.reassign_stats = {}

    @property
    def is_leaf(self):
        return len(self.children) == 0


def _l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _inertia(x):
    if len(x) == 0:
        return 0.0
    center = x.mean(axis=0, keepdims=True)
    return float(((x - center) ** 2).sum())


def _balanced_binary_assignment(dist, min_child_count=1):
    n = dist.shape[0]
    if n < 2:
        return None
    min_child_count = int(max(1, min(int(min_child_count), n // 2)))
    margin = dist[:, 0] - dist[:, 1]
    order = np.argsort(margin)
    count0 = int((margin <= 0).sum())
    count0 = min(max(count0, min_child_count), n - min_child_count)
    labels = np.ones(n, dtype=np.int64)
    labels[order[:count0]] = 0
    return labels


def _binary_kmeans(x, seed=0, niter=30, min_child_count=1):
    n = x.shape[0]
    if n < 2:
        return None, math.inf, None

    rng = np.random.RandomState(seed)
    first = int(rng.randint(n))
    dist = ((x - x[first]) ** 2).sum(axis=1)
    second = int(np.argmax(dist))
    if second == first:
        second = (first + 1) % n
    centers = np.stack([x[first], x[second]], axis=0).astype(np.float32)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(niter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = _balanced_binary_assignment(dist, min_child_count=min_child_count)
        if new_labels is None:
            return None, math.inf, None

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(2):
            members = x[labels == k]
            if len(members) > 0:
                centers[k] = members.mean(axis=0)

    final_dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = _balanced_binary_assignment(final_dist, min_child_count=min_child_count)
    if labels is None:
        return None, math.inf, None
    for k in range(2):
        members = x[labels == k]
        if len(members) == 0:
            return None, math.inf, None
        centers[k] = members.mean(axis=0)
    final_dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    inertia = float(final_dist[np.arange(n), labels].sum())
    return labels, inertia, centers


def _balanced_projection_split(x, centers=None, seed=0):
    n = x.shape[0]
    if n < 2:
        return None, math.inf, None

    if centers is not None:
        direction = centers[1] - centers[0]
    else:
        rng = np.random.RandomState(seed)
        direction = rng.normal(size=x.shape[1]).astype(np.float32)

    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        centered = x - x.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            direction = vt[0]
            norm = np.linalg.norm(direction)
        except np.linalg.LinAlgError:
            norm = 0.0
    if norm <= 1e-12:
        return None, math.inf, None

    score = np.dot(x, direction / norm)
    order = np.argsort(score)
    labels = np.zeros(n, dtype=np.int64)
    labels[order[n // 2:]] = 1

    split_centers = []
    for k in range(2):
        members = x[labels == k]
        if len(members) == 0:
            return None, math.inf, None
        split_centers.append(members.mean(axis=0))
    split_centers = np.stack(split_centers, axis=0).astype(np.float32)
    final_dist = ((x[:, None, :] - split_centers[None, :, :]) ** 2).sum(axis=2)
    inertia = float(final_dist[np.arange(n), labels].sum())
    return labels, inertia, split_centers


def _binary_agreement(a, b):
    same = float((a == b).mean())
    flipped = float((a == (1 - b)).mean())
    return max(same, flipped)


def _softmax(x, temperature=1.0):
    x = x / max(float(temperature), 1e-12)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _bic_score(sse, counts, dim, eps=1e-12):
    """BIC for a local spherical Gaussian mixture with shared variance."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    n = float(counts.sum())
    k = float(len(counts))
    if n <= k or dim <= 0:
        return -math.inf

    sigma2 = max(float(sse) / max(n * float(dim), 1.0), eps)
    log_prior = float((counts * np.log(np.maximum(counts / n, eps))).sum())
    log_likelihood = (
        log_prior -
        0.5 * n * float(dim) * (math.log(2.0 * math.pi * sigma2) + 1.0)
    )
    num_params = int(k) * int(dim) + 1 + (int(k) - 1)
    return log_likelihood - 0.5 * float(num_params) * math.log(max(n, 1.0))


class ProgressiveRecursiveTree(object):
    """Dynamic binary clustering tree for PRC pseudo labels."""

    def __init__(self, num_samples, force_root_split=True, kmeans_iters=30, seed=0,
                 routing_temperature=0.2, depth_penalty_weight=100.0, parent_size_penalty_weight=1.0,
                 tiny_child_penalty_weight=1.0, grow_confidence_threshold=0.9,
                 balanced_assignment_base_ratio=0.2, balanced_assignment_floor_ratio=0.02,
                 true_labels=None):
        self.num_samples = int(num_samples)
        self.force_root_split = bool(force_root_split)
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)
        self.routing_temperature = float(routing_temperature)
        self.depth_penalty_weight = float(depth_penalty_weight)
        self.parent_size_penalty_weight = float(parent_size_penalty_weight)
        self.tiny_child_penalty_weight = float(tiny_child_penalty_weight)
        self.grow_confidence_threshold = float(grow_confidence_threshold)
        self.balanced_assignment_base_ratio = float(balanced_assignment_base_ratio)
        self.balanced_assignment_floor_ratio = float(balanced_assignment_floor_ratio)
        self.true_labels = None if true_labels is None else np.asarray(true_labels)
        self.control_stats = {}

        self.nodes = {
            0: PRCNode(0, None, 0, np.arange(self.num_samples, dtype=np.int64))
        }
        self.next_node_id = 1
        self.sample_paths = [[] for _ in range(self.num_samples)]
        self.stage = 0

    def leaves(self):
        return [node for node in self.nodes.values() if node.is_leaf]

    def internal_node_ids(self):
        return sorted([node_id for node_id, node in self.nodes.items() if not node.is_leaf])

    def _min_child_ratio(self, depth):
        depth_ratio = self.balanced_assignment_base_ratio / float(int(depth) + 1)
        return max(self.balanced_assignment_floor_ratio, depth_ratio)

    def _min_child_count(self, node, n):
        ratio = self._min_child_ratio(node.depth)
        return int(math.ceil(float(n) * ratio))

    def update(self, features, epoch):
        features = _l2_normalize(np.asarray(features, dtype=np.float32))
        reassign_stats = self.soft_reassign(features)
        split_nodes = []
        root_bootstrap_pending = len(self.internal_node_ids()) == 0 and self.force_root_split
        growth_blocked = (
            not root_bootstrap_pending and
            self.grow_confidence_threshold > 0 and
            reassign_stats['mean_confidence'] < self.grow_confidence_threshold
        )
        if growth_blocked:
            self.control_stats = {
                'num_candidates': 0,
                'growth_blocked': True,
                'block_reason': 'route_conf {:.4f} < {:.4f}'.format(
                    reassign_stats['mean_confidence'], self.grow_confidence_threshold),
            }
            return split_nodes, reassign_stats

        candidates = sorted(self.leaves(), key=lambda n: (-n.depth, -len(n.samples), n.node_id))
        split_candidates = []

        for node in candidates:
            candidate = self._evaluate_split_candidate(node, features)
            if candidate is not None:
                split_candidates.append(candidate)

        self.control_stats = {
            'num_candidates': len(split_candidates),
        }

        for candidate in split_candidates:
            stats = self._try_split_node(candidate, epoch)
            if stats is None:
                continue
            split_nodes.append((candidate['node'].node_id, stats))

        if split_nodes:
            self.stage += 1
            self._rebuild_sample_paths()
        return split_nodes, reassign_stats

    def soft_reassign(self, features):
        """Top-down local binary K-means reassignment on the existing tree."""
        old_paths = self.sample_paths
        root = self.nodes[0]
        root.samples = np.arange(self.num_samples, dtype=np.int64)
        stats = {
            'num_internal': len(self.internal_node_ids()),
            'num_changed': 0,
            'mean_confidence': 0.0,
            'num_aligned_flips': 0,
        }
        confidences = []

        def clear_descendants(node):
            for child_id in node.children:
                child = self.nodes[child_id]
                child.samples = np.asarray([], dtype=np.int64)
                clear_descendants(child)

        def prune_descendants(node):
            for child_id in list(node.children):
                child = self.nodes[child_id]
                prune_descendants(child)
                if child_id in self.nodes:
                    del self.nodes[child_id]
            node.children = []

        def route(node):
            if node.is_leaf:
                return
            samples = node.samples
            if len(samples) == 0:
                clear_descendants(node)
                return
            children = [self.nodes[child_id] for child_id in node.children]
            x = features[samples]
            labels, _, centers = _binary_kmeans(
                x, seed=self.seed + self.stage * 3571 + node.node_id,
                niter=self.kmeans_iters, min_child_count=self._min_child_count(node, len(samples)))

            if labels is None or np.bincount(labels, minlength=len(children)).min() == 0:
                prune_descendants(node)
                return

            old_child_samples = [child.samples.copy() for child in children]
            new_child_samples = [samples[labels == child_pos] for child_pos in range(len(children))]
            same_overlap = (
                np.intersect1d(new_child_samples[0], old_child_samples[0], assume_unique=False).size +
                np.intersect1d(new_child_samples[1], old_child_samples[1], assume_unique=False).size
            )
            flipped_overlap = (
                np.intersect1d(new_child_samples[0], old_child_samples[1], assume_unique=False).size +
                np.intersect1d(new_child_samples[1], old_child_samples[0], assume_unique=False).size
            )
            if flipped_overlap > same_overlap:
                labels = 1 - labels
                centers = centers[[1, 0]]
                stats['num_aligned_flips'] += 1

            centers = _l2_normalize(centers.astype(np.float32))
            sims = np.dot(x, centers.T)
            probs = _softmax(sims, temperature=self.routing_temperature)
            route_conf = probs[np.arange(len(samples)), labels]

            for child_pos, child in enumerate(children):
                child.samples = samples[labels == child_pos]

            node.reassign_stats = {
                'n': int(len(samples)),
                'mean_confidence': float(route_conf.mean()) if len(route_conf) else 0.0,
            }
            confidences.extend(route_conf.tolist())
            for child in children:
                route(child)

        route(root)
        self._rebuild_sample_paths()

        for sample_idx, path in enumerate(self.sample_paths):
            if path != old_paths[sample_idx]:
                stats['num_changed'] += 1
        if confidences:
            stats['mean_confidence'] = float(np.mean(confidences))
        return stats

    def _evaluate_split_candidate(self, node, features):
        samples = node.samples
        n = len(samples)

        x = features[samples]
        labels_a, child_inertia, centers = _binary_kmeans(
            x, seed=self.seed + self.stage * 7919 + node.node_id,
            niter=self.kmeans_iters, min_child_count=self._min_child_count(node, n))
        if labels_a is None:
            return None

        is_root_bootstrap = node.node_id == 0 and self.force_root_split and len(node.children) == 0
        if is_root_bootstrap:
            labels_b, balanced_inertia, balanced_centers = _balanced_projection_split(
                x, centers=centers, seed=self.seed + self.stage * 1543 + node.node_id)
            if labels_b is not None:
                labels_a = labels_b
                child_inertia = balanced_inertia
                centers = balanced_centers

        counts = np.bincount(labels_a, minlength=2)
        balance = float(counts.min()) / float(n)
        parent_inertia = _inertia(x)
        gain = (parent_inertia - child_inertia) / max(parent_inertia, 1e-12)
        dim = x.shape[1]
        bic_parent = _bic_score(parent_inertia, [n], dim)
        bic_children = _bic_score(child_inertia, counts, dim)
        delta_bic = bic_children - bic_parent
        min_child = int(counts.min())
        parent_ratio = float(n) / max(float(self.num_samples), 1.0)
        tiny_child_ratio = float(min_child) / max(float(self.num_samples), 1.0)
        assignment_min_ratio = self._min_child_ratio(node.depth)
        assignment_min_count = self._min_child_count(node, n)
        return {
            'node': node,
            'n': n,
            'labels': labels_a,
            'centers': centers,
            'counts': counts,
            'balance': balance,
            'parent_ratio': parent_ratio,
            'tiny_child_ratio': tiny_child_ratio,
            'assignment_min_ratio': assignment_min_ratio,
            'assignment_min_count': assignment_min_count,
            'gain': gain,
            'parent_inertia': parent_inertia,
            'child_inertia': child_inertia,
            'bic_parent': bic_parent,
            'bic_children': bic_children,
            'delta_bic': delta_bic,
            'dim': dim,
        }

    def _split_penalties(self, node, n, counts, dim):
        depth_penalty = (
            self.depth_penalty_weight *
            math.log(max(float(n), 1.0)) *
            float(node.depth + 1) ** 2
        )

        parent_size_penalty = (
            self.parent_size_penalty_weight *
            float(dim) *
            float(self.num_samples) / max(float(n), 1.0)
        )

        min_child = float(np.min(counts))
        tiny_child_penalty = (
            self.tiny_child_penalty_weight *
            float(dim) *
            float(self.num_samples) / max(min_child, 1.0)
        )
        return depth_penalty, parent_size_penalty, tiny_child_penalty

    def _top_labels(self, samples, top_k=3):
        if self.true_labels is None or len(samples) == 0:
            return []
        labels = self.true_labels[np.asarray(samples, dtype=np.int64)]
        values, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)[:top_k]
        total = float(len(samples))
        top_labels = []
        for idx in order:
            value = values[idx]
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = str(value)
            top_labels.append('{}:{:.3f}'.format(value, float(counts[idx]) / max(total, 1.0)))
        return top_labels

    def leaf_label_summary(self, top_k=3, max_leaves=5):
        if self.true_labels is None:
            return None
        leaf_infos = []
        purities = []
        for node in self.leaves():
            n = len(node.samples)
            if n == 0:
                purity = 0.0
                top_labels = []
            else:
                top_labels = self._top_labels(node.samples, top_k=top_k)
                purity = 0.0
                if top_labels:
                    try:
                        purity = float(str(top_labels[0]).split(':')[-1])
                    except (TypeError, ValueError):
                        purity = 0.0
            purities.append(purity)
            leaf_infos.append({
                'node_id': node.node_id,
                'depth': node.depth,
                'n': int(n),
                'purity': purity,
                'top_labels': top_labels,
            })
        if not leaf_infos:
            return None
        leaf_infos.sort(key=lambda item: (-item['n'], item['node_id']))
        return {
            'mean_purity': float(np.mean(purities)),
            'min_purity': float(np.min(purities)),
            'max_purity': float(np.max(purities)),
            'top_leaves': leaf_infos[:max(0, int(max_leaves))],
        }

    def _try_split_node(self, candidate, epoch):
        node = candidate['node']
        samples = node.samples
        n = candidate['n']
        labels_a = candidate['labels']
        counts = candidate['counts']
        balance = candidate['balance']
        parent_ratio = candidate['parent_ratio']
        tiny_child_ratio = candidate['tiny_child_ratio']
        assignment_min_ratio = candidate['assignment_min_ratio']
        assignment_min_count = candidate['assignment_min_count']
        gain = candidate['gain']
        centers = candidate['centers']
        bic_parent = candidate['bic_parent']
        bic_children = candidate['bic_children']
        delta_bic = candidate['delta_bic']
        dim = candidate['dim']
        depth_penalty, parent_size_penalty, tiny_child_penalty = self._split_penalties(
            node, n, counts, dim)
        split_score = delta_bic - depth_penalty - parent_size_penalty - tiny_child_penalty

        is_root_bootstrap = node.node_id == 0 and self.force_root_split and len(node.children) == 0
        accept = is_root_bootstrap or split_score > 0
        if not accept:
            return None

        child_ids = []
        for child_pos in range(2):
            child_samples = samples[labels_a == child_pos]
            child_id = self.next_node_id
            self.next_node_id += 1
            self.nodes[child_id] = PRCNode(
                child_id, node.node_id, node.depth + 1, child_samples, child_pos=child_pos)
            child_ids.append(child_id)

        node.children = child_ids
        node.split_epoch = int(epoch)
        node.split_stats = {
            'n': int(n),
            'balance': balance,
            'gain': gain,
            'score': delta_bic,
            'split_score': split_score,
            'depth_penalty': depth_penalty,
            'parent_size_penalty': parent_size_penalty,
            'tiny_child_penalty': tiny_child_penalty,
            'parent_ratio': parent_ratio,
            'tiny_child_ratio': tiny_child_ratio,
            'assignment_min_ratio': assignment_min_ratio,
            'assignment_min_count': assignment_min_count,
            'parent_top_labels': self._top_labels(samples),
            'child_top_labels': [
                self._top_labels(samples[labels_a == 0]),
                self._top_labels(samples[labels_a == 1]),
            ],
            'bic_parent': bic_parent,
            'bic_children': bic_children,
            'delta_bic': delta_bic,
            'counts': counts.astype(np.int64).tolist(),
            'centers': _l2_normalize(centers.astype(np.float32)),
        }
        return node.split_stats

    def _rebuild_sample_paths(self):
        self.sample_paths = [[] for _ in range(self.num_samples)]
        for node in self.leaves():
            path = []
            current = node
            while current.parent_id is not None:
                path.append((current.parent_id, current.child_pos))
                current = self.nodes[current.parent_id]
            path.reverse()
            for sample_idx in node.samples:
                self.sample_paths[int(sample_idx)] = path

    def targets_for_indices(self, indices):
        indices = [int(i) for i in indices]
        targets = {}
        for row, sample_idx in enumerate(indices):
            for parent_id, child_pos in self.sample_paths[sample_idx]:
                if parent_id not in targets:
                    targets[parent_id] = -np.ones(len(indices), dtype=np.int64)
                targets[parent_id][row] = int(child_pos)
        return targets

    def summary(self):
        leaf_sizes = [len(node.samples) for node in self.leaves()]
        return {
            'num_nodes': len(self.nodes),
            'num_internal': len(self.internal_node_ids()),
            'num_leaves': len(leaf_sizes),
            'max_depth': max(node.depth for node in self.nodes.values()),
            'min_leaf_size': int(min(leaf_sizes)) if leaf_sizes else 0,
            'max_leaf_size': int(max(leaf_sizes)) if leaf_sizes else 0,
        }

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
