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


def _binary_kmeans(x, seed=0, niter=30):
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
        new_labels = dist.argmin(axis=1).astype(np.int64)

        if np.all(new_labels == 0) or np.all(new_labels == 1):
            farthest = int(np.argmax(dist.min(axis=1)))
            new_labels[farthest] = 1 - new_labels[farthest]

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for k in range(2):
            members = x[labels == k]
            if len(members) > 0:
                centers[k] = members.mean(axis=0)

    final_dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = final_dist.argmin(axis=1).astype(np.int64)
    if np.all(labels == 0) or np.all(labels == 1):
        return None, math.inf, None
    inertia = float(final_dist[np.arange(n), labels].sum())
    return labels, inertia, centers


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
                 routing_temperature=0.2, reassign_confidence=0.0):
        self.num_samples = int(num_samples)
        self.force_root_split = bool(force_root_split)
        self.kmeans_iters = int(kmeans_iters)
        self.seed = int(seed)
        self.routing_temperature = float(routing_temperature)
        self.reassign_confidence = float(reassign_confidence)
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

    def update(self, features, epoch):
        features = _l2_normalize(np.asarray(features, dtype=np.float32))
        reassign_stats = self.soft_reassign(features)
        split_nodes = []
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
        """Softly route samples among existing siblings, then commit hard paths."""
        old_paths = self.sample_paths
        previous_child = {}
        for sample_idx, path in enumerate(old_paths):
            for parent_id, child_pos in path:
                previous_child[(sample_idx, parent_id)] = child_pos

        root = self.nodes[0]
        root.samples = np.arange(self.num_samples, dtype=np.int64)
        stats = {
            'num_internal': len(self.internal_node_ids()),
            'num_changed': 0,
            'mean_confidence': 0.0,
        }
        confidences = []

        def clear_descendants(node):
            for child_id in node.children:
                child = self.nodes[child_id]
                child.samples = np.asarray([], dtype=np.int64)
                clear_descendants(child)

        def route(node):
            if node.is_leaf:
                return
            samples = node.samples
            if len(samples) == 0:
                clear_descendants(node)
                return
            children = [self.nodes[child_id] for child_id in node.children]
            centers = []
            for child in children:
                if len(child.samples) == 0:
                    centers.append(features[samples].mean(axis=0))
                else:
                    centers.append(features[child.samples].mean(axis=0))
            centers = _l2_normalize(np.stack(centers, axis=0).astype(np.float32))
            sims = np.dot(features[samples], centers.T)
            probs = _softmax(sims, temperature=self.routing_temperature)
            hard_pos = probs.argmax(axis=1).astype(np.int64)
            max_prob = probs[np.arange(len(samples)), hard_pos]

            if self.reassign_confidence > 0:
                for row, sample_idx in enumerate(samples):
                    old_pos = previous_child.get((int(sample_idx), node.node_id))
                    if old_pos is not None and max_prob[row] < self.reassign_confidence:
                        hard_pos[row] = int(old_pos)

            node.reassign_stats = {
                'n': int(len(samples)),
                'mean_confidence': float(max_prob.mean()) if len(max_prob) else 0.0,
            }
            confidences.extend(max_prob.tolist())

            for child_pos, child in enumerate(children):
                child.samples = samples[hard_pos == child_pos]
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
            x, seed=self.seed + self.stage * 7919 + node.node_id, niter=self.kmeans_iters)
        if labels_a is None:
            return None

        counts = np.bincount(labels_a, minlength=2)
        balance = float(counts.min()) / float(n)
        parent_inertia = _inertia(x)
        gain = (parent_inertia - child_inertia) / max(parent_inertia, 1e-12)
        dim = x.shape[1]
        bic_parent = _bic_score(parent_inertia, [n], dim)
        bic_children = _bic_score(child_inertia, counts, dim)
        delta_bic = bic_children - bic_parent
        return {
            'node': node,
            'n': n,
            'labels': labels_a,
            'centers': centers,
            'counts': counts,
            'balance': balance,
            'gain': gain,
            'parent_inertia': parent_inertia,
            'child_inertia': child_inertia,
            'bic_parent': bic_parent,
            'bic_children': bic_children,
            'delta_bic': delta_bic,
            'x': x,
        }

    def _try_split_node(self, candidate, epoch):
        node = candidate['node']
        samples = node.samples
        n = candidate['n']
        labels_a = candidate['labels']
        centers = candidate['centers']
        counts = candidate['counts']
        balance = candidate['balance']
        gain = candidate['gain']
        child_inertia = candidate['child_inertia']
        bic_parent = candidate['bic_parent']
        bic_children = candidate['bic_children']
        delta_bic = candidate['delta_bic']
        x = candidate['x']

        is_root_bootstrap = node.node_id == 0 and self.force_root_split and len(node.children) == 0
        accept = is_root_bootstrap or delta_bic > 0
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
            'bic_parent': bic_parent,
            'bic_children': bic_children,
            'delta_bic': delta_bic,
            'counts': counts.astype(np.int64).tolist(),
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
