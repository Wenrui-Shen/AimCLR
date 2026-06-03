import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchlight
from torchlight import str2bool

from .pretrain import weights_init
from .processor import Processor, init_seed
from prc import ProgressiveRecursiveTree


class PRC_Processor(Processor):
    """
        Processor for Progressive Recursive Clustering pre-training.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        if self.arg.prc_mode == 'soft':
            self.model.init_soft_tree()
        self.model.apply(weights_init)

    def load_data(self):
        super(PRC_Processor, self).load_data()
        if self.arg.prc_mode == 'soft':
            self.prc_tree = None
            self.soft_leaf_scores = None
            self.soft_leaf_score_ids = None
            return
        num_samples = len(self.data_loader['train'].dataset)
        self.prc_tree = ProgressiveRecursiveTree(
            num_samples=num_samples,
            force_root_split=self.arg.prc_force_root_split,
            kmeans_iters=self.arg.prc_kmeans_iters,
            seed=self.arg.prc_seed,
            routing_temperature=self.arg.prc_routing_temperature,
            depth_penalty_weight=self.arg.prc_depth_penalty_weight,
            parent_size_penalty_weight=self.arg.prc_parent_size_penalty_weight,
            tiny_child_penalty_weight=self.arg.prc_tiny_child_penalty_weight,
            grow_confidence_threshold=self.arg.prc_grow_confidence_threshold,
            balanced_assignment_base_ratio=self.arg.prc_balanced_assignment_base_ratio,
            balanced_assignment_floor_ratio=self.arg.prc_balanced_assignment_floor_ratio,
            true_labels=getattr(self.data_loader['train'].dataset, 'label', None))

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1 ** np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def _model_core(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _prepare_stream(self, data):
        if self.arg.stream == 'joint':
            return data
        if self.arg.stream == 'motion':
            motion = torch.zeros_like(data)
            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
            return motion
        if self.arg.stream == 'bone':
            bone_pairs = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                          (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                          (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
            bone = torch.zeros_like(data)
            for v1, v2 in bone_pairs:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
            return bone
        raise ValueError('Unknown stream: {}'.format(self.arg.stream))

    def _parse_batch(self, batch):
        if len(batch) == 3:
            data_pack, label, index = batch
        elif len(batch) == 2:
            data_pack, label = batch
            index = None
        else:
            raise ValueError('Unsupported PRC batch format')
        return data_pack, label, index

    def _select_view(self, data_pack, view_index):
        if not isinstance(data_pack, (list, tuple)):
            return data_pack
        if len(data_pack) == 0:
            raise ValueError('Empty PRC data pack')
        view_index = int(view_index)
        if view_index < 0:
            view_index = len(data_pack) + view_index
        view_index = min(max(view_index, 0), len(data_pack) - 1)
        return data_pack[view_index]

    def _prepare_data_view(self, data_pack, view_index):
        data = self._select_view(data_pack, view_index)
        data = data.float().to(self.dev, non_blocking=True)
        return self._prepare_stream(data)

    def _extract_features(self):
        dataset = self.data_loader['train'].dataset
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
            drop_last=False,
            worker_init_fn=init_seed)

        was_training = self.model.training
        self.model.eval()
        features = np.zeros((len(dataset), self.arg.model_args['feature_dim']), dtype=np.float32)
        with torch.no_grad():
            for batch in loader:
                data_pack, _, index = self._parse_batch(batch)
                data = self._prepare_data_view(data_pack, self.arg.prc_target_view)
                z = self._model_core().forward_features(data)
                features[index.numpy()] = z.detach().cpu().numpy()

        if was_training:
            self.model.train()
        return features

    def _sync_heads_with_tree(self, split_nodes):
        core = self._model_core()
        for parent_id, _ in split_nodes:
            key = str(int(parent_id))
            if key in core.heads:
                continue
            head = core.add_split_head(parent_id)
            head.apply(weights_init)
            head.to(self.dev)
            self.optimizer.add_param_group({'params': head.parameters()})

    def _update_tree(self, epoch):
        if (epoch - 1) % self.arg.prc_reassign != 0:
            return
        features = self._extract_features()
        split_nodes, reassign_stats = self.prc_tree.update(features, epoch)
        self._sync_heads_with_tree(split_nodes)
        summary = self.prc_tree.summary()
        self.io.print_log(
            'PRC stage {} | splits {} | reassign_changed {} | route_conf {:.4f} | align_flip {} | nodes {} | leaves {} | max_depth {} | leaf_size {}-{}'.format(
                self.prc_tree.stage, len(split_nodes),
                reassign_stats['num_changed'], reassign_stats['mean_confidence'],
                reassign_stats.get('num_aligned_flips', 0),
                summary['num_nodes'], summary['num_leaves'],
                summary['max_depth'], summary['min_leaf_size'], summary['max_leaf_size']))
        control = self.prc_tree.control_stats
        if control:
            self.io.print_log(
                '  candidates | {}'.format(control['num_candidates']))
            if control.get('growth_blocked'):
                self.io.print_log(
                    '  grow blocked | {}'.format(control['block_reason']))
        for parent_id, stats in split_nodes:
            self.io.print_log(
                '  split node {} | n {} | counts {} | gain {:.4f} | score {:.2f} | delta_bic {:.2f} | depth_pen {:.2f} | parent_pen {:.2f} | child_pen {:.2f} | parent_ratio {:.4f} | child_ratio {:.4f} | assign_min {}@{:.3f} | parent_bic {:.2f} | child_bic {:.2f}'.format(
                    parent_id, stats['n'], stats['counts'], stats['gain'],
                    stats['split_score'], stats['delta_bic'], stats['depth_penalty'],
                    stats['parent_size_penalty'], stats['tiny_child_penalty'],
                    stats['parent_ratio'], stats['tiny_child_ratio'],
                    stats['assignment_min_count'], stats['assignment_min_ratio'],
                    stats['bic_parent'], stats['bic_children']))
            if stats.get('parent_top_labels') is not None:
                self.io.print_log(
                    '    top_labels parent {} | child0 {} | child1 {}'.format(
                        stats['parent_top_labels'],
                        stats['child_top_labels'][0],
                        stats['child_top_labels'][1]))
        if self.arg.prc_save_tree:
            self.prc_tree.save(os.path.join(self.arg.work_dir, 'prc_tree_epoch{}.pkl'.format(epoch)))

    def _path_loss(self, logits, index, ce_weight=None, entropy_weight=None):
        if index is None:
            raise ValueError('Hard PRC requires train_feeder_args.return_index: True')
        ce_weight = self.arg.prc_ce_weight if ce_weight is None else float(ce_weight)
        entropy_weight = self.arg.prc_entropy_weight if entropy_weight is None else float(entropy_weight)
        targets = self.prc_tree.targets_for_indices(index.detach().cpu().numpy())
        losses = []
        weights = []
        entropy_penalties = []
        for parent_id, target_np in targets.items():
            if parent_id not in logits:
                continue
            target = torch.from_numpy(target_np).long().to(self.dev)
            mask = target >= 0
            if mask.sum().item() == 0:
                continue
            loss = F.cross_entropy(logits[parent_id][mask], target[mask])
            weight = 1.0
            losses.append(loss * weight)
            weights.append(weight)

            if entropy_weight > 0 and mask.sum().item() >= self.arg.prc_entropy_min_samples:
                probs = torch.softmax(logits[parent_id][mask], dim=1)
                marginal = probs.mean(dim=0)
                entropy = -(marginal * torch.log(marginal.clamp_min(1e-12))).sum()
                entropy = entropy / np.log(float(probs.size(1)))
                entropy_penalties.append(F.relu(self.arg.prc_entropy_floor - entropy) * weight)

        if not losses:
            return None, None, None

        ce_loss = sum(losses) / max(sum(weights), 1e-12)
        if entropy_penalties:
            entropy_loss = sum(entropy_penalties) / max(sum(weights), 1e-12)
        else:
            entropy_loss = ce_loss.new_tensor(0.0)
        loss = ce_weight * ce_loss + entropy_weight * entropy_loss
        return loss, ce_loss, entropy_loss

    def _leaf_probabilities(self, logits, reference):
        leaves = self.prc_tree.leaves()
        if len(leaves) <= 1:
            return None
        leaf_probs = []
        for leaf in leaves:
            path = []
            current = leaf
            while current.parent_id is not None:
                path.append((current.parent_id, current.child_pos))
                current = self.prc_tree.nodes[current.parent_id]
            path.reverse()
            if not path:
                continue
            prob = reference.new_ones(logits[path[0][0]].size(0))
            for parent_id, child_pos in path:
                if parent_id not in logits:
                    prob = None
                    break
                route = torch.softmax(logits[parent_id], dim=1)
                prob = prob * route[:, child_pos]
            if prob is not None:
                leaf_probs.append(prob)

        if len(leaf_probs) <= 1:
            return None
        leaf_probs = torch.stack(leaf_probs, dim=1)
        return leaf_probs / leaf_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _leaf_variance_loss(self, logits, features, reference):
        if self.arg.prc_leaf_variance_weight <= 0:
            return reference.new_tensor(0.0)
        leaf_probs = self._leaf_probabilities(logits, reference)
        if leaf_probs is None:
            return reference.new_tensor(0.0)

        mass = leaf_probs.sum(dim=0).clamp_min(1e-12)
        mean = torch.matmul(leaf_probs.t(), features) / mass.unsqueeze(1)
        second = torch.matmul(leaf_probs.t(), features * features) / mass.unsqueeze(1)
        var = (second - mean * mean).clamp_min(0.0)
        leaf_weight = mass / mass.sum().clamp_min(1e-12)
        per_leaf = (1.0 / (var + self.arg.prc_leaf_variance_eps)).mean(dim=1)
        return (leaf_weight * per_leaf).sum()

    def _maybe_grow_soft_tree(self, epoch):
        if epoch < self.arg.prc_soft_grow_start_epoch:
            return
        grow_interval = max(1, int(self.arg.prc_soft_grow_interval))
        if (epoch - self.arg.prc_soft_grow_start_epoch) % grow_interval != 0:
            return
        if self.soft_leaf_scores is None or self.soft_leaf_score_ids is None:
            return

        core = self._model_core()
        current_leaves = set(core.soft_leaf_ids)
        candidates = [
            (float(score), int(leaf_id))
            for score, leaf_id in zip(self.soft_leaf_scores, self.soft_leaf_score_ids)
            if int(leaf_id) in current_leaves
        ]
        if not candidates:
            return

        candidates.sort(reverse=True)
        grow_count = max(1, int(self.arg.prc_soft_grow_leaves))
        if self.arg.prc_soft_max_leaves > 0:
            remaining = self.arg.prc_soft_max_leaves - len(core.soft_leaf_ids)
            grow_count = min(grow_count, max(0, remaining))
        if grow_count <= 0:
            return

        leaf_ids = [leaf_id for _, leaf_id in candidates[:grow_count]]
        new_modules, new_params = core.grow_soft_tree(
            leaf_ids, noise_scale=self.arg.prc_soft_split_noise)
        for module in new_modules:
            module.apply(weights_init)
            module.to(self.dev)
            new_params.extend(list(module.parameters()))
        if new_params:
            self.optimizer.add_param_group({'params': new_params})
            self.io.print_log(
                'Soft PRC grow epoch {} | split leaves {} | leaves {} | internal {}'.format(
                    epoch, leaf_ids, len(core.soft_leaf_ids), len(core.soft_internal_ids)))
        self.soft_leaf_scores = None
        self.soft_leaf_score_ids = None

    def _soft_tree_loss(self, out, epoch):
        z = out['features']
        leaf_probs = out['leaf_probs']
        route_probs = out['route_probs']
        reach_probs = out['reach_probs']
        prototypes = out['prototypes']
        num_leaves = leaf_probs.size(1)

        dist = 2.0 - 2.0 * torch.matmul(z, prototypes.t())
        compact_loss = (leaf_probs * dist).sum(dim=1).mean()

        leaf_marginal = leaf_probs.mean(dim=0)
        leaf_mass = leaf_probs.detach().sum(dim=0).clamp_min(1e-12)
        leaf_compactness = (leaf_probs.detach() * dist.detach()).sum(dim=0) / leaf_mass
        leaf_scores = leaf_marginal.detach() * leaf_compactness
        uniform_leaf = leaf_marginal.new_full((num_leaves,), 1.0 / float(num_leaves))
        balance_loss = (leaf_marginal * torch.log((leaf_marginal / uniform_leaf).clamp_min(1e-12))).sum()

        node_balance_losses = []
        for node_id, probs in route_probs.items():
            reach = reach_probs.get(node_id, z.new_ones(z.size(0)))
            marginal = (probs * reach.unsqueeze(1)).sum(dim=0) / reach.sum().clamp_min(1e-12)
            uniform_node = marginal.new_full((2,), 0.5)
            node_balance_losses.append(
                (marginal * torch.log((marginal / uniform_node).clamp_min(1e-12))).sum())
        if node_balance_losses:
            node_balance_loss = sum(node_balance_losses) / float(len(node_balance_losses))
        else:
            node_balance_loss = compact_loss.new_tensor(0.0)

        proto_dist = 2.0 - 2.0 * torch.matmul(prototypes, prototypes.t())
        proto_mask = ~torch.eye(num_leaves, dtype=torch.bool, device=proto_dist.device)
        separation_loss = F.relu(self.arg.prc_soft_separation_margin - proto_dist[proto_mask]).mean()

        leaf_entropy = -(leaf_probs * torch.log(leaf_probs.clamp_min(1e-12))).sum(dim=1)
        leaf_entropy = leaf_entropy / np.log(float(num_leaves))
        if epoch >= self.arg.prc_soft_confidence_start_epoch:
            confidence_loss = leaf_entropy.mean()
        else:
            confidence_loss = compact_loss.new_tensor(0.0)

        loss = (
            self.arg.prc_soft_compactness_weight * compact_loss +
            self.arg.prc_soft_balance_weight * balance_loss +
            self.arg.prc_soft_node_balance_weight * node_balance_loss +
            self.arg.prc_soft_separation_weight * separation_loss +
            self.arg.prc_soft_confidence_weight * confidence_loss
        )
        stats = {
            'compact': compact_loss,
            'balance': balance_loss,
            'node_balance': node_balance_loss,
            'separation': separation_loss,
            'confidence': confidence_loss,
            'mean_leaf_entropy': leaf_entropy.mean(),
            'used_leaves': (leaf_marginal > (1.0 / float(num_leaves * 10))).float().sum(),
            'leaf_scores': leaf_scores,
            'leaf_ids': out['leaf_ids'],
        }
        return loss, stats

    def train(self, epoch):
        self.adjust_lr()
        if self.arg.prc_mode == 'hard':
            self._update_tree(epoch)
        else:
            self._maybe_grow_soft_tree(epoch)
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []
        soft_leaf_score_sum = None
        soft_leaf_score_count = 0
        soft_leaf_score_ids = None

        for batch in loader:
            self.global_step += 1
            data_pack, _, index = self._parse_batch(batch)
            data = self._prepare_data_view(data_pack, self.arg.prc_target_view)

            if self.arg.prc_mode == 'soft':
                out = self._model_core().forward_soft(data, temperature=self.arg.prc_soft_temperature)
                loss, stats = self._soft_tree_loss(out, epoch)
                ce_loss = stats['compact']
                entropy_loss = stats['mean_leaf_entropy']
                consistency_loss = ce_loss.new_tensor(0.0)
                leaf_variance_loss = ce_loss.new_tensor(0.0)
                prc_nodes = len(self._model_core().soft_internal_ids)
                batch_size = data.size(0)
                leaf_scores = stats['leaf_scores'].detach().cpu().numpy()
                if soft_leaf_score_sum is None:
                    soft_leaf_score_sum = leaf_scores * batch_size
                    soft_leaf_score_ids = list(stats['leaf_ids'])
                else:
                    soft_leaf_score_sum += leaf_scores * batch_size
                soft_leaf_score_count += batch_size
            else:
                if index is None:
                    raise ValueError('Hard PRC requires train_feeder_args.return_index: True')
                index = index.long()
                node_ids = self.prc_tree.internal_node_ids()
                logits, features = self._model_core()(data, node_ids=node_ids)
                loss, ce_loss, entropy_loss = self._path_loss(logits, index)
                if loss is None:
                    continue
                consistency_loss = loss.new_tensor(0.0)
                if (
                    self.arg.prc_consistency_weight > 0 and
                    isinstance(data_pack, (list, tuple)) and
                    len(data_pack) > 1
                ):
                    cons_data = self._prepare_data_view(data_pack, self.arg.prc_consistency_view)
                    cons_logits, _ = self._model_core()(cons_data, node_ids=node_ids)
                    cons_loss, cons_ce, _ = self._path_loss(
                        cons_logits, index,
                        ce_weight=self.arg.prc_consistency_weight,
                        entropy_weight=0.0)
                    if cons_loss is not None:
                        loss = loss + cons_loss
                        consistency_loss = cons_ce
                leaf_variance_loss = self._leaf_variance_loss(logits, features, loss)
                loss = loss + self.arg.prc_leaf_variance_weight * leaf_variance_loss
                prc_nodes = len(node_ids)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['ce'] = ce_loss.data.item()
            self.iter_info['cons'] = consistency_loss.data.item()
            self.iter_info['ent'] = entropy_loss.data.item()
            self.iter_info['var'] = leaf_variance_loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['prc_nodes'] = prc_nodes
            if self.arg.prc_mode == 'soft':
                self.iter_info['bal'] = stats['balance'].data.item()
                self.iter_info['sep'] = stats['separation'].data.item()
                self.iter_info['used_leaves'] = int(stats['used_leaves'].data.item())
                self.iter_info['prc_leaves'] = len(self._model_core().soft_leaf_ids)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value) if loss_value else 0
        if self.arg.prc_mode == 'soft' and soft_leaf_score_sum is not None:
            self.soft_leaf_scores = soft_leaf_score_sum / max(soft_leaf_score_count, 1)
            self.soft_leaf_score_ids = soft_leaf_score_ids
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Progressive Recursive Clustering pre-training')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='epochs where optimizer reduces lr')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='joint, motion, or bone stream')

        parser.add_argument('--prc_mode', type=str, default='hard', choices=['hard', 'soft'],
                            help='hard uses recursive k-means pseudo labels; soft uses a differentiable tree loss')
        parser.add_argument('--prc_reassign', type=int, default=1, help='epochs between PRC tree updates')
        parser.add_argument('--prc_force_root_split', type=str2bool, default=True, help='bootstrap with root split')
        parser.add_argument('--prc_kmeans_iters', type=int, default=30, help='local k-means iterations')
        parser.add_argument('--prc_routing_temperature', type=float, default=0.2, help='soft tree routing temperature')
        parser.add_argument('--prc_grow_confidence_threshold', type=float, default=0.9,
                            help='skip hard PRC growth when mean route confidence is below this value')
        parser.add_argument('--prc_target_view', type=int, default=1,
                            help='view index used for hard PRC tree updates and path targets')
        parser.add_argument('--prc_consistency_view', type=int, default=0,
                            help='view index used for hard PRC augmentation consistency')
        parser.add_argument('--prc_consistency_weight', type=float, default=0.0,
                            help='weight for hard PRC path consistency on another augmented view')
        parser.add_argument('--prc_ce_weight', type=float, default=1.0,
                            help='weight for hard PRC path cross entropy')
        parser.add_argument('--prc_depth_penalty_weight', type=float, default=100.0,
                            help='weight for depth-dependent hard PRC split penalty')
        parser.add_argument('--prc_parent_size_penalty_weight', type=float, default=1.0,
                            help='weight for parent-size hard PRC split penalty')
        parser.add_argument('--prc_tiny_child_penalty_weight', type=float, default=1.0,
                            help='weight for tiny-child hard PRC split penalty')
        parser.add_argument('--prc_balanced_assignment_base_ratio', type=float, default=0.2,
                            help='base minimum child ratio for balanced hard PRC assignment')
        parser.add_argument('--prc_balanced_assignment_floor_ratio', type=float, default=0.02,
                            help='floor minimum child ratio for balanced hard PRC assignment')
        parser.add_argument('--prc_entropy_weight', type=float, default=0.05, help='weight for marginal entropy floor')
        parser.add_argument('--prc_entropy_floor', type=float, default=0.35, help='normalized marginal entropy floor')
        parser.add_argument('--prc_leaf_variance_weight', type=float, default=0.05,
                            help='weight for inverse within-leaf feature variance')
        parser.add_argument('--prc_leaf_variance_eps', type=float, default=0.01,
                            help='epsilon for inverse within-leaf variance penalty')
        parser.add_argument('--prc_entropy_min_samples', type=int, default=8, help='minimum node samples for entropy term')
        parser.add_argument('--prc_seed', type=int, default=0, help='PRC random seed')
        parser.add_argument('--prc_save_tree', type=str2bool, default=True, help='save tree snapshots')

        parser.add_argument('--prc_soft_temperature', type=float, default=0.2, help='soft routing temperature')
        parser.add_argument('--prc_soft_grow_start_epoch', type=int, default=20,
                            help='epoch to start growing the differentiable soft tree')
        parser.add_argument('--prc_soft_grow_interval', type=int, default=20,
                            help='epochs between soft tree growth steps')
        parser.add_argument('--prc_soft_grow_leaves', type=int, default=1,
                            help='number of high-cost leaves to split at each growth step')
        parser.add_argument('--prc_soft_max_leaves', type=int, default=0,
                            help='optional safety cap for soft leaves; <=0 means uncapped')
        parser.add_argument('--prc_soft_split_noise', type=float, default=0.01,
                            help='prototype perturbation when a soft leaf is split')
        parser.add_argument('--prc_soft_compactness_weight', type=float, default=1.0,
                            help='weight for soft assignment prototype compactness')
        parser.add_argument('--prc_soft_balance_weight', type=float, default=0.1,
                            help='weight for global leaf balance KL')
        parser.add_argument('--prc_soft_node_balance_weight', type=float, default=0.05,
                            help='weight for per-node branch balance KL')
        parser.add_argument('--prc_soft_separation_weight', type=float, default=0.1,
                            help='weight for prototype separation hinge')
        parser.add_argument('--prc_soft_separation_margin', type=float, default=0.5,
                            help='minimum squared distance between normalized leaf prototypes')
        parser.add_argument('--prc_soft_confidence_weight', type=float, default=0.02,
                            help='weight for low-entropy leaf assignment')
        parser.add_argument('--prc_soft_confidence_start_epoch', type=int, default=20,
                            help='epoch to start sharpening leaf assignments')

        return parser
