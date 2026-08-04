import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchlight import str2bool

from .pretrain import PT_Processor


def _weights_init(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            module.weight.data.normal_(1.0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()


class OSEResAProcessor(PT_Processor):
    """ST-GCN pretraining with ReSA and the OSE losses."""

    def load_model(self):
        modalities = self.arg.ose_exemplar_modalities
        if isinstance(modalities, str):
            modalities = [modalities]
        modalities = tuple(str(name).lower() for name in modalities)
        supported_modalities = ('joint', 'motion', 'bone')
        if not modalities:
            raise ValueError(
                'ose_exemplar_modalities must contain at least one stream')
        unknown_modalities = [
            name for name in modalities
            if name not in supported_modalities
        ]
        if unknown_modalities:
            raise ValueError(
                'Unsupported exemplar modalities: {}'.format(
                    unknown_modalities))
        if len(set(modalities)) != len(modalities):
            raise ValueError('ose_exemplar_modalities must not repeat')
        if self.arg.ose_enabled and self.arg.stream not in modalities:
            raise ValueError(
                'The training stream {} must be included in '
                'ose_exemplar_modalities'.format(self.arg.stream))
        self.ose_exemplar_modalities = modalities
        if self.arg.ose_mix_proto_weight < 0:
            raise ValueError('ose_mix_proto_weight must be non-negative')
        if self.arg.ose_mix_ins_weight < 0:
            raise ValueError('ose_mix_ins_weight must be non-negative')
        if self.arg.resa_weight < 0:
            raise ValueError('resa_weight must be non-negative')
        if self.arg.resa_weight == 0 and not self.arg.ose_enabled:
            raise ValueError(
                'At least one of ReSA or OSE must be enabled')
        if self.arg.ose_topk < 0:
            raise ValueError('ose_topk must be non-negative')
        if self.arg.ose_exemplar_views < 1:
            raise ValueError('ose_exemplar_views must be at least 1')
        if self.arg.ose_prototype_stage not in (0, 1, 2, 3):
            raise ValueError(
                'ose_prototype_stage must be one of 0, 1, 2, 3')
        if self.arg.queue_contrast_weight < 0:
            raise ValueError('queue_contrast_weight must be non-negative')
        if self.arg.smoke_test_iterations < 0:
            raise ValueError('smoke_test_iterations must be non-negative')
        if (self.arg.smoke_test_iterations > 0 and
                self.arg.num_epoch != 1):
            raise ValueError(
                'Smoke tests require --num_epoch 1 to avoid a long run')
        if (self.arg.queue_contrast_weight > 0 and
                not math.isclose(self.arg.queue_contrast_weight, 1.0)):
            raise ValueError(
                'The current protocol fixes queue_contrast_weight at 1.0')
        mix_enabled = (
            self.arg.ose_mix_proto_weight > 0 or
            self.arg.ose_mix_ins_weight > 0)
        queue_contrast_enabled = self.arg.queue_contrast_weight > 0
        if mix_enabled and not self.arg.ose_enabled:
            raise ValueError('Lmix cannot be enabled when OSE is disabled')
        if queue_contrast_enabled and not self.arg.ose_enabled:
            raise ValueError(
                'Category-corrected queue contrast requires OSE')
        if mix_enabled and self.arg.ose_mix_alpha <= 0:
            raise ValueError('ose_mix_alpha must be positive')

        model_args = dict(self.arg.model_args)
        model_args['ose_enabled'] = self.arg.ose_enabled
        model_args['ose_prototype_stage'] = self.arg.ose_prototype_stage
        model_args['queue_contrast_enabled'] = queue_contrast_enabled
        self.model = self.io.load_model(
            self.arg.model, **model_args)
        self.model.apply(_weights_init)
        if hasattr(self.model.encoder_q, 'reset_parameters'):
            self.model.encoder_q.reset_parameters()
        self.model.reset_momentum_encoder()
        if not self.arg.ose_enabled:
            mode = 'ReSA-only'
        elif self.arg.resa_weight == 0:
            mode = 'OSE-only'
        elif mix_enabled and queue_contrast_enabled:
            mode = 'ReSA+Lproto+Lmix+Lqueue-corr'
        elif mix_enabled:
            mode = 'ReSA+Lproto+Lmix'
        elif queue_contrast_enabled:
            mode = 'ReSA+Lproto+Lqueue-corr'
        else:
            mode = 'ReSA+Lproto'
        self.io.print_log('Training mode | {} | OSE {}'.format(
            mode, 'enabled' if self.arg.ose_enabled else 'disabled'))
        self.io.print_log(
            'Loss weights | ReSA {:.4f} | OSE prototype {:.4f}'.format(
                self.arg.resa_weight, self.arg.ose_lambda))
        if self.arg.ose_match_exemplar_split and not self.arg.ose_enabled:
            self.io.print_log(
                'ReSA-only split | exclude the same one-shot exemplars as '
                'OSE ablations')
        if self.arg.ose_enabled:
            projector_mode = ('separate ReSA/OSE projectors'
                              if self.model.ose_separate_projector
                              else 'shared ReSA/OSE projector')
            self.io.print_log('Projector mode | {}'.format(projector_mode))
        if self.arg.ose_enabled:
            self.io.print_log(
                'OSE mix | proto_weight {:.4f} | ins_weight {:.4f} | '
                'beta_alpha {:.4f}'.format(
                    self.arg.ose_mix_proto_weight,
                    self.arg.ose_mix_ins_weight,
                    self.arg.ose_mix_alpha))
            self.io.print_log(
                'OSE prototype | stage P{} | queue_neighbors {} | '
                'exemplar_views {} (1 online + {} EMA)'.format(
                    self.arg.ose_prototype_stage,
                    self.arg.ose_topk,
                    self.arg.ose_exemplar_views,
                    self.arg.ose_exemplar_views - 1))
            ema_modalities = [
                name for name in self.ose_exemplar_modalities
                if name != self.arg.stream
            ]
            self.io.print_log(
                'OSE labeled modalities | online {} | EMA extras {}'.format(
                    self.arg.stream,
                    ','.join(ema_modalities) if ema_modalities else 'none'))
        if queue_contrast_enabled:
            self.io.print_log(
                'Corrected weak queue | weight {:.1f} | dim {} | size {} | '
                'temperature {:.4f}'.format(
                    self.arg.queue_contrast_weight,
                    self.model.instance_feature_dim,
                    self.model.instance_queue_size,
                    self.model.instance_temperature))

    def train_log_writer(self, epoch):
        super().train_log_writer(epoch)
        if self.arg.queue_contrast_weight > 0:
            for name in (
                    'queue_corr', 'mean_category_confidence',
                    'mean_negative_weight', 'min_negative_weight'):
                self.train_writer.add_scalar(
                    name, self.iter_info[name], self.global_step)

    def load_data(self):
        super().load_data()
        if self.arg.ose_enabled or self.arg.ose_match_exemplar_split:
            self._select_exemplars()
            if self.arg.ose_exclude_exemplars:
                self._exclude_exemplars_from_unlabeled_loader()

    def load_optimizer(self):
        parameters = [parameter for parameter in self.model.parameters()
                      if parameter.requires_grad]
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters, lr=self.arg.base_lr, momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters, lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(
                self.arg.optimizer))

    def _select_exemplars(self):
        dataset = self.data_loader['train'].dataset
        if not hasattr(dataset, 'label'):
            raise ValueError('Exemplar selection requires dataset labels')
        if not getattr(dataset, 'return_index', False):
            raise ValueError(
                'OSE neighbor diagnostics require train_feeder_args.'
                'return_index: True')

        labels = np.asarray(dataset.label)
        class_ids = sorted(np.unique(labels).tolist())
        if self.arg.ose_num_class > 0:
            class_ids = class_ids[:self.arg.ose_num_class]

        path = self.arg.ose_exemplar_index_path
        if path and os.path.isfile(path):
            payload = np.load(path, allow_pickle=True).item()
            cached_class_ids = np.asarray(payload['class_ids']).tolist()
            cached_indices = np.asarray(
                payload['indices'], dtype=np.int64).tolist()
            cached_seed = payload.get('seed')
            cached_num_samples = payload.get('num_samples')

            errors = []
            if cached_seed is None or int(cached_seed) != int(
                    self.arg.ose_exemplar_seed):
                errors.append('seed {} != {}'.format(
                    cached_seed, self.arg.ose_exemplar_seed))
            if cached_class_ids != class_ids:
                errors.append('class IDs do not match the current dataset')
            if len(cached_indices) != len(cached_class_ids):
                errors.append('class and exemplar counts differ')
            if (cached_num_samples is not None and
                    int(cached_num_samples) != len(labels)):
                errors.append('dataset size {} != {}'.format(
                    cached_num_samples, len(labels)))
            for class_id, index in zip(cached_class_ids, cached_indices):
                if index < 0 or index >= len(labels):
                    errors.append('index {} is out of range'.format(index))
                    break
                if labels[index] != class_id:
                    errors.append(
                        'index {} has label {}, expected {}'.format(
                            index, labels[index], class_id))
                    break
            if errors:
                raise ValueError(
                    'Invalid OSE exemplar cache {}: {}. Use a cache path '
                    'specific to the seed and dataset.'.format(
                        path, '; '.join(errors)))

            self.ose_class_ids = cached_class_ids
            self.ose_exemplar_indices = cached_indices
        else:
            rng = np.random.RandomState(self.arg.ose_exemplar_seed)
            indices = []
            for class_id in class_ids:
                candidates = np.where(labels == class_id)[0]
                if candidates.size == 0:
                    raise ValueError('No samples found for class {}'.format(
                        class_id))
                indices.append(int(rng.choice(candidates)))
            self.ose_class_ids = class_ids
            self.ose_exemplar_indices = indices
            if path:
                folder = os.path.dirname(path)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                np.save(path, {
                    'class_ids': np.asarray(class_ids),
                    'indices': np.asarray(indices, dtype=np.int64),
                    'seed': int(self.arg.ose_exemplar_seed),
                    'num_samples': len(labels),
                })

        preview = ', '.join(
            '{}:{}'.format(class_id, index)
            for class_id, index in zip(
                self.ose_class_ids[:10], self.ose_exemplar_indices[:10]))
        self.io.print_log(
            'OSE exemplars | classes {} | seed {} | {}'.format(
                len(self.ose_class_ids), self.arg.ose_exemplar_seed, preview))

    def _exclude_exemplars_from_unlabeled_loader(self):
        loader = self.data_loader['train']
        dataset = loader.dataset
        excluded = set(self.ose_exemplar_indices)
        unlabeled_indices = [
            index for index in range(len(dataset)) if index not in excluded]
        if len(unlabeled_indices) < loader.batch_size:
            raise ValueError(
                'Not enough unlabeled samples after excluding OSE exemplars')

        sampler = torch.utils.data.SubsetRandomSampler(unlabeled_indices)
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            timeout=loader.timeout,
            worker_init_fn=loader.worker_init_fn)
        self.io.print_log(
            'OSE unlabeled split | {} samples | excluded {} exemplars'.format(
                len(unlabeled_indices), len(excluded)))

    @staticmethod
    def _parse_batch(batch):
        if len(batch) == 3:
            data_pack, label, index = batch
        elif len(batch) == 2:
            data_pack, label = batch
            index = None
        else:
            raise ValueError('Unsupported ReSA batch format')
        return data_pack, label, index

    def _prepare_stream(self, data, stream=None):
        stream = self.arg.stream if stream is None else stream
        if stream == 'joint':
            return data
        if stream == 'motion':
            motion = torch.zeros_like(data)
            motion[:, :, :-1, :, :] = (
                data[:, :, 1:, :, :] - data[:, :, :-1, :, :])
            return motion
        if stream == 'bone':
            bone_pairs = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (21, 21), (22, 23), (23, 8), (24, 25), (25, 12),
            ]
            bone = torch.zeros_like(data)
            for v1, v2 in bone_pairs:
                bone[:, :, :, v1 - 1, :] = (
                    data[:, :, :, v1 - 1, :] -
                    data[:, :, :, v2 - 1, :])
            return bone
        raise ValueError('Unknown stream: {}'.format(stream))

    def _raw_exemplar_batch(self):
        dataset = self.data_loader['train'].dataset
        samples = []
        for index in self.ose_exemplar_indices:
            sample = np.array(dataset.data[index])
            if hasattr(dataset, '_aug'):
                sample = dataset._aug(sample)
            samples.append(sample)
        exemplars = torch.from_numpy(np.stack(samples, axis=0)).float()
        exemplars = exemplars.to(self.dev, non_blocking=True)
        return exemplars

    def _exemplar_batch(self, stream=None):
        return self._prepare_stream(
            self._raw_exemplar_batch(), stream=stream)

    def _exemplar_batches(self):
        # Derive all structural modalities from the same augmented exemplar;
        # otherwise random augmentation would be confounded with modality.
        raw_exemplars = self._raw_exemplar_batch()
        batches = [self._prepare_stream(
            raw_exemplars, stream=self.arg.stream)]
        batches.extend([
            self._prepare_stream(raw_exemplars, stream=modality)
            for modality in self.ose_exemplar_modalities
            if modality != self.arg.stream
        ])
        # Preserve the older independent-view option in addition to the new
        # deterministic structural modalities.
        batches.extend([
            self._exemplar_batch(stream=self.arg.stream)
            for _ in range(self.arg.ose_exemplar_views - 1)
        ])
        return batches

    def _training_progress(self, epoch, batch_index, num_batches):
        return (epoch - 1) + float(batch_index + 1) / max(num_batches, 1)

    def _set_learning_rate(self, progress):
        warmup = float(self.arg.resa_warmup_epoch)
        if warmup > 0 and progress <= warmup:
            lr = self.arg.base_lr * progress / warmup
        else:
            decay_progress = (progress - warmup) / max(
                self.arg.num_epoch - warmup, 1.0)
            decay_progress = min(max(decay_progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            lr = (self.arg.resa_final_lr +
                  (self.arg.base_lr - self.arg.resa_final_lr) * cosine)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def _momentum(self, progress):
        progress = min(max(progress / max(self.arg.num_epoch, 1), 0.0), 1.0)
        return 1.0 - (1.0 - self.arg.resa_momentum) * (
            math.cos(math.pi * progress) + 1.0) / 2.0

    # Stage2 overrides these hooks to collect transition diagnostics without
    # duplicating the ReSA/OSE training loop used by the from-scratch model.
    def _diagnostics_epoch_start(self, epoch):
        pass

    def _diagnostics_before_backward(self, batch_index, resa_objective,
                                     ose_objective):
        return {}

    def _diagnostics_after_backward(self, batch_index):
        return {}

    def _diagnostics_record_batch(self, epoch, losses, total_loss,
                                  diagnostics):
        pass

    def _diagnostics_epoch_end(self, epoch):
        pass

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        self._diagnostics_epoch_start(epoch)
        if self.arg.ose_enabled:
            neighbor_correct = np.zeros(
                len(self.ose_class_ids), dtype=np.int64)
            neighbor_total = np.zeros(
                len(self.ose_class_ids), dtype=np.int64)
            component_count_total = np.zeros(
                len(self.ose_class_ids), dtype=np.int64)
            component_count_batches = 0
            overlap_rates = []
            dataset_labels = np.asarray(loader.dataset.label)

        for batch_index, batch in enumerate(loader):
            self.global_step += 1
            data_pack, _, sample_indices = self._parse_batch(batch)
            if len(data_pack) == 2:
                view_a, view_b = data_pack
            elif len(data_pack) == 3:
                _, view_a, view_b = data_pack
            else:
                raise ValueError('ReSA requires exactly two training views')
            view_a = self._prepare_stream(
                view_a.float().to(self.dev, non_blocking=True))
            view_b = self._prepare_stream(
                view_b.float().to(self.dev, non_blocking=True))
            if self.arg.ose_enabled and sample_indices is not None:
                sample_indices = sample_indices.long().to(
                    self.dev, non_blocking=True)

            compute_mix_proto = (
                self.arg.ose_enabled and
                self.arg.ose_mix_proto_weight > 0)
            compute_mix_ins = (
                self.arg.ose_enabled and
                self.arg.ose_mix_ins_weight > 0)
            compute_mix = compute_mix_proto or compute_mix_ins
            mixed_view = None
            mix_index = None
            mix_beta = None
            if compute_mix:
                mix_index = torch.randperm(
                    view_a.size(0), device=view_a.device)
                mix_beta = float(np.random.beta(
                    self.arg.ose_mix_alpha, self.arg.ose_mix_alpha))
                # Eq. (10): x is the online Lproto view (view_b), while
                # x' is the shuffled teacher Lproto view (view_a).
                mixed_view = (
                    mix_beta * view_b +
                    (1.0 - mix_beta) * view_a[mix_index])

            progress = self._training_progress(
                epoch, batch_index, len(loader))
            self._set_learning_rate(progress)
            momentum = self._momentum(progress)

            if self.arg.ose_enabled:
                exemplar_views = self._exemplar_batches()
                exemplar = exemplar_views[0]
                losses = self.model(
                    view_a, view_b, exemplar,
                    momentum=momentum, ose_topk=self.arg.ose_topk,
                    ose_alpha=self.arg.ose_alpha,
                    ose_tau_s=self.arg.ose_tau_s,
                    ose_tau_t=self.arg.ose_tau_t,
                    sample_indices=sample_indices,
                    mixed_view=mixed_view,
                    mix_index=mix_index,
                    mix_beta=mix_beta,
                    compute_mix_proto=compute_mix_proto,
                    compute_mix_ins=compute_mix_ins,
                    extra_exemplar_views=exemplar_views[1:])
                ose_objective = self.arg.ose_lambda * losses['proto']
                if compute_mix_proto:
                    ose_objective = (
                        ose_objective + self.arg.ose_mix_proto_weight *
                        losses['mix_proto'])
                if compute_mix_ins:
                    ose_objective = (
                        ose_objective + self.arg.ose_mix_ins_weight *
                        losses['mix_ins'])
                if self.arg.queue_contrast_weight > 0:
                    ose_objective = (
                        ose_objective + self.arg.queue_contrast_weight *
                        losses['queue_corr'])
                resa_objective = losses['cluster']
                loss = (self.arg.resa_weight * resa_objective +
                        ose_objective)

                selected_indices = losses[
                    'neighbor_sample_indices'].detach().cpu().numpy()
                component_counts = losses[
                    'prototype_component_counts'].detach().cpu().numpy()
                component_count_total += component_counts
                component_count_batches += 1
                overlap_rates.append(float(
                    losses['neighbor_overlap_rate'].item()))
                batch_purity = 0.0
                if selected_indices.size > 0:
                    valid = np.logical_and(
                        selected_indices >= 0,
                        selected_indices < len(dataset_labels))
                    selected_labels = np.full(
                        selected_indices.shape, -1, dtype=np.int64)
                    selected_labels[valid] = dataset_labels[
                        selected_indices[valid]]
                    expected_labels = np.asarray(
                        self.ose_class_ids, dtype=np.int64)[:, None]
                    correct = np.logical_and(
                        selected_labels == expected_labels, valid)
                    batch_correct = correct.sum(axis=1)
                    batch_total = valid.sum(axis=1)
                    neighbor_correct += batch_correct
                    neighbor_total += batch_total
                    if batch_total.sum() > 0:
                        batch_purity = float(
                            batch_correct.sum()) / float(batch_total.sum())
            else:
                losses = self.model(
                    view_a, view_b, momentum=momentum)
                resa_objective = losses['cluster']
                ose_objective = None
                loss = self.arg.resa_weight * resa_objective

            self.optimizer.zero_grad()
            diagnostics = self._diagnostics_before_backward(
                batch_index, resa_objective, ose_objective)
            loss.backward()
            diagnostics.update(
                self._diagnostics_after_backward(batch_index))
            self.optimizer.step()

            self.iter_info['loss'] = loss.item()
            self.iter_info['cluster'] = losses['cluster'].item()
            self.iter_info['cluster_h'] = losses['cluster_entropy'].item()
            self.iter_info['cluster_kl'] = losses['cluster_kl'].item()
            if self.arg.ose_enabled:
                self.iter_info['proto'] = losses['proto'].item()
                self.iter_info['align'] = losses['align'].item()
                self.iter_info['disp'] = losses['disp'].item()
                self.iter_info['mix'] = losses['mix'].item()
                self.iter_info['mix_p'] = losses['mix_proto'].item()
                self.iter_info['mix_i'] = losses['mix_ins'].item()
                self.iter_info['target_h'] = losses['target_entropy'].item()
                self.iter_info['align_kl'] = losses['align_kl'].item()
                self.iter_info['queue'] = int(losses['queue_fill'].item())
                self.iter_info['nn_purity'] = '{:.4f}'.format(batch_purity)
                self.iter_info['nn_overlap'] = '{:.4f}'.format(
                    losses['neighbor_overlap_rate'].item())
                self.iter_info['components'] = '{:.2f}'.format(
                    losses['prototype_component_counts'].float().mean().item())
                if self.arg.queue_contrast_weight > 0:
                    self.iter_info['queue_corr'] = (
                        losses['queue_corr'].item())
                    self.iter_info['mean_category_confidence'] = (
                        losses['mean_category_confidence'].item())
                    self.iter_info['mean_negative_weight'] = (
                        losses['mean_negative_weight'].item())
                    self.iter_info['min_negative_weight'] = (
                        losses['min_negative_weight'].item())
                    self.iter_info['instance_queue_ptr'] = int(
                        losses['instance_queue_ptr'].item())
            self.iter_info['ema_m'] = '{:.6f}'.format(momentum)
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self._diagnostics_record_batch(
                epoch, losses, loss, diagnostics)
            loss_values.append(loss.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            if (self.arg.smoke_test_iterations > 0 and
                    batch_index + 1 >= self.arg.smoke_test_iterations):
                self.io.print_log(
                    'Smoke test stopped after {} iterations.'.format(
                        self.arg.smoke_test_iterations))
                break

        self.epoch_info['train_mean_loss'] = np.mean(loss_values)
        self.train_writer.add_scalar(
            'loss', self.epoch_info['train_mean_loss'], epoch)
        self._diagnostics_epoch_end(epoch)
        if not self.arg.ose_enabled:
            self.show_epoch_info()
            return

        total_neighbors = int(neighbor_total.sum())
        epoch_purity = (float(neighbor_correct.sum()) / total_neighbors
                        if total_neighbors > 0 else 0.0)
        self.epoch_info['neighbor_purity'] = epoch_purity
        mean_overlap = float(np.mean(overlap_rates)) if overlap_rates else 0.0
        self.epoch_info['neighbor_overlap'] = mean_overlap
        self.train_writer.add_scalar('neighbor_purity', epoch_purity, epoch)
        self.train_writer.add_scalar('neighbor_overlap', mean_overlap, epoch)
        self.show_epoch_info()
        random_purity = 1.0 / max(len(self.ose_class_ids), 1)
        self.io.print_log(
            'OSE neighbor diagnostic | purity {:.4f} | random {:.4f} | '
            'overlap {:.4f}'.format(
                epoch_purity, random_purity, mean_overlap))
        if component_count_batches > 0:
            mean_components = (
                component_count_total.astype(np.float64) /
                float(component_count_batches))
            self.io.print_log(
                'OSE prototype components | mean {:.2f} | min {:.2f} | '
                'max {:.2f}'.format(
                    float(mean_components.mean()),
                    float(mean_components.min()),
                    float(mean_components.max())))
        details = []
        for class_index, class_id in enumerate(self.ose_class_ids):
            if neighbor_total[class_index] > 0:
                class_purity = (float(neighbor_correct[class_index]) /
                                float(neighbor_total[class_index]))
                correct_per_topk = self.arg.ose_topk * class_purity
                details.append('{}:{:.2f}/{}'.format(
                    class_id, correct_per_topk, self.arg.ose_topk))
            else:
                details.append('{}:n/a'.format(class_id))
        for start in range(0, len(details), 10):
            self.io.print_log(
                'OSE neighbor per class | ' +
                ' '.join(details[start:start + 10]))

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = PT_Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help, parents=[parent_parser],
            description='ReSA with one-shot exemplar prototype learning')

        parser.add_argument('--stream', type=str, default='joint')
        parser.add_argument('--resa_momentum', type=float, default=0.996)
        parser.add_argument('--resa_warmup_epoch', type=int, default=2)
        parser.add_argument('--resa_final_lr', type=float, default=0.0)
        parser.add_argument('--resa_weight', type=float, default=1.0)
        parser.add_argument('--ose_enabled', type=str2bool, default=True)
        parser.add_argument('--ose_num_class', type=int, default=0)
        parser.add_argument('--ose_exemplar_seed', type=int, default=0)
        parser.add_argument('--ose_exemplar_index_path', type=str, default='')
        parser.add_argument('--ose_exclude_exemplars', type=str2bool,
                            default=True)
        parser.add_argument(
            '--ose_match_exemplar_split', type=str2bool, default=False,
            help='select/exclude the same exemplars even when OSE is disabled')
        parser.add_argument('--ose_topk', type=int, default=8)
        parser.add_argument('--ose_prototype_stage', type=int, default=0)
        parser.add_argument(
            '--ose_exemplar_views', type=int, default=1,
            help='total weak exemplar views: one online plus EMA views')
        parser.add_argument(
            '--ose_exemplar_modalities', type=str, nargs='+',
            default=['joint'],
            help='labeled exemplar streams; training stream is online and '
                 'the remaining streams are EMA structural views')
        parser.add_argument('--ose_alpha', type=float, default=0.75)
        parser.add_argument('--ose_tau_s', type=float, default=0.1)
        parser.add_argument('--ose_tau_t', type=float, default=0.04)
        parser.add_argument('--ose_lambda', type=float, default=1.0)
        parser.add_argument('--ose_mix_proto_weight', type=float, default=0.0)
        parser.add_argument('--ose_mix_ins_weight', type=float, default=0.0)
        parser.add_argument('--ose_mix_alpha', type=float, default=1.0)
        parser.add_argument('--queue_contrast_weight', type=float, default=0.0)
        parser.add_argument(
            '--smoke_test_iterations', type=int, default=0,
            help='stop each smoke-test epoch after this many iterations')
        return parser
