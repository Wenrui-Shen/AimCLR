import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import str2bool

from .pretrain_aimclr import AimCLR_Processor


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


class OSEAimCLR_Processor(AimCLR_Processor):
    """AimCLR A2 with shared-queue P1 prototypes and constrained NNM."""

    def load_model(self):
        if self.arg.ose_enabled and len(self.arg.device) != 1:
            raise ValueError(
                'AimCLR+OSE formal training is single-GPU only because its '
                'queues are not synchronized across DataParallel replicas')
        if self.arg.ose_mu is not None:
            if (self.arg.ose_mix_proto_weight != 0.0 or
                    self.arg.ose_mix_ins_weight != 0.0):
                raise ValueError(
                    'Use either legacy ose_mu or the two explicit Lmix '
                    'weights, not both')
            self.arg.ose_mix_proto_weight = self.arg.ose_mu
            self.arg.ose_mix_ins_weight = self.arg.ose_mu
        if self.arg.ose_mix_proto_weight < 0:
            raise ValueError('ose_mix_proto_weight must be non-negative')
        if self.arg.ose_mix_ins_weight < 0:
            raise ValueError('ose_mix_ins_weight must be non-negative')
        if self.arg.ose_topk < 0:
            raise ValueError('ose_topk must be non-negative')
        if self.arg.ose_exemplar_views < 1:
            raise ValueError('ose_exemplar_views must be at least 1')
        if self.arg.ose_lambda < 0:
            raise ValueError('ose_lambda must be non-negative')
        if self.arg.ose_tau_s <= 0 or self.arg.ose_tau_t <= 0:
            raise ValueError('OSE temperatures must be positive')
        if not 0.0 <= self.arg.ose_alpha <= 1.0:
            raise ValueError('ose_alpha must be in [0, 1]')
        mix_enabled = (
            self.arg.ose_mix_proto_weight > 0 or
            self.arg.ose_mix_ins_weight > 0)
        if mix_enabled and not self.arg.ose_enabled:
            raise ValueError('Lmix cannot be enabled when OSE is disabled')
        if mix_enabled and self.arg.ose_mix_alpha <= 0:
            raise ValueError('ose_mix_alpha must be positive')

        model_args = dict(self.arg.model_args)
        model_args['ose_enabled'] = self.arg.ose_enabled
        self.model = self.io.load_model(self.arg.model, **model_args)
        self.model.apply(_weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.re_criterion = nn.L1Loss(reduction='none')
        self.model.reset_momentum_encoder()

        if not self.arg.ose_enabled:
            mode = 'AimCLR-only (A0)'
        elif (self.arg.ose_mix_proto_weight > 0 and
              self.arg.ose_mix_ins_weight > 0):
            mode = 'A2: AimCLR+P1-proto+M-F+OSE-constrained-NNM'
        elif not mix_enabled:
            mode = 'AimCLR+shared-queue-P1-proto'
        else:
            mode = 'AimCLR+shared-queue-P1-proto+partial-A2'
        self.io.print_log('Training mode | {}'.format(mode))
        if self.arg.ose_enabled:
            self.io.print_log(
                'OSE mix | proto_weight {:.4f} | ins_weight {:.4f} | '
                'beta_alpha {:.4f}'.format(
                    self.arg.ose_mix_proto_weight,
                    self.arg.ose_mix_ins_weight,
                    self.arg.ose_mix_alpha))
            self.io.print_log(
                'OSE A2 | shared_queue True | mutually_exclusive True | '
                'queue_neighbors {} | constrained_positive_max 1 | '
                'activation epoch>{} | '
                'exemplar_views {} (1 online + {} EMA)'.format(
                    self.arg.ose_topk,
                    self.arg.mining_epoch,
                    self.arg.ose_exemplar_views,
                    self.arg.ose_exemplar_views - 1))

    def load_data(self):
        super().load_data()
        if self.arg.ose_enabled:
            self._select_exemplars()
            if self.arg.ose_exclude_exemplars:
                self._exclude_exemplars_from_unlabeled_loader()

    def _select_exemplars(self):
        dataset = self.data_loader['train'].dataset
        if not hasattr(dataset, 'label'):
            raise ValueError(
                'AimCLR+Lproto requires labels for exemplar selection')
        if not getattr(dataset, 'return_index', False):
            raise ValueError(
                'AimCLR+OSE requires train_feeder_args.return_index: True')

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
                    raise ValueError(
                        'No samples found for class {}'.format(class_id))
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
                len(self.ose_class_ids),
                self.arg.ose_exemplar_seed, preview))

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
            raise ValueError('Unsupported AimCLR batch format')
        return data_pack, label, index

    def _prepare_stream(self, data):
        if self.arg.stream == 'joint':
            return data
        if self.arg.stream == 'motion':
            motion = torch.zeros_like(data)
            motion[:, :, :-1, :, :] = (
                data[:, :, 1:, :, :] - data[:, :, :-1, :, :])
            return motion
        if self.arg.stream == 'bone':
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
        raise ValueError('Unknown stream: {}'.format(self.arg.stream))

    def _exemplar_batch(self):
        dataset = self.data_loader['train'].dataset
        samples = []
        for index in self.ose_exemplar_indices:
            sample = np.array(dataset.data[index])
            if hasattr(dataset, '_aug'):
                sample = dataset._aug(sample)
            samples.append(sample)
        exemplars = torch.from_numpy(np.stack(samples, axis=0)).float()
        exemplars = exemplars.to(self.dev, non_blocking=True)
        return self._prepare_stream(exemplars)

    def _exemplar_batches(self):
        return [
            self._exemplar_batch()
            for _ in range(self.arg.ose_exemplar_views)
        ]

    def _ose_active(self, epoch):
        return self.arg.ose_enabled and epoch > self.arg.mining_epoch

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_values = []
        if self.arg.ose_enabled:
            neighbor_correct = np.zeros(
                len(self.ose_class_ids), dtype=np.int64)
            neighbor_total = np.zeros(
                len(self.ose_class_ids), dtype=np.int64)
            dataset_labels = np.asarray(loader.dataset.label)

        for batch in loader:
            self.global_step += 1
            data_pack, _, sample_indices = self._parse_batch(batch)
            if len(data_pack) != 3:
                raise ValueError('AimCLR requires exactly three training views')
            data1, data2, data3 = data_pack
            data1 = self._prepare_stream(
                data1.float().to(self.dev, non_blocking=True))
            data2 = self._prepare_stream(
                data2.float().to(self.dev, non_blocking=True))
            data3 = self._prepare_stream(
                data3.float().to(self.dev, non_blocking=True))

            compute_ose = self._ose_active(epoch)
            if self.arg.ose_enabled:
                if sample_indices is None:
                    raise ValueError(
                        'AimCLR A2 requires sample indices for its shared '
                        'queue sidecar')
                sample_indices = sample_indices.long().to(
                    self.dev, non_blocking=True)
            if compute_ose:
                exemplar_views = self._exemplar_batches()
                exemplar = exemplar_views[0]
            else:
                exemplar_views = []
                exemplar = None

            compute_mix_proto = (
                compute_ose and self.arg.ose_mix_proto_weight > 0)
            compute_mix_ins = (
                compute_ose and self.arg.ose_mix_ins_weight > 0)
            compute_mix = compute_mix_proto or compute_mix_ins
            mixed_view = None
            mix_index = None
            mix_beta = None
            if compute_mix:
                mix_index = torch.randperm(
                    data2.size(0), device=data2.device)
                mix_beta = float(np.random.beta(
                    self.arg.ose_mix_alpha, self.arg.ose_mix_alpha))
                mixed_view = (
                    mix_beta * data2 +
                    (1.0 - mix_beta) * data3[mix_index])

            nnm = epoch > self.arg.mining_epoch
            output1, target1, output2, output3, target2, ose_losses = (
                self.model(
                    data1, data2, data3, nnm=nnm, topk=self.arg.topk,
                    exemplar=exemplar, mixed_view=mixed_view,
                    mix_index=mix_index, mix_beta=mix_beta,
                    compute_ose=compute_ose,
                    compute_mix_proto=compute_mix_proto,
                    compute_mix_ins=compute_mix_ins,
                    extra_exemplar_views=exemplar_views[1:],
                    sample_indices=sample_indices,
                    ose_topk=self.arg.ose_topk,
                    ose_alpha=self.arg.ose_alpha,
                    ose_tau_s=self.arg.ose_tau_s,
                    ose_tau_t=self.arg.ose_tau_t))
            model = self.model.module if hasattr(
                self.model, 'module') else self.model
            model.update_ptr(output1.size(0))

            if target1.dim() == 2:
                loss1 = -(
                    F.log_softmax(output1, dim=1) * target1
                ).sum(1) / target1.sum(1)
                loss1 = loss1.mean()
            else:
                loss1 = self.loss(output1, target1)
            loss2 = -torch.mean(torch.sum(
                torch.log(output2.clamp_min(1e-12)) * target2, dim=1))
            loss3 = -torch.mean(torch.sum(
                torch.log(output3.clamp_min(1e-12)) * target2, dim=1))
            base_loss = loss1 + (loss2 + loss3) / 2.0
            loss = base_loss

            zero = base_loss.new_tensor(0.0)
            proto_loss = zero
            align_loss = zero
            disp_loss = zero
            mix_proto_loss = zero
            mix_ins_loss = zero
            target_entropy = zero
            align_kl = zero
            queue_fill = 0
            prototype_components = zero
            neighbor_overlap = zero
            nnm_positive_rate = zero
            nnm_candidate_count = zero
            nnm_same_sample_filtered = zero
            batch_purity = 0.0
            if ose_losses is not None:
                proto_loss = ose_losses['proto']
                align_loss = ose_losses['align']
                disp_loss = ose_losses['disp']
                mix_proto_loss = ose_losses['mix_proto']
                mix_ins_loss = ose_losses['mix_ins']
                target_entropy = ose_losses['target_entropy']
                align_kl = ose_losses['align_kl']
                queue_fill = int(ose_losses['queue_fill'].item())
                prototype_components = ose_losses[
                    'prototype_component_counts'].float()
                neighbor_overlap = ose_losses['neighbor_overlap_rate']
                nnm_positive_rate = ose_losses['nnm_positive_rate']
                nnm_candidate_count = ose_losses['nnm_candidate_count']
                nnm_same_sample_filtered = ose_losses[
                    'nnm_same_sample_filtered']
                loss = loss + self.arg.ose_lambda * proto_loss
                if compute_mix_proto:
                    loss = (
                        loss + self.arg.ose_mix_proto_weight *
                        mix_proto_loss)
                if compute_mix_ins:
                    loss = (
                        loss + self.arg.ose_mix_ins_weight *
                        mix_ins_loss)

                selected_indices = ose_losses[
                    'neighbor_sample_indices'].detach().cpu().numpy()
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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.item()
            self.iter_info['base'] = base_loss.item()
            self.iter_info['proto'] = proto_loss.item()
            self.iter_info['align'] = align_loss.item()
            self.iter_info['disp'] = disp_loss.item()
            self.iter_info['mix_p'] = mix_proto_loss.item()
            self.iter_info['mix_i'] = mix_ins_loss.item()
            self.iter_info['target_h'] = target_entropy.item()
            self.iter_info['align_kl'] = align_kl.item()
            self.iter_info['queue'] = queue_fill
            self.iter_info['components'] = (
                '{:.2f}'.format(prototype_components.mean().item())
                if prototype_components.numel() > 0 else '0.00')
            self.iter_info['nn_overlap'] = '{:.4f}'.format(
                neighbor_overlap.item())
            self.iter_info['nnm_pos_rate'] = '{:.4f}'.format(
                nnm_positive_rate.item())
            self.iter_info['nnm_candidates'] = '{:.2f}'.format(
                nnm_candidate_count.item())
            self.iter_info['same_filtered'] = '{:.0f}'.format(
                nnm_same_sample_filtered.item())
            self.iter_info['nn_purity'] = '{:.4f}'.format(batch_purity)
            self.iter_info['ose'] = int(compute_ose)
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_values.append(loss.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_values)
        self.train_writer.add_scalar(
            'loss', self.epoch_info['train_mean_loss'], epoch)
        if not self.arg.ose_enabled:
            self.show_epoch_info()
            return

        total_neighbors = int(neighbor_total.sum())
        epoch_purity = (float(neighbor_correct.sum()) / total_neighbors
                        if total_neighbors > 0 else 0.0)
        self.epoch_info['neighbor_purity'] = epoch_purity
        self.train_writer.add_scalar('neighbor_purity', epoch_purity, epoch)
        self.show_epoch_info()
        random_purity = 1.0 / max(len(self.ose_class_ids), 1)
        self.io.print_log(
            'OSE neighbor diagnostic | purity {:.4f} | random {:.4f}'.format(
                epoch_purity, random_purity))
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
        parent_parser = AimCLR_Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help, parents=[parent_parser],
            description='AimCLR with one-shot exemplar prototype learning')

        parser.add_argument('--ose_enabled', type=str2bool, default=True)
        parser.add_argument('--ose_num_class', type=int, default=0)
        parser.add_argument('--ose_exemplar_seed', type=int, default=0)
        parser.add_argument('--ose_exemplar_index_path', type=str, default='')
        parser.add_argument('--ose_exclude_exemplars', type=str2bool,
                            default=True)
        parser.add_argument('--ose_topk', type=int, default=4)
        parser.add_argument(
            '--ose_exemplar_views', type=int, default=1,
            help='total weak exemplar views: one online plus EMA views')
        parser.add_argument('--ose_alpha', type=float, default=0.75)
        parser.add_argument('--ose_tau_s', type=float, default=0.1)
        parser.add_argument('--ose_tau_t', type=float, default=0.04)
        parser.add_argument('--ose_lambda', type=float, default=1.0)
        parser.add_argument('--ose_mix_proto_weight', type=float, default=0.0)
        parser.add_argument('--ose_mix_ins_weight', type=float, default=0.0)
        parser.add_argument('--ose_mix_alpha', type=float, default=1.0)
        parser.add_argument(
            '--ose_mu', type=float, default=None,
            help='legacy alias that sets both explicit Lmix weights')
        return parser
