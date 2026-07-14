import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    """ST-GCN pretraining with ReSA and the OSE prototype loss."""

    def load_model(self):
        self.model = self.io.load_model(
            self.arg.model, **self.arg.model_args)
        self.model.apply(_weights_init)
        self.model.reset_momentum_encoder()

    def load_data(self):
        super().load_data()
        self._select_exemplars()

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
            raise ValueError('ReSA+Lproto requires labels for exemplar selection')

        labels = np.asarray(dataset.label)
        class_ids = sorted(np.unique(labels).tolist())
        if self.arg.ose_num_class > 0:
            class_ids = class_ids[:self.arg.ose_num_class]

        path = self.arg.ose_exemplar_index_path
        if path and os.path.isfile(path):
            payload = np.load(path, allow_pickle=True).item()
            self.ose_class_ids = list(payload['class_ids'])
            self.ose_exemplar_indices = list(payload['indices'])
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
                })

        preview = ', '.join(
            '{}:{}'.format(class_id, index)
            for class_id, index in zip(
                self.ose_class_ids[:10], self.ose_exemplar_indices[:10]))
        self.io.print_log(
            'OSE exemplars | classes {} | seed {} | {}'.format(
                len(self.ose_class_ids), self.arg.ose_exemplar_seed, preview))

    @staticmethod
    def _parse_batch(batch):
        if len(batch) == 3:
            data_pack, _, _ = batch
        elif len(batch) == 2:
            data_pack, _ = batch
        else:
            raise ValueError('Unsupported ReSA batch format')
        return data_pack

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

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []

        for batch_index, batch in enumerate(loader):
            self.global_step += 1
            data_pack = self._parse_batch(batch)
            if len(data_pack) == 2:
                weak_view_a, weak_view_b = data_pack
            elif len(data_pack) == 3:
                _, weak_view_a, weak_view_b = data_pack
            else:
                raise ValueError('ReSA requires exactly two weak views')
            weak_view_a = self._prepare_stream(
                weak_view_a.float().to(self.dev, non_blocking=True))
            weak_view_b = self._prepare_stream(
                weak_view_b.float().to(self.dev, non_blocking=True))
            exemplar = self._exemplar_batch()

            progress = self._training_progress(
                epoch, batch_index, len(loader))
            self._set_learning_rate(progress)
            momentum = self._momentum(progress)

            losses = self.model(
                weak_view_a, weak_view_b, exemplar,
                momentum=momentum, ose_topk=self.arg.ose_topk,
                ose_alpha=self.arg.ose_alpha,
                ose_tau_s=self.arg.ose_tau_s,
                ose_tau_t=self.arg.ose_tau_t)
            loss = losses['cluster'] + self.arg.ose_lambda * losses['proto']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.item()
            self.iter_info['cluster'] = losses['cluster'].item()
            self.iter_info['cluster_h'] = losses['cluster_entropy'].item()
            self.iter_info['cluster_kl'] = losses['cluster_kl'].item()
            self.iter_info['proto'] = losses['proto'].item()
            self.iter_info['align'] = losses['align'].item()
            self.iter_info['disp'] = losses['disp'].item()
            self.iter_info['target_h'] = losses['target_entropy'].item()
            self.iter_info['align_kl'] = losses['align_kl'].item()
            self.iter_info['queue'] = int(losses['queue_fill'].item())
            self.iter_info['ema_m'] = '{:.6f}'.format(momentum)
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_values.append(loss.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_values)
        self.train_writer.add_scalar(
            'loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

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
        parser.add_argument('--ose_num_class', type=int, default=0)
        parser.add_argument('--ose_exemplar_seed', type=int, default=0)
        parser.add_argument('--ose_exemplar_index_path', type=str, default='')
        parser.add_argument('--ose_topk', type=int, default=8)
        parser.add_argument('--ose_alpha', type=float, default=0.75)
        parser.add_argument('--ose_tau_s', type=float, default=0.04)
        parser.add_argument('--ose_tau_t', type=float, default=0.1)
        parser.add_argument('--ose_lambda', type=float, default=1.0)
        return parser
