import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from .pretrain_aimclr import AimCLR_Processor


class OSEAimCLR_Processor(AimCLR_Processor):
    """Processor for one-shot exemplar-guided AimCLR pre-training."""

    def load_data(self):
        super(OSEAimCLR_Processor, self).load_data()
        self._select_exemplars()

    def _select_exemplars(self):
        dataset = self.data_loader['train'].dataset
        if not hasattr(dataset, 'label'):
            raise ValueError('OSE-AimCLR requires training dataset labels to select one exemplar per class')

        labels = np.asarray(dataset.label)
        class_ids = sorted(np.unique(labels).tolist())
        if self.arg.ose_num_class > 0:
            class_ids = class_ids[:self.arg.ose_num_class]

        if self.arg.ose_exemplar_index_path and os.path.isfile(self.arg.ose_exemplar_index_path):
            payload = np.load(self.arg.ose_exemplar_index_path, allow_pickle=True).item()
            self.ose_class_ids = list(payload['class_ids'])
            self.ose_exemplar_indices = list(payload['indices'])
        else:
            rng = np.random.RandomState(self.arg.ose_exemplar_seed)
            exemplar_indices = []
            for class_id in class_ids:
                candidates = np.where(labels == class_id)[0]
                if len(candidates) == 0:
                    raise ValueError('No samples found for class {}'.format(class_id))
                exemplar_indices.append(int(rng.choice(candidates)))
            self.ose_class_ids = class_ids
            self.ose_exemplar_indices = exemplar_indices
            if self.arg.ose_exemplar_index_path:
                folder = os.path.dirname(self.arg.ose_exemplar_index_path)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                np.save(self.arg.ose_exemplar_index_path, {
                    'class_ids': np.asarray(self.ose_class_ids),
                    'indices': np.asarray(self.ose_exemplar_indices, dtype=np.int64),
                    'seed': int(self.arg.ose_exemplar_seed),
                })

        preview = ', '.join(
            '{}:{}'.format(c, i)
            for c, i in zip(self.ose_class_ids[:10], self.ose_exemplar_indices[:10]))
        self.io.print_log(
            'OSE exemplars | classes {} | seed {} | {}'.format(
                len(self.ose_class_ids), self.arg.ose_exemplar_seed, preview))

    def _parse_batch(self, batch):
        if len(batch) == 3:
            data_pack, label, index = batch
        elif len(batch) == 2:
            data_pack, label = batch
            index = None
        else:
            raise ValueError('Unsupported OSE-AimCLR batch format')
        return data_pack, label, index

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

    def _exemplar_batch(self):
        dataset = self.data_loader['train'].dataset
        samples = []
        for index in self.ose_exemplar_indices:
            data_numpy = np.array(dataset.data[index])
            if hasattr(dataset, '_aug'):
                data_numpy = dataset._aug(data_numpy)
            samples.append(data_numpy)
        data = torch.from_numpy(np.stack(samples, axis=0)).float().to(self.dev, non_blocking=True)
        return self._prepare_stream(data)

    def _ose_weight(self, epoch):
        if epoch <= self.arg.ose_warmup_epoch:
            return 0.0
        if self.arg.ose_ramp_epoch <= 0:
            return 1.0
        progress = float(epoch - self.arg.ose_warmup_epoch) / float(self.arg.ose_ramp_epoch)
        return min(max(progress, 0.0), 1.0)

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for batch in loader:
            self.global_step += 1
            data_pack, _, _ = self._parse_batch(batch)
            data1, data2, data3 = data_pack
            data1 = self._prepare_stream(data1.float().to(self.dev, non_blocking=True))
            data2 = self._prepare_stream(data2.float().to(self.dev, non_blocking=True))
            data3 = self._prepare_stream(data3.float().to(self.dev, non_blocking=True))

            ose_weight = self._ose_weight(epoch)
            compute_ose = ose_weight > 0
            exemplar = None
            im_mix = None
            mix_index = None
            mix_beta = None
            if compute_ose:
                exemplar = self._exemplar_batch()
                mix_index = torch.randperm(data2.size(0), device=data2.device)
                mix_beta = float(np.random.beta(self.arg.ose_mix_alpha, self.arg.ose_mix_alpha))
                im_mix = mix_beta * data2 + (1.0 - mix_beta) * data3[mix_index]

            if epoch <= self.arg.mining_epoch:
                output1, target1, output2, output3, target2, ose_losses = self.model(
                    data1, data2, data3,
                    exemplar=exemplar, im_mix=im_mix, mix_index=mix_index, mix_beta=mix_beta,
                    compute_ose=compute_ose, ose_topk=self.arg.ose_topk,
                    ose_alpha=self.arg.ose_alpha, ose_tau_s=self.arg.ose_tau_s,
                    ose_tau_t=self.arg.ose_tau_t)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = self.loss(output1, target1)
            else:
                output1, mask, output2, output3, target2, ose_losses = self.model(
                    data1, data2, data3, nnm=True, topk=self.arg.topk,
                    exemplar=exemplar, im_mix=im_mix, mix_index=mix_index, mix_beta=mix_beta,
                    compute_ose=compute_ose, ose_topk=self.arg.ose_topk,
                    ose_alpha=self.arg.ose_alpha, ose_tau_s=self.arg.ose_tau_s,
                    ose_tau_t=self.arg.ose_tau_t)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = - (F.log_softmax(output1, dim=1) * mask).sum(1) / mask.sum(1)
                loss1 = loss1.mean()

            loss2 = -torch.mean(torch.sum(torch.log(output2.clamp_min(1e-12)) * target2, dim=1))
            loss3 = -torch.mean(torch.sum(torch.log(output3.clamp_min(1e-12)) * target2, dim=1))
            base_loss = loss1 + (loss2 + loss3) / 2.
            proto_loss = base_loss.new_tensor(0.0)
            mix_loss = base_loss.new_tensor(0.0)
            if ose_losses is not None:
                proto_loss = ose_losses['proto']
                mix_loss = ose_losses['mix']
            loss = base_loss + ose_weight * (
                self.arg.ose_lambda * proto_loss +
                self.arg.ose_mu * mix_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['base'] = base_loss.data.item()
            self.iter_info['proto'] = proto_loss.data.item()
            self.iter_info['mix'] = mix_loss.data.item()
            self.iter_info['ose_w'] = '{:.3f}'.format(ose_weight)
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = AimCLR_Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='One-shot exemplar-guided AimCLR pre-training')

        parser.add_argument('--ose_num_class', type=int, default=0,
                            help='number of classes to ground; <=0 uses every class in labels')
        parser.add_argument('--ose_exemplar_seed', type=int, default=0,
                            help='fixed random seed for one exemplar per class')
        parser.add_argument('--ose_exemplar_index_path', type=str, default='',
                            help='optional .npy path to load/save selected exemplar indices')
        parser.add_argument('--ose_warmup_epoch', type=int, default=20,
                            help='epochs trained with plain AimCLR before enabling OSE losses')
        parser.add_argument('--ose_ramp_epoch', type=int, default=20,
                            help='epochs used to linearly ramp OSE loss weights after warmup')
        parser.add_argument('--ose_topk', type=int, default=8,
                            help='nearest memory neighbors per exemplar prototype')
        parser.add_argument('--ose_alpha', type=float, default=0.75,
                            help='discriminative neighbor score balance')
        parser.add_argument('--ose_tau_s', type=float, default=0.04,
                            help='student temperature for prototype and mix losses')
        parser.add_argument('--ose_tau_t', type=float, default=0.1,
                            help='teacher temperature for prototype targets')
        parser.add_argument('--ose_lambda', type=float, default=1.0,
                            help='weight for exemplar-guided prototype loss')
        parser.add_argument('--ose_mu', type=float, default=1.0,
                            help='weight for exemplar-guided mixup consistency loss')
        parser.add_argument('--ose_mix_alpha', type=float, default=1.0,
                            help='Beta distribution parameter for skeleton input mixup')

        return parser
