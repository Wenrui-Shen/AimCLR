import argparse
import csv
import math
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torchlight import str2bool

from .pretrain_ose_resa import OSEResAProcessor


_AIMCLR_PROJECTOR_MAP = OrderedDict([
    ('encoder_q.fc.0.weight', 'projector_q.0.weight'),
    ('encoder_q.fc.0.bias', 'projector_q.0.bias'),
    ('encoder_q.fc.2.weight', 'projector_q.2.weight'),
    ('encoder_q.fc.2.bias', 'projector_q.2.bias'),
])


_STAGE2_MEAN_METRICS = (
    'loss', 'cluster', 'cluster_entropy', 'cluster_kl', 'proto', 'align',
    'disp', 'align_kl', 'mix_proto', 'mix_ins', 'target_entropy',
    'encoder_feature_std', 'resa_projector_feature_std',
    'ose_projector_feature_std', 'encoder_offdiag_cos',
    'resa_projector_offdiag_cos', 'ose_projector_offdiag_cos',
    'encoder_resa_relation_cos', 'encoder_ose_relation_cos',
    'resa_ose_relation_cos', 'relation_target_pred_cos',
    'relation_top1_agreement',
)

_STAGE2_GRADIENT_METRICS = (
    'resa_encoder_grad_norm', 'ose_encoder_grad_norm',
    'encoder_grad_cos', 'resa_projector_grad_norm',
    'ose_projector_grad_norm', 'shared_projector_grad_cos',
    'resa_predictor_grad_norm', 'actual_encoder_grad_norm',
    'actual_resa_projector_grad_norm',
    'actual_ose_projector_grad_norm', 'actual_predictor_grad_norm',
)

_STAGE2_DIAGNOSTIC_FIELDS = (
    'epoch', 'training_mode', 'ose_enabled', 'separate_projector',
    'resa_weight', 'batches', 'backbone_lr', 'head_lr',
) + tuple('mean_' + name for name in _STAGE2_MEAN_METRICS) + tuple(
    'first_batch_' + name for name in _STAGE2_GRADIENT_METRICS) + (
    'encoder_param_drift', 'resa_projector_param_drift',
    'ose_projector_param_drift', 'predictor_param_drift',
)


def _unwrap_state_dict(checkpoint):
    """Return a plain, non-DataParallel state dict from common checkpoints."""
    if not isinstance(checkpoint, dict):
        raise ValueError('Stage1 checkpoint must contain a state dict')
    for key in ('state_dict', 'model_state_dict', 'model'):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    state_dict = OrderedDict()
    for name, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        if name.startswith('module.'):
            name = name[len('module.'):]
        state_dict[name] = value.detach().cpu()
    if not state_dict:
        raise ValueError('Stage1 checkpoint does not contain tensor weights')
    return state_dict


def _copy_checked(target_state, target_name, source_state, source_name):
    if source_name not in source_state:
        raise ValueError(
            'Stage1 checkpoint is missing required weight {}'.format(
                source_name))
    if target_name not in target_state:
        raise ValueError(
            'Stage2 model is missing required weight {}'.format(target_name))
    source = source_state[source_name]
    target = target_state[target_name]
    if source.shape != target.shape:
        raise ValueError(
            'Cannot transfer {} {} to {} {}'.format(
                source_name, tuple(source.shape),
                target_name, tuple(target.shape)))
    target_state[target_name] = source.to(dtype=target.dtype).clone()


def transfer_aimclr_stage1(model, checkpoint, load_projector=True):
    """Initialize ReSA/OSE from AimCLR without carrying its negative queue.

    The online AimCLR backbone initializes the Stage2 online branch.  The
    Stage2 EMA branch is then reset from that online branch, so Stage2 starts
    with exactly aligned online/teacher parameters.  With ``load_projector``,
    AimCLR's two-layer ``encoder_q.fc`` is moved into the exact-layout
    ``projector_q`` used jointly by ReSA and OSE.
    """
    source_state = _unwrap_state_dict(checkpoint)
    target_state = model.state_dict()

    backbone_targets = [
        name for name in target_state
        if name.startswith('encoder_q.') and
        not name.startswith('encoder_q.fc.')
    ]
    if not backbone_targets:
        raise ValueError('Stage2 model does not expose an encoder_q backbone')
    for target_name in backbone_targets:
        _copy_checked(
            target_state, target_name, source_state, target_name)

    projector_targets = []
    if load_projector:
        if getattr(model, 'projector_type', None) != 'aimclr':
            raise ValueError(
                'Loading the AimCLR head requires '
                'model_args.projector_type: aimclr')
        for source_name, target_name in _AIMCLR_PROJECTOR_MAP.items():
            _copy_checked(
                target_state, target_name, source_state, source_name)
            projector_targets.append(target_name)

    model.load_state_dict(target_state)
    # Do not import encoder_k or the AimCLR queue.  An identical ReSA teacher
    # at the transition boundary is less ambiguous than retaining a lagged
    # encoder trained under the old objective.
    model.reset_momentum_encoder()
    return {
        'backbone_tensors': len(backbone_targets),
        'projector_tensors': len(projector_targets),
        'source': 'encoder_q',
    }


class OSEResAStage2Processor(OSEResAProcessor):
    """Stage2 semantic consolidation initialized from an AimCLR checkpoint."""

    def load_model(self):
        super().load_model()
        if self.arg.stage2_prefill_batch_size <= 0:
            raise ValueError('stage2_prefill_batch_size must be positive')
        if self.arg.stage2_head_lr <= 0:
            raise ValueError('stage2_head_lr must be positive')
        if self.arg.stage2_head_final_lr < 0:
            raise ValueError('stage2_head_final_lr must be non-negative')
        if not self.arg.stage2_diagnostic_filename:
            raise ValueError('stage2_diagnostic_filename must not be empty')
        # The ST-GCN classifier is bypassed by forward_features throughout
        # pretraining. Keep it out of the Stage2 optimizer explicitly.
        for parameter in self.model.encoder_q.fc.parameters():
            parameter.requires_grad = False
        self._stage2_resumed = False
        self._stage2_diagnostic_reference = {}

    def load_optimizer(self):
        backbone_parameters = []
        head_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('module.'):
                name = name[len('module.'):]
            if name.startswith('encoder_q.'):
                backbone_parameters.append(parameter)
            else:
                head_parameters.append(parameter)
        if not backbone_parameters:
            raise ValueError('Stage2 optimizer found no online backbone')
        if not head_parameters:
            raise ValueError('Stage2 optimizer found no trainable head')

        parameter_groups = [
            {
                'params': backbone_parameters,
                'lr': self.arg.base_lr,
                'stage2_role': 'backbone',
            },
            {
                'params': head_parameters,
                'lr': self.arg.stage2_head_lr,
                'stage2_role': 'head',
            },
        ]
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameter_groups, lr=self.arg.base_lr, momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameter_groups, lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(
                'Unsupported optimizer: {}'.format(self.arg.optimizer))
        self.io.print_log(
            'Stage2 optimizer | backbone lr {:.6f} | head lr {:.6f}'.format(
                self.arg.base_lr, self.arg.stage2_head_lr))

    def _scheduled_lr(self, initial_lr, final_lr, progress):
        warmup = float(self.arg.resa_warmup_epoch)
        if warmup > 0 and progress <= warmup:
            return float(initial_lr) * progress / warmup
        decay_progress = (progress - warmup) / max(
            self.arg.num_epoch - warmup, 1.0)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return (float(final_lr) +
                (float(initial_lr) - float(final_lr)) * cosine)

    def _set_learning_rate(self, progress):
        backbone_lr = self._scheduled_lr(
            self.arg.base_lr, self.arg.resa_final_lr, progress)
        head_lr = self._scheduled_lr(
            self.arg.stage2_head_lr,
            self.arg.stage2_head_final_lr, progress)
        for param_group in self.optimizer.param_groups:
            role = param_group.get('stage2_role')
            if role == 'backbone':
                param_group['lr'] = backbone_lr
            elif role == 'head':
                param_group['lr'] = head_lr
            else:
                raise ValueError(
                    'Unknown Stage2 optimizer group: {}'.format(role))
        self.lr = backbone_lr
        self.head_lr = head_lr
        self.iter_info['head_lr'] = '{:.6f}'.format(head_lr)

    def train_log_writer(self, epoch):
        super().train_log_writer(epoch)
        self.train_writer.add_scalar(
            'head_lr', self.head_lr, self.global_step)

    @staticmethod
    def _snapshot_module(module, trainable_only=False):
        if module is None:
            return {}
        return {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in module.named_parameters()
            if not trainable_only or parameter.requires_grad
        }

    def _initialize_stage2_diagnostics(self):
        if not self.arg.stage2_diagnostics:
            return
        model = self._core_model(self.model)
        self._stage2_diagnostic_reference = {
            'encoder': self._snapshot_module(
                model.encoder_q, trainable_only=True),
            'resa_projector': self._snapshot_module(model.projector_q),
            'predictor': self._snapshot_module(model.predictor),
            'ose_projector': self._snapshot_module(
                model.ose_projector_q
                if model.ose_separate_projector else None),
        }
        self._stage2_diagnostic_path = os.path.join(
            self.arg.work_dir, self.arg.stage2_diagnostic_filename)
        fresh_run = not self.arg.weights and int(self.arg.start_epoch) == 0
        mode = 'w' if fresh_run else 'a'
        needs_header = (
            mode == 'w' or not os.path.isfile(self._stage2_diagnostic_path) or
            os.path.getsize(self._stage2_diagnostic_path) == 0)
        with open(self._stage2_diagnostic_path, mode, newline='') as handle:
            if needs_header:
                writer = csv.DictWriter(
                    handle, fieldnames=_STAGE2_DIAGNOSTIC_FIELDS)
                writer.writeheader()
        self.io.print_log(
            'Stage2 diagnostics | epoch CSV {}'.format(
                self._stage2_diagnostic_path))

    @staticmethod
    def _module_trainable_parameters(module):
        if module is None:
            return []
        return [
            parameter for parameter in module.parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _gradient_map(objective, parameters):
        if objective is None or not objective.requires_grad:
            return {}
        gradients = torch.autograd.grad(
            objective, parameters, retain_graph=True, allow_unused=True)
        result = {}
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                # Diagnostics run once per epoch. Moving these detached
                # gradients to CPU avoids retaining two full GPU gradient maps.
                result[id(parameter)] = gradient.detach().float().cpu()
        return result

    @staticmethod
    def _mapped_gradient_norm(gradient_map, parameters):
        squared = 0.0
        found = False
        for parameter in parameters:
            gradient = gradient_map.get(id(parameter))
            if gradient is None:
                continue
            squared += float(torch.sum(gradient * gradient).item())
            found = True
        return math.sqrt(squared) if found else 0.0

    @staticmethod
    def _mapped_gradient_cosine(left_map, right_map, parameters):
        dot = 0.0
        left_squared = 0.0
        right_squared = 0.0
        found = False
        for parameter in parameters:
            left = left_map.get(id(parameter))
            right = right_map.get(id(parameter))
            if left is None or right is None:
                continue
            dot += float(torch.sum(left * right).item())
            left_squared += float(torch.sum(left * left).item())
            right_squared += float(torch.sum(right * right).item())
            found = True
        if not found or left_squared <= 0.0 or right_squared <= 0.0:
            return float('nan')
        return dot / math.sqrt(left_squared * right_squared)

    @staticmethod
    def _actual_gradient_norm(parameters):
        squared = 0.0
        found = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            squared += float(torch.sum(gradient * gradient).item())
            found = True
        return math.sqrt(squared) if found else 0.0

    def _diagnostic_parameter_groups(self):
        model = self._core_model(self.model)
        encoder = self._module_trainable_parameters(model.encoder_q)
        resa_projector = self._module_trainable_parameters(model.projector_q)
        predictor = self._module_trainable_parameters(model.predictor)
        ose_projector = (
            self._module_trainable_parameters(model.ose_projector_q)
            if model.ose_separate_projector else resa_projector)
        all_parameters = []
        seen = set()
        for group in (encoder, resa_projector, predictor, ose_projector):
            for parameter in group:
                if id(parameter) not in seen:
                    all_parameters.append(parameter)
                    seen.add(id(parameter))
        return {
            'all': all_parameters,
            'encoder': encoder,
            'resa_projector': resa_projector,
            'ose_projector': ose_projector,
            'predictor': predictor,
            'separate': bool(model.ose_separate_projector),
        }

    def _diagnostics_epoch_start(self, epoch):
        if not self.arg.stage2_diagnostics:
            return
        self._stage2_diagnostic_sums = {
            name: 0.0 for name in _STAGE2_MEAN_METRICS}
        self._stage2_diagnostic_counts = {
            name: 0 for name in _STAGE2_MEAN_METRICS}
        self._stage2_gradient_diagnostics = {}
        self._stage2_diagnostic_batches = 0

    def _diagnostics_before_backward(self, batch_index, resa_objective,
                                     ose_objective):
        if (not self.arg.stage2_diagnostics or
                not self.arg.stage2_diagnostic_gradients or
                batch_index != 0):
            return {}
        groups = self._diagnostic_parameter_groups()
        resa_gradients = self._gradient_map(
            resa_objective, groups['all'])
        ose_gradients = self._gradient_map(
            ose_objective, groups['all'])
        diagnostics = {
            'resa_encoder_grad_norm': self._mapped_gradient_norm(
                resa_gradients, groups['encoder']),
            'ose_encoder_grad_norm': self._mapped_gradient_norm(
                ose_gradients, groups['encoder']),
            'encoder_grad_cos': self._mapped_gradient_cosine(
                resa_gradients, ose_gradients, groups['encoder']),
            'resa_projector_grad_norm': self._mapped_gradient_norm(
                resa_gradients, groups['resa_projector']),
            'ose_projector_grad_norm': self._mapped_gradient_norm(
                ose_gradients, groups['ose_projector']),
            'shared_projector_grad_cos': (
                self._mapped_gradient_cosine(
                    resa_gradients, ose_gradients,
                    groups['resa_projector'])
                if not groups['separate'] else float('nan')),
            'resa_predictor_grad_norm': self._mapped_gradient_norm(
                resa_gradients, groups['predictor']),
        }
        return diagnostics

    def _diagnostics_after_backward(self, batch_index):
        if (not self.arg.stage2_diagnostics or
                not self.arg.stage2_diagnostic_gradients or
                batch_index != 0):
            return {}
        groups = self._diagnostic_parameter_groups()
        return {
            'actual_encoder_grad_norm': self._actual_gradient_norm(
                groups['encoder']),
            'actual_resa_projector_grad_norm': self._actual_gradient_norm(
                groups['resa_projector']),
            'actual_ose_projector_grad_norm': self._actual_gradient_norm(
                groups['ose_projector']),
            'actual_predictor_grad_norm': self._actual_gradient_norm(
                groups['predictor']),
        }

    @staticmethod
    def _finite_scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = float(value.detach().item())
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
        return value if math.isfinite(value) else None

    def _diagnostics_record_batch(self, epoch, losses, total_loss,
                                  diagnostics):
        if not self.arg.stage2_diagnostics:
            return
        values = {'loss': total_loss}
        values.update(losses)
        for name in _STAGE2_MEAN_METRICS:
            value = self._finite_scalar(values.get(name))
            if value is None:
                continue
            self._stage2_diagnostic_sums[name] += value
            self._stage2_diagnostic_counts[name] += 1
        if diagnostics:
            self._stage2_gradient_diagnostics.update(diagnostics)
        self._stage2_diagnostic_batches += 1

    @staticmethod
    def _module_parameter_drift(module, reference):
        if module is None or not reference:
            return float('nan')
        numerator = 0.0
        denominator = 0.0
        for name, parameter in module.named_parameters():
            if name not in reference:
                continue
            current = parameter.detach().float().cpu()
            initial = reference[name]
            difference = current - initial
            numerator += float(torch.sum(difference * difference).item())
            denominator += float(torch.sum(initial * initial).item())
        if denominator <= 0.0:
            return float('nan')
        return math.sqrt(numerator / denominator)

    def _diagnostics_epoch_end(self, epoch):
        if not self.arg.stage2_diagnostics:
            return
        model = self._core_model(self.model)
        if not hasattr(self, '_stage2_diagnostic_path'):
            raise RuntimeError('Stage2 diagnostics were not initialized')
        if not self.arg.ose_enabled:
            training_mode = 'resa_only'
        elif self.arg.resa_weight == 0:
            training_mode = 'ose_only'
        elif model.ose_separate_projector:
            training_mode = 'resa_ose_separate_projector'
        else:
            training_mode = 'resa_ose_shared_projector'
        row = {
            'epoch': int(epoch),
            'training_mode': training_mode,
            'ose_enabled': int(bool(self.arg.ose_enabled)),
            'separate_projector': int(model.ose_separate_projector),
            'resa_weight': float(self.arg.resa_weight),
            'batches': int(self._stage2_diagnostic_batches),
            'backbone_lr': float(self.lr),
            'head_lr': float(self.head_lr),
        }
        for name in _STAGE2_MEAN_METRICS:
            count = self._stage2_diagnostic_counts[name]
            row['mean_' + name] = (
                self._stage2_diagnostic_sums[name] / count
                if count > 0 else float('nan'))
        for name in _STAGE2_GRADIENT_METRICS:
            row['first_batch_' + name] = (
                self._stage2_gradient_diagnostics.get(
                    name, float('nan')))
        reference = self._stage2_diagnostic_reference
        row['encoder_param_drift'] = self._module_parameter_drift(
            model.encoder_q, reference.get('encoder', {}))
        row['resa_projector_param_drift'] = self._module_parameter_drift(
            model.projector_q, reference.get('resa_projector', {}))
        row['predictor_param_drift'] = self._module_parameter_drift(
            model.predictor, reference.get('predictor', {}))
        row['ose_projector_param_drift'] = self._module_parameter_drift(
            model.ose_projector_q if model.ose_separate_projector else None,
            reference.get('ose_projector', {}))
        with open(self._stage2_diagnostic_path, 'a', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=_STAGE2_DIAGNOSTIC_FIELDS,
                extrasaction='ignore')
            writer.writerow(row)
        for name, value in row.items():
            if name.startswith('mean_') or name.endswith('_param_drift'):
                scalar = self._finite_scalar(value)
                if scalar is not None:
                    self.train_writer.add_scalar(
                        'stage2_diag/' + name, scalar, epoch)
        self.io.print_log(
            'Stage2 diagnostic epoch {} | cluster_kl {:.4f} | '
            'H-ReSA relation {:.4f} | encoder grad cosine {}'.format(
                epoch, row['mean_cluster_kl'],
                row['mean_encoder_resa_relation_cos'],
                '{:.4f}'.format(row['first_batch_encoder_grad_cos'])
                if math.isfinite(row['first_batch_encoder_grad_cos'])
                else 'n/a'))

    def load_weights(self):
        # ``weights`` remains available for resuming a full Stage2 model.  A
        # fresh Stage2 run instead uses ``stage1_weights`` and the explicit
        # cross-architecture transfer below.
        if self.arg.weights:
            self.model = self.io.load_weights(
                self.model, self.arg.weights, self.arg.ignore_weights)
            self._stage2_resumed = True
            self.io.print_log(
                'Stage2 resume | full ReSA/OSE checkpoint loaded; '
                'Stage1 transfer and queue prefill skipped')
            self._initialize_stage2_diagnostics()
            return

        path = self.arg.stage1_weights
        if not path:
            raise ValueError(
                'A fresh Stage2 run requires --stage1_weights')
        if not os.path.isfile(path):
            raise ValueError(
                'Stage1 AimCLR checkpoint does not exist: {}'.format(path))
        self.io.print_log(
            'Stage2 initialization | load AimCLR weights from {}'.format(
                path))
        checkpoint = torch.load(path, map_location='cpu')
        report = transfer_aimclr_stage1(
            self.model, checkpoint,
            load_projector=self.arg.stage2_load_projector)
        self.io.print_log(
            'Stage2 transfer | source {} | backbone tensors {} | '
            'projector tensors {} | AimCLR queue discarded'.format(
                report['source'], report['backbone_tensors'],
                report['projector_tensors']))
        self._initialize_stage2_diagnostics()

    def load_data(self):
        super().load_data()
        if (self.arg.ose_enabled and self.arg.stage2_prefill_queue and
                not self._stage2_resumed):
            self._prefill_ose_queue()

    @staticmethod
    def _core_model(model):
        return model.module if hasattr(model, 'module') else model

    @torch.no_grad()
    def _prefill_ose_queue(self):
        """Fill the OSE memory before the first Stage2 optimizer step."""
        model = self._core_model(self.model)
        if not hasattr(model, 'queue'):
            raise ValueError('Stage2 queue prefill requires OSE to be enabled')
        dataset = self.data_loader['train'].dataset
        if not hasattr(dataset, 'data'):
            raise ValueError('Stage2 queue prefill requires dataset.data')

        excluded = set(self.ose_exemplar_indices)
        candidates = np.asarray([
            index for index in range(len(dataset))
            if index not in excluded
        ], dtype=np.int64)
        if candidates.size == 0:
            raise ValueError('No samples are available for Stage2 queue prefill')
        rng = np.random.RandomState(self.arg.stage2_prefill_seed)
        rng.shuffle(candidates)
        selected = candidates[:min(model.queue_size, candidates.size)]

        model.queue.zero_()
        model.queue_ptr.zero_()
        model.queue_filled.zero_()
        model.queue_sample_indices.fill_(-1)

        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        random.seed(self.arg.stage2_prefill_seed)
        np.random.seed(self.arg.stage2_prefill_seed)
        was_training = model.training
        model.eval()
        try:
            batch_size = self.arg.stage2_prefill_batch_size
            for start in range(0, selected.size, batch_size):
                batch_indices = selected[start:start + batch_size]
                samples = []
                for index in batch_indices:
                    sample = np.array(dataset.data[int(index)])
                    if (self.arg.stage2_prefill_augmented and
                            hasattr(dataset, '_aug')):
                        sample = dataset._aug(sample)
                    samples.append(sample)
                data = torch.from_numpy(np.stack(samples)).float()
                data = data.to(self.dev, non_blocking=True)
                data = self._prepare_stream(data)
                projected = model.teacher_projection(data)
                sample_indices = torch.as_tensor(
                    batch_indices, dtype=torch.long, device=projected.device)
                model._dequeue_and_enqueue(projected, sample_indices)
        finally:
            if was_training:
                model.train()
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)

        view_name = ('weak-augmented' if self.arg.stage2_prefill_augmented
                     else 'clean')
        self.io.print_log(
            'Stage2 OSE queue prefill | {} / {} slots | {} view | '
            'seed {} | excluded {} exemplars'.format(
                int(model.queue_filled.item()), model.queue_size, view_name,
                self.arg.stage2_prefill_seed, len(excluded)))

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = OSEResAProcessor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help, parents=[parent_parser],
            description='AimCLR to ReSA+OSE Stage2 pretraining')
        parser.add_argument('--stage1_weights', default='')
        parser.add_argument(
            '--stage2_load_projector', type=str2bool, default=True)
        parser.add_argument(
            '--stage2_prefill_queue', type=str2bool, default=True)
        parser.add_argument(
            '--stage2_prefill_augmented', type=str2bool, default=True)
        parser.add_argument('--stage2_prefill_seed', type=int, default=0)
        parser.add_argument(
            '--stage2_prefill_batch_size', type=int, default=128)
        parser.add_argument('--stage2_head_lr', type=float, default=0.01)
        parser.add_argument(
            '--stage2_head_final_lr', type=float, default=0.0)
        parser.add_argument(
            '--stage2_diagnostics', type=str2bool, default=True,
            help='write one row of Stage2 transition diagnostics per epoch')
        parser.add_argument(
            '--stage2_diagnostic_gradients', type=str2bool, default=True,
            help='measure ReSA/OSE gradient norms and cosine on the first '
                 'batch of each epoch')
        parser.add_argument(
            '--stage2_diagnostic_filename', type=str,
            default='stage2_diagnostics.csv')
        return parser
