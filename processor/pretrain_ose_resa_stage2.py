import argparse
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torchlight import str2bool

from .pretrain_ose_resa import OSEResAProcessor


_AIMCLR_PROJECTOR_MAP = OrderedDict([
    ('encoder_q.fc.0.weight', 'projector_q.0.weight'),
    ('encoder_q.fc.0.bias', 'projector_q.0.bias'),
    ('encoder_q.fc.2.weight', 'projector_q.2.weight'),
    ('encoder_q.fc.2.bias', 'projector_q.2.bias'),
])


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
        self._stage2_resumed = False

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
        return parser

