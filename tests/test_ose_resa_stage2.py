import types
import unittest

import numpy as np
import torch

from net.aimclr import AimCLR
from net.ose_resa import OSEResA
from processor.pretrain_ose_resa_stage2 import (
    OSEResAStage2Processor,
    transfer_aimclr_stage1,
)


class _MemoryDataset(object):

    def __init__(self, count=6):
        rng = np.random.RandomState(3)
        self.data = rng.randn(count, 3, 4, 5, 1).astype(np.float32)
        self.label = list(range(count))

    def __len__(self):
        return len(self.label)


class _ExemplarDataset(object):

    def __init__(self, count=3):
        values = np.arange(
            count * 3 * 4 * 25, dtype=np.float32)
        self.data = values.reshape(count, 3, 4, 25, 1)
        self.label = list(range(count))
        self.augmentation_calls = 0

    def __len__(self):
        return len(self.label)

    def _aug(self, sample):
        self.augmentation_calls += 1
        return sample + 1.0


class _LogSink(object):

    def __init__(self):
        self.messages = []

    def print_log(self, message):
        self.messages.append(message)


class OSEResAStage2Test(unittest.TestCase):

    def _aimclr(self):
        return AimCLR(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            queue_size=8,
            mlp=True,
            in_channels=3,
            hidden_dim=8,
            num_class=3)

    def _stage2(self, queue_size=8):
        return OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_type='aimclr',
            use_predictor=False,
            ose_enabled=True,
            queue_size=queue_size,
            hidden_dim=8,
            num_class=3,
            dropout=0.0)

    def test_transfer_reuses_online_backbone_and_aimclr_head(self):
        torch.manual_seed(11)
        source = self._aimclr()
        with torch.no_grad():
            source.encoder_q.input_layer.weight.fill_(0.25)
            source.encoder_q.fc[0].weight.fill_(0.5)
            source.encoder_q.fc[2].weight.fill_(0.75)
            source.encoder_k.input_layer.weight.fill_(-1.0)

        target = self._stage2()
        report = transfer_aimclr_stage1(
            target, source.state_dict(), load_projector=True)

        self.assertTrue(torch.equal(
            target.encoder_q.input_layer.weight,
            source.encoder_q.input_layer.weight))
        self.assertTrue(torch.equal(
            target.encoder_k.input_layer.weight,
            source.encoder_q.input_layer.weight))
        self.assertTrue(torch.equal(
            target.projector_q[0].weight,
            source.encoder_q.fc[0].weight))
        self.assertTrue(torch.equal(
            target.projector_q[2].weight,
            source.encoder_q.fc[2].weight))
        self.assertTrue(torch.equal(
            target.projector_k[2].weight,
            target.projector_q[2].weight))
        self.assertEqual(report['projector_tensors'], 4)
        self.assertEqual(target.queue_filled.item(), 0)

    def test_transfer_rejects_incompatible_resa_projector(self):
        source = self._aimclr()
        target = OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_hidden_dim=12,
            projector_layers=2,
            projector_type='resa',
            hidden_dim=8,
            num_class=3)
        with self.assertRaisesRegex(ValueError, 'projector_type'):
            transfer_aimclr_stage1(
                target, source.state_dict(), load_projector=True)

    def test_prefill_populates_only_non_exemplar_slots(self):
        processor = OSEResAStage2Processor.__new__(
            OSEResAStage2Processor)
        processor.model = self._stage2(queue_size=4)
        processor.dev = 'cpu'
        processor.arg = types.SimpleNamespace(
            stage2_prefill_seed=5,
            stage2_prefill_batch_size=2,
            stage2_prefill_augmented=False,
            stream='joint')
        processor.ose_exemplar_indices = [1]
        processor.data_loader = {
            'train': types.SimpleNamespace(dataset=_MemoryDataset())}
        processor.io = _LogSink()
        processor.model.train()

        processor._prefill_ose_queue()

        self.assertEqual(processor.model.queue_filled.item(), 4)
        self.assertEqual(processor.model.queue_ptr.item(), 0)
        filled_indices = processor.model.queue_sample_indices[:4]
        self.assertNotIn(1, filled_indices.tolist())
        self.assertEqual(torch.unique(filled_indices).numel(), 4)
        self.assertTrue(torch.allclose(
            processor.model.queue[:, :4].norm(dim=0),
            torch.ones(4), atol=1e-6))
        self.assertTrue(processor.model.training)

    def test_native_stage2_uses_separate_backbone_and_head_lrs(self):
        processor = OSEResAStage2Processor.__new__(
            OSEResAStage2Processor)
        processor.model = OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_hidden_dim=12,
            projector_layers=2,
            projector_type='resa',
            use_predictor=True,
            hidden_dim=8,
            num_class=3)
        for parameter in processor.model.encoder_q.fc.parameters():
            parameter.requires_grad = False
        processor.arg = types.SimpleNamespace(
            optimizer='SGD', base_lr=0.01, stage2_head_lr=0.25,
            resa_final_lr=0.0, stage2_head_final_lr=0.0,
            resa_warmup_epoch=0, num_epoch=100,
            nesterov=False, weight_decay=1e-5)
        processor.io = _LogSink()
        processor.iter_info = {}

        processor.load_optimizer()
        processor._set_learning_rate(progress=50.0)

        groups = {
            group['stage2_role']: group
            for group in processor.optimizer.param_groups
        }
        self.assertAlmostEqual(groups['backbone']['lr'], 0.005)
        self.assertAlmostEqual(groups['head']['lr'], 0.125)
        self.assertEqual(processor.iter_info['head_lr'], '0.125000')

    def test_exemplar_modalities_share_one_augmentation_draw(self):
        processor = OSEResAStage2Processor.__new__(
            OSEResAStage2Processor)
        dataset = _ExemplarDataset()
        processor.data_loader = {
            'train': types.SimpleNamespace(dataset=dataset)}
        processor.ose_exemplar_indices = [0, 1, 2]
        processor.ose_exemplar_modalities = ('joint', 'motion', 'bone')
        processor.arg = types.SimpleNamespace(
            stream='joint', ose_exemplar_views=1)
        processor.dev = 'cpu'

        joint, motion, bone = processor._exemplar_batches()

        self.assertEqual(dataset.augmentation_calls, 3)
        self.assertTrue(torch.allclose(
            motion[:, :, :-1], joint[:, :, 1:] - joint[:, :, :-1]))
        self.assertTrue(torch.count_nonzero(motion[:, :, -1]).item() == 0)
        self.assertTrue(torch.allclose(
            bone[:, :, :, 0], joint[:, :, :, 0] - joint[:, :, :, 1]))

    def test_multi_augmentation_jmb_groups_each_share_one_draw(self):
        processor = OSEResAStage2Processor.__new__(
            OSEResAStage2Processor)
        dataset = _ExemplarDataset()
        processor.data_loader = {
            'train': types.SimpleNamespace(dataset=dataset)}
        processor.ose_exemplar_indices = [0, 1, 2]
        processor.ose_exemplar_modalities = ('joint', 'motion', 'bone')
        processor.arg = types.SimpleNamespace(
            stream='joint', ose_exemplar_views=2)
        processor.dev = 'cpu'

        groups = processor._exemplar_view_groups()

        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 3 for group in groups))
        self.assertEqual(dataset.augmentation_calls, 6)
        for joint, motion, bone in groups:
            self.assertTrue(torch.allclose(
                motion[:, :, :-1], joint[:, :, 1:] - joint[:, :, :-1]))
            self.assertTrue(torch.count_nonzero(
                motion[:, :, -1]).item() == 0)
            self.assertTrue(torch.allclose(
                bone[:, :, :, 0],
                joint[:, :, :, 0] - joint[:, :, :, 1]))

    def test_dual_projector_gradient_diagnostic_localizes_conflict(self):
        processor = OSEResAStage2Processor.__new__(
            OSEResAStage2Processor)
        processor.model = OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_hidden_dim=12,
            projector_layers=2,
            projector_type='resa',
            use_predictor=True,
            ose_enabled=True,
            ose_separate_projector=True,
            queue_size=8,
            hidden_dim=8,
            num_class=3,
            dropout=0.0)
        processor.arg = types.SimpleNamespace(
            stage2_diagnostics=True,
            stage2_diagnostic_gradients=True)
        view_a = torch.randn(4, 3, 4, 5, 1)
        view_b = torch.randn(4, 3, 4, 5, 1)
        exemplar = torch.randn(3, 3, 4, 5, 1)
        losses = processor.model(
            view_a, view_b, exemplar, ose_topk=0,
            sample_indices=torch.arange(4))

        diagnostic = processor._diagnostics_before_backward(
            0, losses['cluster'], losses['proto'])
        self.assertTrue(np.isfinite(diagnostic['encoder_grad_cos']))
        self.assertTrue(np.isnan(
            diagnostic['shared_projector_grad_cos']))
        self.assertGreater(
            diagnostic['resa_projector_grad_norm'], 0.0)
        self.assertGreater(
            diagnostic['ose_projector_grad_norm'], 0.0)

        processor.model.zero_grad()
        (losses['cluster'] + losses['proto']).backward()
        actual = processor._diagnostics_after_backward(0)
        self.assertGreater(actual['actual_encoder_grad_norm'], 0.0)
        self.assertGreater(
            actual['actual_resa_projector_grad_norm'], 0.0)
        self.assertGreater(
            actual['actual_ose_projector_grad_norm'], 0.0)


if __name__ == '__main__':
    unittest.main()
