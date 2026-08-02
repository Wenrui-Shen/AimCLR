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


if __name__ == '__main__':
    unittest.main()

