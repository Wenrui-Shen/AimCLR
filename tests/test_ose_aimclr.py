import types
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ose_aimclr import OSEAimCLR
from processor.pretrain_ose_aimclr import OSEAimCLR_Processor


class TinyAimEncoder(nn.Module):
    """Small encoder exposing AimCLR's feature and drop interfaces."""

    def __init__(self, in_channels=3, hidden_dim=8, num_class=5, **kwargs):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.forward_feature_calls = 0

    def forward_features(self, x, drop=False):
        self.forward_feature_calls += 1
        features = self.input_layer(x.mean(dim=(2, 3, 4)))
        if drop:
            return features, features * 0.5
        return features

    def forward(self, x, drop=False):
        features = self.forward_features(x, drop=drop)
        if drop:
            return self.fc(features[0]), self.fc(features[1])
        return self.fc(features)


class TinyTripletDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.label = [0] * 4 + [1] * 4 + [2] * 4
        self.data = np.arange(
            12 * 3 * 2 * 2, dtype=np.float32).reshape(12, 3, 2, 2, 1)
        self.return_index = True

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample = np.array(self.data[index])
        return [sample, sample, sample], self.label[index], index


class CaptureIO:

    def __init__(self):
        self.messages = []

    def print_log(self, message):
        self.messages.append(message)


class OSEAimCLRA2Test(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        self.batch_size = 4
        self.num_classes = 3
        shape = (self.batch_size, 3, 4, 5, 1)
        self.extreme = torch.randn(*shape)
        self.query = torch.randn(*shape)
        self.key = torch.randn(*shape)
        self.exemplar = torch.randn(
            self.num_classes, 3, 4, 5, 1)
        self.sample_indices = torch.arange(self.batch_size)

    def _model(self, ose_enabled=True):
        return OSEAimCLR(
            base_encoder='tests.test_ose_aimclr.TinyAimEncoder',
            pretrain=True,
            feature_dim=5,
            queue_size=8,
            momentum=0.9,
            Temperature=0.2,
            mlp=True,
            hidden_dim=8,
            num_class=self.num_classes,
            dropout=0.0,
            ose_enabled=ose_enabled)

    def _mixed_inputs(self):
        permutation = torch.tensor([2, 0, 3, 1])
        beta = 0.35
        mixed_view = (
            beta * self.query +
            (1.0 - beta) * self.key[permutation])
        return mixed_view, permutation, beta

    @staticmethod
    def _weak_loss(logits, target):
        if target.dim() == 1:
            return F.cross_entropy(logits, target)
        return -(
            F.log_softmax(logits, dim=1) * target
        ).sum(dim=1).div(target.sum(dim=1)).mean()

    def test_aimclr_only_path_remains_cpu_safe(self):
        model = self._model(ose_enabled=False)
        outputs = model(self.extreme, self.query, self.key)

        self.assertIsNone(outputs[-1])
        self.assertEqual(outputs[1].dim(), 1)
        self.assertEqual(outputs[1].device.type, 'cpu')
        self.assertEqual(model.queue_filled.item(), self.batch_size)
        model.update_ptr(self.batch_size)
        self.assertEqual(model.queue_ptr.item(), self.batch_size)

    def test_a2_uses_one_native_queue_and_old_entries_only(self):
        model = self._model()
        model.train()
        self.assertFalse(hasattr(model, 'ose_queue'))
        self.assertFalse(hasattr(model, 'ose_projector_q'))

        first_indices = torch.arange(10, 10 + self.batch_size)
        first_outputs = model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            compute_ose=True,
            sample_indices=first_indices,
            ose_topk=4)
        first = first_outputs[-1]

        self.assertEqual(
            tuple(first['neighbor_sample_indices'].shape),
            (self.num_classes, 0))
        self.assertTrue(torch.all(
            first['prototype_component_counts'] == 1))
        self.assertEqual(first_outputs[1].dim(), 1)
        self.assertEqual(first['nnm_positive_rate'].item(), 0.0)
        self.assertEqual(model.queue_filled.item(), self.batch_size)
        self.assertTrue(torch.equal(
            model.queue_sample_indices[:self.batch_size], first_indices))

        model.update_ptr(self.batch_size)
        second_indices = torch.arange(20, 20 + self.batch_size)
        second_outputs = model(
            self.extreme, self.query, self.key,
            nnm=True,
            exemplar=self.exemplar,
            compute_ose=True,
            sample_indices=second_indices,
            ose_topk=4)
        second = second_outputs[-1]
        valid = second['neighbor_queue_indices'] >= 0
        selected_slots = second['neighbor_queue_indices'][valid]

        self.assertEqual(second_outputs[1].dim(), 2)
        self.assertEqual(
            tuple(second_outputs[1].shape),
            (self.batch_size, 1 + model.K))
        queue_positive_count = second_outputs[1][:, 1:].sum(dim=1)
        self.assertTrue(torch.all(queue_positive_count <= 1))
        self.assertTrue(torch.all(second_outputs[1][:, 0] == 1))
        self.assertAlmostEqual(
            second['nnm_positive_rate'].item(),
            (queue_positive_count > 0).float().mean().item())
        self.assertEqual(selected_slots.numel(), self.batch_size)
        self.assertEqual(
            torch.unique(selected_slots).numel(), selected_slots.numel())
        self.assertTrue(torch.all(selected_slots < self.batch_size))
        self.assertEqual(second['neighbor_overlap_rate'].item(), 0.0)
        self.assertEqual(model.queue_filled.item(), model.K)

    def test_mutually_exclusive_q4_never_reuses_a_queue_slot(self):
        model = self._model()
        model.queue.copy_(F.normalize(torch.randn_like(model.queue), dim=0))
        model.queue_filled[0] = model.K
        model.queue_sample_indices.copy_(torch.arange(model.K))
        exemplar_z = F.normalize(
            torch.randn(self.num_classes, model.queue.size(0)), dim=1)

        state = model._class_prototypes(
            exemplar_z, topk=4, alpha=0.75)
        valid = state['neighbor_valid']
        selected = state['neighbor_queue_indices'][valid]

        self.assertEqual(torch.unique(selected).numel(), selected.numel())
        self.assertTrue(torch.all(
            valid.sum(dim=1) <= 4))
        self.assertTrue(torch.all(
            state['component_counts'] <= 5))
        self.assertEqual(state['overlap_rate'].item(), 0.0)

    def test_constrained_nnm_uses_one_best_valid_candidate(self):
        model = self._model()
        model.queue_sample_indices.copy_(torch.tensor(
            [10, 11, 20, 31, 40, 41, 42, 43]))
        teacher_target = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        prototype_state = {
            'neighbor_queue_indices': torch.tensor([
                [0, 1, -1, -1],
                [2, -1, -1, -1],
                [-1, -1, -1, -1],
            ]),
            'neighbor_valid': torch.tensor([
                [True, True, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]),
        }
        normal_similarity = torch.zeros(3, model.K)
        extreme_similarity = torch.zeros_like(normal_similarity)
        dropped_similarity = torch.zeros_like(normal_similarity)
        normal_similarity[0, 0] = 0.99
        normal_similarity[0, 1] = 0.20
        extreme_similarity[0, 1] = 0.90

        mask, state = model._ose_constrained_nnm_mask(
            teacher_target, prototype_state,
            normal_similarity, extreme_similarity, dropped_similarity,
            sample_indices=torch.tensor([10, 20, 30]))

        # Row 0 cannot reuse slot 0 (same sample), so the extreme stream's
        # slot 1 wins.  Row 1 loses its only same-sample candidate and row 2
        # has no P1 candidates.
        self.assertEqual(mask[0, 1].item(), 1.0)
        self.assertEqual(mask.sum().item(), 1.0)
        self.assertTrue(torch.equal(
            state['nnm_selected_queue_indices'],
            torch.tensor([1, -1, -1])))
        self.assertTrue(torch.equal(
            state['nnm_selected_sample_indices'],
            torch.tensor([11, -1, -1])))
        self.assertAlmostEqual(state['nnm_positive_rate'].item(), 1.0 / 3.0)
        self.assertAlmostEqual(state['nnm_candidate_count'].item(), 1.0 / 3.0)
        self.assertEqual(state['nnm_same_sample_filtered'].item(), 2.0)

    def test_proto_mix_and_constrained_nnm_share_native_head_gradients(self):
        model = self._model()
        model.train()
        # Seed the old shared queue with valid EMA-like entries.
        model.queue.copy_(F.normalize(torch.randn_like(model.queue), dim=0))
        model.queue_filled[0] = model.K
        model.queue_sample_indices.copy_(torch.arange(model.K))
        mixed_view, permutation, beta = self._mixed_inputs()

        outputs = model(
            self.extreme, self.query, self.key,
            nnm=True,
            exemplar=self.exemplar,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_ose=True,
            compute_mix_proto=True,
            compute_mix_ins=True,
            sample_indices=self.sample_indices,
            ose_topk=4)
        losses = outputs[-1]

        loss2 = -torch.mean(torch.sum(
            torch.log(outputs[2].clamp_min(1e-12)) * outputs[4], dim=1))
        loss3 = -torch.mean(torch.sum(
            torch.log(outputs[3].clamp_min(1e-12)) * outputs[4], dim=1))
        total = (
            self._weak_loss(outputs[0], outputs[1]) +
            (loss2 + loss3) / 2.0 +
            losses['proto'] + losses['mix_proto'] + losses['mix_ins'])
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertGreater(losses['mix_proto'].item(), 0.0)
        self.assertGreater(losses['mix_ins'].item(), 0.0)
        self.assertIsNotNone(model.encoder_q.input_layer.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.encoder_q.fc.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder_k.parameters()))

    def test_mix_terms_remain_independently_switchable(self):
        mixed_view, permutation, beta = self._mixed_inputs()
        proto_model = self._model()
        proto_losses = proto_model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_ose=True,
            compute_mix_proto=True,
            sample_indices=self.sample_indices,
            ose_topk=0)[-1]
        self.assertGreater(proto_losses['mix_proto'].item(), 0.0)
        self.assertEqual(proto_losses['mix_ins'].item(), 0.0)

        instance_model = self._model()
        instance_losses = instance_model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_ose=True,
            compute_mix_ins=True,
            sample_indices=self.sample_indices,
            ose_topk=0)[-1]
        self.assertEqual(instance_losses['mix_proto'].item(), 0.0)
        self.assertGreater(instance_losses['mix_ins'].item(), 0.0)

    def test_exemplars_are_removed_from_unlabeled_sampler(self):
        dataset = TinyTripletDataset()
        processor = OSEAimCLR_Processor.__new__(OSEAimCLR_Processor)
        processor.arg = types.SimpleNamespace(
            ose_num_class=3,
            ose_exemplar_seed=0,
            ose_exemplar_index_path='')
        processor.io = CaptureIO()
        processor.data_loader = {
            'train': torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=True, drop_last=True,
                num_workers=0)
        }

        processor._select_exemplars()
        processor._exclude_exemplars_from_unlabeled_loader()
        loader = processor.data_loader['train']
        sampled = set(loader.sampler.indices)
        excluded = set(processor.ose_exemplar_indices)

        self.assertTrue(excluded)
        self.assertTrue(sampled.isdisjoint(excluded))
        self.assertEqual(len(sampled), len(dataset) - self.num_classes)

    def test_ose_and_nnm_share_one_activation_epoch(self):
        processor = OSEAimCLR_Processor.__new__(OSEAimCLR_Processor)
        processor.arg = types.SimpleNamespace(
            ose_enabled=True, mining_epoch=150)
        self.assertFalse(processor._ose_active(150))
        self.assertTrue(processor._ose_active(151))


if __name__ == '__main__':
    unittest.main()
