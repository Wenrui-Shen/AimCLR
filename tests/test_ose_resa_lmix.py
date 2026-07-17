import unittest

import torch
import torch.nn as nn

from net.ose_resa import OSEResA


class TinyEncoder(nn.Module):
    """Small encoder exposing the interface required by OSEResA."""

    def __init__(self, in_channels=3, hidden_dim=8, num_class=3, **kwargs):
        super().__init__()
        self.input_layer = nn.Linear(in_channels, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.forward_feature_calls = 0

    def forward_features(self, x):
        self.forward_feature_calls += 1
        pooled = x.mean(dim=(2, 3, 4))
        return self.input_layer(pooled)

    def forward(self, x):
        return self.fc(self.forward_features(x))


class OSEResALmixTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 4
        self.num_classes = 3
        self.view_a = torch.randn(self.batch_size, 3, 4, 5, 1)
        self.view_b = torch.randn(self.batch_size, 3, 4, 5, 1)
        self.exemplar = torch.randn(self.num_classes, 3, 4, 5, 1)
        self.sample_indices = torch.arange(self.batch_size)

    def _model(self):
        return OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_hidden_dim=12,
            projector_layers=2,
            use_predictor=True,
            ose_enabled=True,
            queue_size=8,
            hidden_dim=8,
            num_class=self.num_classes,
            dropout=0.0)

    def test_disabled_mix_keeps_baseline_path(self):
        model = self._model()
        model.train()
        losses = model(
            self.view_a, self.view_b, self.exemplar,
            sample_indices=self.sample_indices)

        self.assertEqual(losses['mix_proto'].item(), 0.0)
        self.assertEqual(losses['mix_ins'].item(), 0.0)
        self.assertEqual(model.encoder_q.forward_feature_calls, 3)
        self.assertEqual(model.encoder_k.forward_feature_calls, 2)
        self.assertEqual(model.queue_ptr.item(), self.batch_size)
        self.assertEqual(model.queue_filled.item(), self.batch_size)

    def test_full_mix_uses_projector_only_and_does_not_enter_queue(self):
        model = self._model()
        model.train()
        permutation = torch.tensor([2, 0, 3, 1])
        beta = 0.35
        mixed_view = (
            beta * self.view_b +
            (1.0 - beta) * self.view_a[permutation])

        losses = model(
            self.view_a, self.view_b, self.exemplar,
            sample_indices=self.sample_indices,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_mix_proto=True,
            compute_mix_ins=True)

        self.assertTrue(torch.isfinite(losses['mix_proto']))
        self.assertTrue(torch.isfinite(losses['mix_ins']))
        self.assertGreater(losses['mix_proto'].item(), 0.0)
        self.assertGreater(losses['mix_ins'].item(), 0.0)
        self.assertTrue(torch.allclose(
            losses['mix'],
            losses['mix_proto'] + losses['mix_ins']))
        self.assertEqual(model.encoder_q.forward_feature_calls, 4)
        self.assertEqual(model.encoder_k.forward_feature_calls, 2)
        self.assertEqual(model.queue_ptr.item(), self.batch_size)
        self.assertEqual(model.queue_filled.item(), self.batch_size)

        (losses['mix_proto'] + losses['mix_ins']).backward()
        self.assertIsNotNone(model.encoder_q.input_layer.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.projector_q.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.predictor.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder_k.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.projector_k.parameters()))

    def test_mix_terms_can_be_enabled_independently(self):
        permutation = torch.tensor([1, 3, 0, 2])
        beta = 0.6
        mixed_view = (
            beta * self.view_b +
            (1.0 - beta) * self.view_a[permutation])

        proto_model = self._model()
        proto_losses = proto_model(
            self.view_a, self.view_b, self.exemplar,
            sample_indices=self.sample_indices,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_mix_proto=True)
        self.assertGreater(proto_losses['mix_proto'].item(), 0.0)
        self.assertEqual(proto_losses['mix_ins'].item(), 0.0)

        instance_model = self._model()
        instance_losses = instance_model(
            self.view_a, self.view_b, self.exemplar,
            sample_indices=self.sample_indices,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_mix_ins=True)
        self.assertEqual(instance_losses['mix_proto'].item(), 0.0)
        self.assertGreater(instance_losses['mix_ins'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
