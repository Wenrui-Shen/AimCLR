import types
import unittest

import numpy as np
import torch
import torch.nn as nn

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

    @staticmethod
    def _aug(sample):
        return sample

    def __getitem__(self, index):
        sample = np.array(self.data[index])
        return [sample, sample, sample], self.label[index], index


class CaptureIO:

    def __init__(self):
        self.messages = []

    def print_log(self, message):
        self.messages.append(message)


class OSEAimCLRTest(unittest.TestCase):

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
        self.extra_exemplar_views = [
            torch.randn(self.num_classes, 3, 4, 5, 1)
            for _ in range(4)]
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
            ose_enabled=ose_enabled,
            ose_feature_dim=6,
            ose_projector_hidden_dim=12,
            ose_projector_layers=2,
            ose_queue_size=8)

    def _mixed_inputs(self):
        permutation = torch.tensor([2, 0, 3, 1])
        beta = 0.35
        mixed_view = (
            beta * self.query +
            (1.0 - beta) * self.key[permutation])
        return mixed_view, permutation, beta

    def test_aimclr_only_path_is_cpu_safe_and_skips_ose(self):
        model = self._model(ose_enabled=False)
        outputs = model(self.extreme, self.query, self.key)

        self.assertIsNone(outputs[-1])
        self.assertEqual(model.encoder_q.forward_feature_calls, 2)
        self.assertEqual(model.encoder_k.forward_feature_calls, 1)
        self.assertEqual(outputs[1].device.type, 'cpu')
        model.update_ptr(self.batch_size)
        self.assertEqual(model.queue_ptr.item(), self.batch_size)

    def test_mv4_uses_five_components_without_reading_or_writing_ose_queue(self):
        model = self._model()
        model.train()
        teacher_projector_bn = model.ose_projector_k[1]
        mixed_view, permutation, beta = self._mixed_inputs()

        outputs = model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_ose=True,
            compute_mix_proto=True,
            compute_mix_ins=True,
            extra_exemplar_views=self.extra_exemplar_views,
            sample_indices=self.sample_indices,
            ose_topk=0)
        losses = outputs[-1]

        self.assertTrue(torch.isfinite(losses['proto']))
        self.assertGreater(losses['mix_proto'].item(), 0.0)
        self.assertGreater(losses['mix_ins'].item(), 0.0)
        self.assertEqual(
            tuple(losses['neighbor_sample_indices'].shape),
            (self.num_classes, 0))
        self.assertEqual(losses['prototype_components'].item(), 5)
        self.assertEqual(model.ose_queue_filled.item(), 0)
        self.assertEqual(model.encoder_q.forward_feature_calls, 4)
        self.assertEqual(model.encoder_k.forward_feature_calls, 5)
        self.assertEqual(teacher_projector_bn.num_batches_tracked.item(), 1)

        (losses['proto'] + losses['mix']).backward()
        self.assertIsNotNone(model.encoder_q.input_layer.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.ose_projector_q.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder_q.fc.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.encoder_k.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.ose_projector_k.parameters()))

    def test_q4_reads_only_preexisting_neighbors_and_tracks_sample_indices(self):
        model = self._model()
        model.train()
        first_indices = torch.arange(10, 10 + self.batch_size)
        first = model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            compute_ose=True,
            sample_indices=first_indices,
            ose_topk=4)[-1]

        self.assertEqual(
            tuple(first['neighbor_sample_indices'].shape),
            (self.num_classes, 0))
        self.assertEqual(first['prototype_components'].item(), 1)
        self.assertEqual(first['mix_proto'].item(), 0.0)
        self.assertEqual(first['mix_ins'].item(), 0.0)
        self.assertEqual(model.encoder_q.forward_feature_calls, 3)
        self.assertEqual(model.encoder_k.forward_feature_calls, 1)
        self.assertEqual(model.ose_queue_filled.item(), self.batch_size)

        second = model(
            self.extreme, self.query, self.key,
            exemplar=self.exemplar,
            compute_ose=True,
            sample_indices=torch.arange(20, 20 + self.batch_size),
            ose_topk=4)[-1]
        selected = second['neighbor_sample_indices']
        self.assertEqual(tuple(selected.shape), (self.num_classes, 4))
        self.assertEqual(second['prototype_components'].item(), 5)
        self.assertTrue(torch.all(selected >= 10))
        self.assertTrue(torch.all(selected < 10 + self.batch_size))
        self.assertEqual(model.ose_queue_filled.item(), 8)

    def test_mix_terms_are_independent(self):
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
        for class_id, index in zip(
                processor.ose_class_ids,
                processor.ose_exemplar_indices):
            self.assertEqual(dataset.label[index], class_id)


if __name__ == '__main__':
    unittest.main()
