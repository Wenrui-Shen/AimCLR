import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from feeder.ose_resa_feeder import Feeder
from net.ose_resa import OSEResA


class ReSAOSEViewAugmentationTest(unittest.TestCase):

    def _feeder(self, **kwargs):
        with mock.patch.object(Feeder, 'load_data'):
            return Feeder('data.npy', 'label.pkl', **kwargs)

    def test_default_pipeline_is_three_transforms_with_half_probability(self):
        feeder = self._feeder()
        self.assertEqual(
            feeder.augmentation_methods,
            ('temporal_crop', 'shear', 'rotation'))
        self.assertEqual(feeder.augmentation_probability, 0.5)

    def test_pipeline_visits_every_transform_and_triggers_independently(self):
        feeder = self._feeder()
        applied = []

        def track(data, name):
            applied.append(name)
            return data

        feeder._apply_augmentation = track
        sample = np.zeros((3, 4, 5, 1), dtype=np.float32)
        with mock.patch(
                'feeder.ose_resa_feeder.random.random',
                side_effect=[0.1, 0.8, 0.49]):
            feeder._aug(sample)

        self.assertEqual(applied, ['temporal_crop', 'rotation'])

    def test_two_views_use_independent_pipeline_draws(self):
        feeder = self._feeder(return_index=True)
        feeder.data = np.zeros((1, 3, 4, 5, 1), dtype=np.float32)
        feeder.label = [7]
        first = np.ones((3, 4, 5, 1), dtype=np.float32)
        second = np.full((3, 4, 5, 1), 2.0, dtype=np.float32)
        feeder._aug = mock.Mock(side_effect=[first, second])

        views, label, index = feeder[0]

        self.assertEqual(feeder._aug.call_count, 2)
        self.assertTrue(np.array_equal(views[0], first))
        self.assertTrue(np.array_equal(views[1], second))
        self.assertEqual(label, 7)
        self.assertEqual(index, 0)


class ReSAOSEPrototypeStageTest(unittest.TestCase):

    def _model(self, stage, feature_dim=3, queue_size=8):
        model = OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=feature_dim,
            projector_hidden_dim=12,
            projector_layers=2,
            use_predictor=True,
            ose_enabled=True,
            ose_prototype_stage=stage,
            queue_size=queue_size,
            hidden_dim=8,
            num_class=3,
            dropout=0.0)
        return model

    @staticmethod
    def _fill_queue(model, memory):
        memory = F.normalize(memory, dim=1)
        count = memory.size(0)
        model.queue[:, :count].copy_(memory.t())
        model.queue_filled[0] = count
        model.queue_sample_indices[:count] = torch.arange(count)
        return memory

    @staticmethod
    def _legacy_p0(exemplar, memory, topk, alpha):
        exemplar = F.normalize(exemplar, dim=1)
        similarity = exemplar @ memory.t()
        other_similarity = similarity.unsqueeze(0).expand(
            exemplar.size(0), -1, -1).clone()
        diagonal = torch.eye(exemplar.size(0), dtype=torch.bool)
        other_similarity[diagonal] = -float('inf')
        max_other = other_similarity.max(dim=1)[0]
        score = alpha * similarity - (1.0 - alpha) * max_other
        indices = torch.topk(score, k=topk, dim=1).indices
        prototypes = []
        for class_index in range(exemplar.size(0)):
            components = torch.cat([
                exemplar[class_index:class_index + 1],
                memory[indices[class_index]],
            ], dim=0)
            weights = torch.softmax(
                components @ exemplar[class_index], dim=0)
            prototypes.append((weights.unsqueeze(1) * components).sum(dim=0))
        return torch.stack(prototypes), indices

    def setUp(self):
        self.exemplar = torch.eye(3)
        self.memory = torch.tensor([
            [1.0, 0.1, 0.0],
            [0.9, 0.2, 0.0],
            [0.8, 0.3, 0.0],
            [0.7, 0.4, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.2, 1.0],
        ])
        self.alpha = 0.75

    def test_p0_numerically_matches_legacy_q4_aggregation(self):
        model = self._model(stage=0)
        memory = self._fill_queue(model, self.memory)
        expected, expected_indices = self._legacy_p0(
            self.exemplar, memory, topk=4, alpha=self.alpha)

        prototypes, selected, counts, _ = model._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha)

        self.assertTrue(torch.allclose(prototypes, expected))
        self.assertTrue(torch.equal(selected, expected_indices))
        self.assertTrue(torch.equal(counts, torch.full((3,), 5)))

    def test_p1_assigns_each_queue_slot_to_at_most_one_class(self):
        model = self._model(stage=1)
        self._fill_queue(model, self.memory)

        _, selected, counts, overlap = model._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha)

        valid = selected[selected >= 0]
        self.assertEqual(valid.numel(), torch.unique(valid).numel())
        self.assertTrue(torch.all(counts >= 1))
        self.assertTrue(torch.all(counts <= 5))
        self.assertEqual(overlap.item(), 0.0)
        self.assertTrue(torch.any(selected < 0))

    def test_multimodal_label_prototype_is_used_without_neighbors(self):
        model = self._model(stage=1)
        extra = torch.tensor([
            [[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]],
            [[0.6, 0.8, 0.0], [0.0, 0.8, 0.6]],
            [[0.6, 0.0, 0.8], [0.0, 0.6, 0.8]],
        ])
        expected = model._fuse_labeled_exemplars(
            self.exemplar, extra)

        prototypes, selected, counts, overlap = model._class_prototypes(
            self.exemplar, topk=0, alpha=self.alpha,
            extra_exemplar_z=extra)

        self.assertTrue(torch.allclose(prototypes, expected))
        self.assertEqual(tuple(selected.shape), (3, 0))
        self.assertTrue(torch.equal(counts, torch.ones(3, dtype=torch.long)))
        self.assertEqual(overlap.item(), 0.0)
        self.assertTrue(torch.allclose(
            prototypes.norm(dim=1), torch.ones(3), atol=1e-6))

    def test_multimodal_label_prototype_guides_neighbor_selection(self):
        model = self._model(stage=1)
        self._fill_queue(model, self.memory)
        extra = torch.tensor([
            [[0.8, 0.6, 0.0], [0.8, 0.0, 0.6]],
            [[0.6, 0.8, 0.0], [0.0, 0.8, 0.6]],
            [[0.6, 0.0, 0.8], [0.0, 0.6, 0.8]],
        ])
        fused = model._fuse_labeled_exemplars(self.exemplar, extra)
        memory = F.normalize(self.memory, dim=1)
        similarity = fused @ memory.t()
        other_similarity = similarity.unsqueeze(0).expand(
            fused.size(0), -1, -1).clone()
        diagonal = torch.eye(fused.size(0), dtype=torch.bool)
        other_similarity[diagonal] = -float('inf')
        score = (
            self.alpha * similarity -
            (1.0 - self.alpha) * other_similarity.max(dim=1)[0])
        owner = score.argmax(dim=0)

        _, selected, _, _ = model._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha,
            extra_exemplar_z=extra)

        for class_index in range(fused.size(0)):
            valid = selected[class_index][selected[class_index] >= 0]
            if valid.numel() > 0:
                self.assertTrue(torch.all(owner[valid] == class_index))

    def test_p2_uses_competition_scores_for_aggregation(self):
        p1 = self._model(stage=1)
        p2 = self._model(stage=2)
        self._fill_queue(p1, self.memory)
        self._fill_queue(p2, self.memory)

        p1_prototypes, p1_selected, _, _ = p1._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha)
        p2_prototypes, p2_selected, _, _ = p2._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha)

        self.assertTrue(torch.equal(p1_selected, p2_selected))
        self.assertFalse(torch.allclose(p1_prototypes, p2_prototypes))
        normalized_memory = F.normalize(self.memory, dim=1)
        class_zero_indices = p2_selected[0][p2_selected[0] >= 0]
        components = torch.cat([
            self.exemplar[0:1],
            normalized_memory[class_zero_indices],
        ], dim=0)
        all_similarity = self.exemplar @ components.t()
        expected_scores = (
            self.alpha * all_similarity[0] -
            (1.0 - self.alpha) * all_similarity[1:].max(dim=0)[0])
        expected = (
            torch.softmax(expected_scores, dim=0).unsqueeze(1) *
            components).sum(dim=0)
        self.assertTrue(torch.allclose(p2_prototypes[0], expected))

    def test_p3_outputs_unit_norm_prototypes(self):
        model = self._model(stage=3)
        self._fill_queue(model, self.memory)

        prototypes, _, _, _ = model._class_prototypes(
            self.exemplar, topk=4, alpha=self.alpha)

        self.assertTrue(torch.allclose(
            prototypes.norm(dim=1), torch.ones(3), atol=1e-6))

    def test_invalid_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            self._model(stage=4)


if __name__ == '__main__':
    unittest.main()
