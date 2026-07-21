import unittest

import torch
import torch.nn.functional as F

from net.ose_resa import OSEResA
from tests.test_ose_resa_lmix import TinyEncoder


class OSEResAQueueCorrectionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        self.batch_size = 4
        self.num_classes = 3
        self.view_a = torch.randn(self.batch_size, 3, 4, 5, 1)
        self.view_b = torch.randn(self.batch_size, 3, 4, 5, 1)
        self.exemplar = torch.randn(
            self.num_classes, 3, 4, 5, 1)

    def _model(self, queue_contrast_enabled=True):
        return OSEResA(
            base_encoder='tests.test_ose_resa_lmix.TinyEncoder',
            pretrain=True,
            feature_dim=6,
            projector_hidden_dim=12,
            projector_layers=2,
            use_predictor=True,
            ose_enabled=True,
            queue_size=8,
            queue_contrast_enabled=queue_contrast_enabled,
            instance_feature_dim=4,
            instance_queue_size=5,
            instance_temperature=0.2,
            hidden_dim=8,
            num_class=self.num_classes,
            dropout=0.0)

    def test_uniform_sidecars_reproduce_raw_infonce_logits(self):
        model = self._model()
        query = torch.randn(2, 4)
        key = torch.randn(2, 4)
        category = F.one_hot(
            torch.tensor([0, 2]), self.num_classes).float()

        logits, weight, _ = model._queue_contrastive_logits(
            query, key, category)
        expected_positive = torch.sum(
            F.normalize(query, dim=1) * F.normalize(key, dim=1),
            dim=1, keepdim=True) / model.instance_temperature
        expected_negative = torch.matmul(
            F.normalize(query, dim=1), model.instance_queue.detach())
        expected_negative = expected_negative / model.instance_temperature

        self.assertTrue(torch.equal(weight, torch.ones_like(weight)))
        self.assertTrue(torch.allclose(
            logits[:, :1], expected_positive, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(
            logits[:, 1:], expected_negative, atol=1e-7, rtol=1e-6))

    def test_high_confidence_same_class_is_suppressed_only_as_negative(self):
        model = self._model()
        queued_classes = torch.tensor([0, 1, 2, 1, 2])
        model.category_queue.copy_(
            F.one_hot(queued_classes, self.num_classes).float().t())
        model.confidence_queue.fill_(1.0)
        category = F.one_hot(
            torch.tensor([0]), self.num_classes).float()
        query = torch.randn(1, 4)
        key = torch.randn(1, 4)

        logits, weight, confidence = model._queue_contrastive_logits(
            query, key, category)
        expected_positive = torch.sum(
            F.normalize(query, dim=1) * F.normalize(key, dim=1), dim=1)
        expected_positive = expected_positive / model.instance_temperature

        self.assertAlmostEqual(confidence.item(), 1.0, places=6)
        self.assertAlmostEqual(weight[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(weight[0, 1].item(), 1.0, places=6)
        self.assertTrue(torch.allclose(
            logits[:, 0], expected_positive, atol=1e-7, rtol=1e-6))

    def test_instance_category_confidence_queues_wrap_in_lockstep(self):
        model = self._model()
        first_keys = torch.eye(4)
        first_classes = torch.tensor([0, 1, 2, 0])
        first_categories = F.one_hot(
            first_classes, self.num_classes).float()
        first_confidence = torch.tensor([0.1, 0.2, 0.3, 0.4])
        model._dequeue_and_enqueue_instance(
            first_keys, first_categories, first_confidence)

        second_keys = -torch.eye(4)[:3]
        second_classes = torch.tensor([2, 1, 0])
        second_categories = F.one_hot(
            second_classes, self.num_classes).float()
        second_confidence = torch.tensor([0.5, 0.6, 0.7])
        model._dequeue_and_enqueue_instance(
            second_keys, second_categories, second_confidence)

        expected_keys = torch.stack([
            second_keys[1], second_keys[2], first_keys[2],
            first_keys[3], second_keys[0]], dim=0)
        expected_classes = torch.tensor([1, 0, 2, 0, 2])
        expected_confidence = torch.tensor([0.6, 0.7, 0.3, 0.4, 0.5])
        self.assertEqual(model.instance_queue_ptr.item(), 2)
        self.assertTrue(torch.allclose(
            model.instance_queue.t(), expected_keys))
        self.assertTrue(torch.equal(
            model.category_queue.argmax(dim=0), expected_classes))
        self.assertTrue(torch.allclose(
            model.confidence_queue, expected_confidence))

    def test_queue_loss_detaches_category_and_teacher_but_trains_query(self):
        model = self._model()
        online_features = torch.randn(3, 8, requires_grad=True)
        teacher_features = torch.randn(3, 8, requires_grad=True)
        category_logits = torch.randn(
            3, self.num_classes, requires_grad=True)
        category = torch.softmax(category_logits, dim=1)

        loss, _, weight, confidence = model._queue_contrastive_loss(
            online_features, teacher_features, category)
        loss.backward()

        self.assertIsNotNone(online_features.grad)
        self.assertIsNone(teacher_features.grad)
        self.assertIsNone(category_logits.grad)
        self.assertFalse(weight.requires_grad)
        self.assertFalse(confidence.requires_grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.instance_projector_q.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.instance_projector_k.parameters()))

    def test_full_q4_mf_queue_forward_reuses_backbone_and_backpropagates(self):
        model = self._model()
        model.train()
        permutation = torch.tensor([2, 0, 3, 1])
        beta = 0.4
        mixed_view = (
            beta * self.view_b +
            (1.0 - beta) * self.view_a[permutation])
        old_instance_queue = model.instance_queue.clone()
        queue_events = []
        original_loss = model._queue_contrastive_loss
        original_enqueue = model._dequeue_and_enqueue_instance

        def tracked_loss(*args, **kwargs):
            queue_events.append('logits')
            return original_loss(*args, **kwargs)

        def tracked_enqueue(*args, **kwargs):
            queue_events.append('enqueue')
            return original_enqueue(*args, **kwargs)

        model._queue_contrastive_loss = tracked_loss
        model._dequeue_and_enqueue_instance = tracked_enqueue

        losses = model(
            self.view_a, self.view_b, self.exemplar,
            ose_topk=4,
            sample_indices=torch.arange(self.batch_size),
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_mix_proto=True,
            compute_mix_ins=True)
        total = (
            losses['cluster'] + losses['proto'] +
            losses['mix_proto'] + losses['mix_ins'] +
            losses['queue_corr'])
        total.backward()

        self.assertTrue(torch.isfinite(losses['queue_corr']))
        self.assertEqual(model.encoder_q.forward_feature_calls, 4)
        self.assertEqual(model.encoder_k.forward_feature_calls, 2)
        self.assertEqual(queue_events, ['logits', 'enqueue'])
        self.assertEqual(model.instance_queue_ptr.item(), self.batch_size)
        self.assertFalse(torch.equal(model.instance_queue, old_instance_queue))
        self.assertIsNotNone(model.encoder_q.input_layer.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.instance_projector_q.parameters()))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.instance_projector_k.parameters()))

    def test_disabled_switch_preserves_existing_q4_mf_losses(self):
        torch.manual_seed(31)
        baseline = self._model(queue_contrast_enabled=False)
        torch.manual_seed(31)
        corrected = self._model(queue_contrast_enabled=True)
        permutation = torch.tensor([1, 3, 0, 2])
        beta = 0.65
        mixed_view = (
            beta * self.view_b +
            (1.0 - beta) * self.view_a[permutation])
        kwargs = dict(
            ose_topk=4,
            sample_indices=torch.arange(self.batch_size),
            mixed_view=mixed_view,
            mix_index=permutation,
            mix_beta=beta,
            compute_mix_proto=True,
            compute_mix_ins=True)

        baseline_losses = baseline(
            self.view_a, self.view_b, self.exemplar, **kwargs)
        corrected_losses = corrected(
            self.view_a, self.view_b, self.exemplar, **kwargs)
        for name in ('cluster', 'proto', 'mix_proto', 'mix_ins'):
            self.assertTrue(torch.equal(
                baseline_losses[name], corrected_losses[name]), name)

    def test_state_dict_contains_all_queue_correction_state(self):
        model = self._model()
        state = model.state_dict()
        for name in (
                'instance_projector_q.0.weight',
                'instance_projector_k.0.weight',
                'instance_queue', 'category_queue', 'confidence_queue',
                'instance_queue_ptr'):
            self.assertIn(name, state)
        clone = self._model()
        clone.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
