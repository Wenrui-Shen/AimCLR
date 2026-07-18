import unittest

import torch

from net.ctrgcn import Model


class CTRGCNTest(unittest.TestCase):

    def test_forward_features_and_classifier_shapes(self):
        model = Model(
            in_channels=3,
            hidden_channels=12,
            hidden_dim=48,
            num_class=60,
            dropout=0.0,
            num_point=25,
            num_person=2,
            graph_args={
                'layout': 'ntu-rgb+d',
                'strategy': 'spatial',
            })
        model.eval()
        inputs = torch.randn(2, 3, 16, 25, 2)

        with torch.no_grad():
            features = model.forward_features(inputs)
            logits = model(inputs)

        self.assertEqual(features.shape, (2, 48))
        self.assertEqual(logits.shape, (2, 60))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_official_width_outputs_256_features(self):
        model = Model(
            hidden_channels=64,
            hidden_dim=256,
            graph_args={
                'layout': 'ntu-rgb+d',
                'strategy': 'spatial',
            })
        self.assertEqual(model.fc.in_features, 256)
        self.assertEqual(len(model.layers), 10)


if __name__ == '__main__':
    unittest.main()
