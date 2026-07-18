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

    def test_shallow_depth_profiles_keep_output_dimension(self):
        expected_channels = {
            3: [8, 16, 32],
            8: [8, 8, 8, 8, 16, 16, 16, 32],
        }
        for num_layers, channels in expected_channels.items():
            with self.subTest(num_layers=num_layers):
                model = Model(
                    hidden_channels=8,
                    hidden_dim=32,
                    num_layers=num_layers,
                    graph_args={
                        'layout': 'ntu-rgb+d',
                        'strategy': 'spatial',
                    })
                actual_channels = [
                    layer.gcn.bn.num_features for layer in model.layers]
                self.assertEqual(actual_channels, channels)
                self.assertEqual(model.fc.in_features, 32)

    def test_rejects_undefined_depth_profile(self):
        with self.assertRaisesRegex(ValueError, 'num_layers'):
            Model(
                hidden_channels=8,
                hidden_dim=32,
                num_layers=7,
                graph_args={
                    'layout': 'ntu-rgb+d',
                    'strategy': 'spatial',
                })


if __name__ == '__main__':
    unittest.main()
