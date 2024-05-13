import unittest
import torch
from utils import LayerNorm


class TestUtils(unittest.TestCase):
    def test_layernorm(self):
        layer_norm = LayerNorm(embed_dim=6)
        x = torch.randn(5, 6)
        y = layer_norm(x)
        y_layer_mean = y.mean(-1)
        y_layer_std = y.std(-1)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y_layer_mean, torch.zeros_like(y_layer_mean), atol=1e-5))
        self.assertTrue(torch.allclose(y_layer_std, torch.ones_like(y_layer_std), atol=1e-5))


if __name__ == '__main__':
    unittest.main()