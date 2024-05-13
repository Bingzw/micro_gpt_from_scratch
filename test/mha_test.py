import unittest
import torch
from mha import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        d_in = 16
        d_out = 24
        sequence_length = 50
        dropout = 0.1
        num_heads = 3
        qkv_bias = False
        self.mha = MultiHeadAttention(d_in, d_out, sequence_length, dropout, num_heads, qkv_bias)
        batch_size = 10
        self.x = torch.randn(batch_size, sequence_length, d_in)

    def test_forward(self):
        result = self.mha(self.x)
        # Assert the expected result
        expected_shape = (10, 50, 24)
        self.assertEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()