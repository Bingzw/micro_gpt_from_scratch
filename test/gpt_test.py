import unittest
import torch
from gpt import FeedForward, TransformerBlock, GPTModel


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 50257,  # Vocabulary size
            "sequence_length": 1024,  # max length of the sequence
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False  # Query-Key-Value bias
        }
        self.ff = FeedForward(self.config)
        self.transformerblock = TransformerBlock(self.config)
        self.gpt = GPTModel(self.config)

    def test_feedforward(self):
        x = torch.randn(10, 20, self.config["emb_dim"])
        result = self.ff(x)
        # Assert the expected result
        expected_shape = (10, 20, self.config["emb_dim"])
        self.assertEqual(result.shape, expected_shape)

    def test_transformerblock(self):
        x = torch.randn(10, 20, self.config["emb_dim"])
        result = self.transformerblock(x)
        # Assert the expected result
        expected_shape = (10, 20, self.config["emb_dim"])
        self.assertEqual(result.shape, expected_shape)

    def test_gpt(self):
        x = torch.randint(0, self.config["vocab_size"], (10, 20))
        result = self.gpt(x)
        # Assert the expected result
        expected_shape = (10, 20, self.config["vocab_size"])
        self.assertEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()