import unittest
import torch
from utils import LayerNorm, text_to_token_ids, token_ids_to_text, calc_loss_batch, calc_loss_loader
from gpt import GPTModel
from tokenizer import get_gpt2_tokenizer
from torch.utils.data import DataLoader


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 50257,
            "sequence_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False
        }
        self.model = GPTModel(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = get_gpt2_tokenizer()

    def test_layernorm(self):
        layer_norm = LayerNorm(embed_dim=6)
        x = torch.randn(5, 6)
        y = layer_norm(x)
        y_layer_mean = y.mean(-1)
        y_layer_std = y.std(-1)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y_layer_mean, torch.zeros_like(y_layer_mean), atol=1e-5))
        self.assertTrue(torch.allclose(y_layer_std, torch.ones_like(y_layer_std), atol=1e-5))

    def test_text_to_token_ids(self):
        text = "Hello, how are you?"
        token_ids = text_to_token_ids(text, self.tokenizer)
        self.assertIsInstance(token_ids, torch.Tensor)
        self.assertEqual(token_ids.shape, (1, 6))

    def test_token_ids_to_text(self):
        token_ids = torch.tensor([[15496, 11, 703, 389, 345, 30]])
        text = token_ids_to_text(token_ids, self.tokenizer)
        self.assertIsInstance(text, str)
        self.assertEqual(text, "Hello, how are you?")

    def test_calc_loss_batch(self):
        input_batch = torch.randint(0, self.config["vocab_size"], (2, self.config["sequence_length"]))
        target_batch = torch.randint(0, self.config["vocab_size"], (2, self.config["sequence_length"]))
        loss = calc_loss_batch(input_batch=input_batch, target_batch=target_batch, model=self.model, device=self.device)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())

    def test_calc_loss_loader(self):
        # Create a random tensor of size (100, sequence_length)
        random_data = torch.randint(0, self.config["vocab_size"], (100, self.config["sequence_length"]))
        # Add an extra dimension to represent the batch dimension
        random_data = random_data.unsqueeze(1)
        # Convert the random tensor into a DataLoader
        dataloader = DataLoader(random_data, batch_size=2)
        loss = calc_loss_loader(dataloader, self.model, self.device)
        self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main()