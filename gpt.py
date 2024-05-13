import torch
import torch.nn as nn
from utils import LayerNorm, GELU
from mha import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
            GELU(),
            nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            sequence_length=config["sequence_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config)
        self.layer_norm1 = LayerNorm(config["emb_dim"])
        self.layer_norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        # shortcut connection for feedforward block
        shortcut = x
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embeddings = nn.Embedding(config["sequence_length"], config["emb_dim"])
        self.drop_layer = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.output_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)  # conver the output to vocab
        # size to forecast the next word

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(torch.arange(sequence_length).to(input_ids.device))
        x = token_embed + position_embed
        x = self.drop_layer(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
