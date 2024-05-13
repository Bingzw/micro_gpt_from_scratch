import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, sequence_length, dropout, num_heads, qkv_bias=False):
        """
        :param d_in: the dimension of the input sequence
        :param d_out: the dimension of the output sequence
        :param sequence_length: the length of the input sequence
        :param dropout: the dropout rate
        :param num_heads: the number of heads
        :param qkv_bias: whether to use bias in the query, key, value linear layers
        """
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # the hidden dim of each head
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1))

    def forward(self, x):
        batch, sequence_length, d_in = x.shape

        # unroll the last dimension: (batch, sequence_length, d_out) -> (batch, sequence_length, num_heads, head_dim)
        keys = self.W_key(x).view(batch, sequence_length, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(batch, sequence_length, self.num_heads, self.head_dim)
        values = self.W_value(x).view(batch, sequence_length, self.num_heads, self.head_dim)

        # Transpose: (batch, sequence_length, num_heads, head_dim) -> (batch, num_heads, sequence_length, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute the attention scores with causal mask
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask_bool = self.mask.bool()[:sequence_length, :sequence_length]
        attn_scores.masked_fill_(mask_bool, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, values).transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch, sequence_length, self.d_out)
        attn_output = self.out_proj(attn_output) # note that the output projection is not necessary needed and it would
        # not affect the correctness of the model when removed

        return attn_output





