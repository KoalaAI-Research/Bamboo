import math
import torch.nn as nn
import torch.nn.functional as F
import torch

# With this implementation, the SlidingWindowMultiHeadAttention class provides a local attention mechanism
# that focuses only on a subset of tokens around a given query within a specified window size.
class SlidingWindowMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, window_size, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.window_size = window_size  # Window size for sliding window attention

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Register a buffer for the sliding window mask
        self.register_buffer('mask', self.create_sliding_window_mask(context_length, window_size))

    @staticmethod
    def create_sliding_window_mask(context_length, window_size):
        # Create a block matrix with a sliding window of attention
        mask = torch.zeros(context_length, context_length)
        for i in range(context_length):
            start = max(0, i - window_size)
            end = min(context_length, i + window_size + 1)
            mask[i, start:end] = 1
        return mask

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention within the sliding window
        attn_scores = queries @ keys.transpose(2, 3) / math.sqrt(self.head_dim)

        # Apply the sliding window mask
        mask = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_vec = torch.matmul(attn_probs, values)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec