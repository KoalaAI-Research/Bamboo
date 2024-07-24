import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, window_size, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.window_size = window_size
        self.scale = 1 / math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Compute Q, K, V in a single matrix multiplication
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create and apply sliding window mask
        mask = self.create_sliding_window_mask(num_tokens, self.window_size, x.device)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        context_vec = torch.matmul(attn_probs, v)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

    @staticmethod
    def create_sliding_window_mask(num_tokens, window_size, device):
        mask = torch.zeros(num_tokens, num_tokens, device=device)
        for i in range(num_tokens):
            start = max(0, i - window_size)
            end = min(num_tokens, i + window_size + 1)
            mask[i, start:end] = 1
        return mask.unsqueeze(0).unsqueeze(0)  # Add dimensions for batch and heads