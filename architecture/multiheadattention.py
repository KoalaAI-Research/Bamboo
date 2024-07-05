import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, window_size):
        super(MultiHeadAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.emb_dim = emb_dim

        assert self.emb_dim % self.num_heads == 0

        self.depth = emb_dim // num_heads
        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.dense = nn.Linear(emb_dim, emb_dim)
    
    def sliding_window_mask(self, length):
        mask = torch.full((length, length), float("-inf"))
        for i in range(length):
            start = max(0, i - self.window_size)
            end = min(length, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        q = self.wq(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        # Apply sliding window mask
        mask = self.sliding_window_mask(seq_length).to(matmul_qk.device)
        matmul_qk += mask

        # Apply softmax
        attention_weights = F.softmax(matmul_qk, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_dim)
        output = self.dense(output)

        return output
    
# PyTorch implementation of Multi-Head Attention with Scaled Dot-Product Attention and Causal Masking
# Uses a self-attention algorithm called Flash-Attention (v1)
# Reference: https://arxiv.org/abs/2205.14135
class MultiHeadFlashAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec
    
"""from flash_attn.flash_attention import FlashAttention
class MultiHeadSlidingFlashAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, window_size):
        super(MultiHeadAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.emb_dim = emb_dim

        assert self.emb_dim % self.num_heads == 0

        self.depth = emb_dim // num_heads
        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.dense = nn.Linear(emb_dim, emb_dim)
        
        self.flash_attention = FlashAttention(cuda_fp16=True)  # For FP16 support

    def sliding_window_mask(self, length):
        mask = torch.full((length, length), float("-inf"))
        for i in range(length):
            start = max(0, i - self.window_size)
            end = min(length, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        q = self.wq(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_length, self.num_heads, self.depth).transpose(1, 2)

        qkv = torch.stack([q, k, v], dim=2)  # Combine Q, K, V for Flash Attention

        if mask is None:
            mask = self.sliding_window_mask(seq_length).to(qkv.device)
        
        output = self.flash_attention(qkv, mask)  # Using Flash Attention
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_dim)
        output = self.dense(output)

        return output"""

#If you have flash attention v2, you can use this implementation instead of the one above:
"""
from flash_attn_v2 import flash_attn_func
class MultiHeadFlashAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv

        # FlashAttention V2
        context_vec = flash_attn_func(
            q = queries,
            k = keys,
            v = values,
            dropout_p = self.dropout if self.training else 0.0,
            causal = True  # For auto-regressive models
        )

        context_vec = context_vec.permute(0, 2, 1, 3).contiguous().view(batch_size, num_tokens, -1)
        context_vec = self.proj(context_vec)

        return context_vec
"""