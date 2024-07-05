import torch
import torch.nn as nn
import architecture.multiheadattention as mha

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x) #faster approximation, old code below for reference

        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, emb_dim, drop_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = mha.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
    
from architecture.multiheadattention import MultiHeadAttention
import torch.utils.checkpoint as checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, window_size, drop_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads, window_size)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.ffn = FeedForward(emb_dim, drop_rate)

    def forward(self, x):
        # Using checkpointing for attention and feedforward layers
        x = checkpoint.checkpoint(self.attention, x)
        x = self.norm1(x)
        x = checkpoint.checkpoint(self.ffn, x)
        out = self.norm2(x)
        return out