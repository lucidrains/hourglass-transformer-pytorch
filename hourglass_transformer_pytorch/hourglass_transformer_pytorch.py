import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

# main class

class HourglassTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        depth,
        heads = 8,
        dim_head = 64,
        causal = True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
        device = x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        return self.to_logits(x)
