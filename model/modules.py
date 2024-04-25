import math
import torch
import torch.nn as nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        scaling = self.emb_dim ** 1/2
        energy = nn.functional.softmax(energy / scaling, dim=-1)
        
        energy = self.att_drop(energy)
        output = torch.einsum('bhqv, bhvd -> bhqd', energy, v)
        output = rearrange(output, 'b h q d -> b q (h d)')
        output = self.projection(output)
        return output

class MultiheadCrossAttention(nn.Module):
    # operate only cross-attention for decoder
    # just change the query, key, value ? 
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(1024, emb_dim)
        self.v = nn.Linear(1024, emb_dim)

    def forward(self, x, enc_output):
        q = rearrange(self.q(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k(enc_output), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v(enc_output), 'b n (h d) -> b h n d', h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        scaling = self.emb_dim ** 1/2
        energy = nn.functional.softmax(energy / scaling, dim=-1)
        energy = self.att_drop(energy)
        output = torch.einsum('bhqv, bhvd -> bhqd', energy, v)
        output = rearrange(output, 'b h q d -> b q (h d)')
        output = self.projection(output)
        return output
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_dim, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_dim * expansion, emb_dim)
        )
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        self.block1 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            MultiHeadAttention(emb_dim, num_heads, dropout),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            FeedForwardBlock(emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res1 = x
        x = self.block1(x)
        x += res1
        res2 = x
        x = self.block2(x)
        x += res2
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn1 = MultiheadCrossAttention(emb_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.block2 = nn.Sequential(
            nn.LayerNorm (emb_dim),
            FeedForwardBlock(emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, enc_output):
        res1 = x
        x = self.norm1(x)
        x = self.attn1(x, enc_output)
        x = self.drop1(x)
        x += res1
        res2 = x
        x = self.block2(x)
        x += res2 
        return x
    
    """ for diffusion """

def linear_beta_schedule(timesteps, beta_start, beta_end):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end

    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clamp(betas, 0, 0.999)