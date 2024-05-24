import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import collections 
from itertools import repeat
from collections import OrderedDict

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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
    def __init__(self, enc_emb_dim, dec_emb_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = dec_emb_dim
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(dec_emb_dim, dec_emb_dim)
        self.q = nn.Linear(dec_emb_dim, dec_emb_dim)
        self.k = nn.Linear(enc_emb_dim, dec_emb_dim)
        self.v = nn.Linear(enc_emb_dim, dec_emb_dim)

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
    def __init__(self, enc_emb_dim, dec_emb_dim, num_heads, dropout=0.5):
        super().__init__()
        self.emb_dim = dec_emb_dim
        self.norm1 = nn.LayerNorm(dec_emb_dim)
        self.attn1 = MultiheadCrossAttention(enc_emb_dim, dec_emb_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.block2 = nn.Sequential(
            nn.LayerNorm (dec_emb_dim),
            FeedForwardBlock(dec_emb_dim),
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