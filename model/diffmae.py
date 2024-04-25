import einops
import torch
import torch.nn as nn

from model.modules import *
# patch embedding, positional encoding, noise addition ...
# from modules import *


class DiffMAE(nn.Module):
    def __init__(self, args, diffusion, num_heads=8, enc_depth=4):
        super().__init__()
        self.patch_size = args.patch_size
        self.mask_ratio = args.mask_ratio
        self.patch_embed = nn.Conv1d(args.n_feats, args.emb_dim, kernel_size=args.patch_size,
                                          stride=args.patch_size)
        self.num_patches = int(args.input_length / args.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, args.emb_dim) * .02)
        self.norm = nn.LayerNorm(args.emb_dim)

        self.enc1 = EncoderBlock(args.emb_dim, num_heads)
        self.enc2 = EncoderBlock(args.emb_dim, num_heads)
        self.enc3 = EncoderBlock(args.emb_dim, num_heads)
        self.enc4 = EncoderBlock(args.emb_dim, num_heads)

        self.decoder_embed = nn.Linear(args.emb_dim, args.dec_emb_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, int(self.num_patches * args.mask_ratio), args.dec_emb_dim) * .02)
        self.decoder_norm = nn.LayerNorm(args.emb_dim)
        self.decoder_pred = nn.Linear(args.dec_emb_dim, args.patch_size * args.n_feats, bias=True)

        self.dec1 = DecoderBlock(args.dec_emb_dim, num_heads)
        self.dec2 = DecoderBlock(args.dec_emb_dim, num_heads)
        self.dec3 = DecoderBlock(args.dec_emb_dim, num_heads)
        self.dec4 = DecoderBlock(args.dec_emb_dim, num_heads)

        self.diffusion = diffusion

    def patchify(self, seq):
        """
        seq: (N, sequence_length, features)
        x: (N, L, patch_size**2 *3)

        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3) patch_size 제곱이 되는 것은 가로, 세로가 따로 있기 때문임
        """
        p = self.patch_size
        f = seq.shape[2]

        np = seq.shape[1] // p # input_sequence // patch_size # number of patches
        x = einops.rearrange(seq, 'b (np p) f -> b np (p f)', b=seq.shape[0], np=np, p=p, f=f)

        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        f = x.shape[2] // p
        np = x.shape[1]
    
        seq = einops.rearrange(x, 'b np (p f) -> b (np p) f', b=x.shape[0], np=np, p=p, f=f)
        return seq

    def random_masking(self, input, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]

        visible_tokens = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        masked_tokens = torch.gather(x, dim=1, index=ids_masked.unsqueeze(-1).repeat(1, 1, D))
        visible_input = torch.gather(input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input.shape[2]))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_tokens, masked_tokens, visible_input, ids_masked, ids_restore, mask

    def forward(self, x):
        input = self.patchify(x)
        t = self.diffusion.sample_timesteps(x.shape[0])

        x = self.patch_embed(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x += self.pos_embed[:, 1:, :]

        x, mask_token, visible_input, ids_masked, ids_restore, mask = self.random_masking(input, x, self.mask_ratio)

        mask_token, noise = self.diffusion.noise_samples(mask_token, t)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        mask_token = self.decoder_embed(mask_token)
        mask_token += self.decoder_pos_embed

        x5 = self.dec1(mask_token, x4)
        x6 = self.dec2(x5, x3)
        x7 = self.dec3(x6, x2)
        x8 = self.dec4(x7, x1)
        x8 = self.decoder_pred(x8)

        return x8, visible_input, ids_masked, ids_restore, mask
