from functools import partial
import einops
import torch
import torch.nn as nn

from model.modules import PatchEmbed, EncoderBlock, DecoderBlock, get_2d_sincos_pos_embed

class DiffMAE(nn.Module):
    def __init__(self, args, diffusion, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.patch_size = args.patch_size
        self.mask_ratio = args.mask_ratio
        self.patch_embed = PatchEmbed(args.img_size, args.patch_size,
                                      args.n_channels, args.emb_dim)
        
        self.num_patches = int(args.img_size // args.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, args.emb_dim) * .02)
        self.norm = norm_layer(args.emb_dim)

        self.blocks = nn.ModuleList([
            EncoderBlock(args.emb_dim, args.num_heads) for i in range(args.depth)])

        self.decoder_embed = nn.Linear(args.emb_dim, args.dec_emb_dim, bias=True)
        # num_masked_patches = self.num_patches - int(self.num_patches * (1 - args.mask_ratio))
        # self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_masked_patches + 1, args.dec_emb_dim) * .02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, args.dec_emb_dim) * .02)
        self.decoder_norm = norm_layer(args.dec_emb_dim)
        self.decoder_pred = nn.Linear(args.dec_emb_dim, args.patch_size **2 * args.n_channels, bias=True)

        self.dec_blocks = nn.ModuleList([
            DecoderBlock(args.emb_dim, args.dec_emb_dim, args.num_heads) for i in range(args.depth)])

        self.diffusion = diffusion

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(int(self.patch_embed.num_patches*self.mask_ratio)**.5), cls_token=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
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

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_tokens, masked_tokens, mask, ids_restore, ids_masked, ids_keep
    
    def forward(self, x):
        t = self.diffusion.sample_timesteps(x.shape[0])

        x = self.patch_embed(x)
        x += self.pos_embed[:, 1:, :]

        x, mask_token, mask, ids_restore, ids_masked, ids_keep = self.random_masking(x, self.mask_ratio)

        mask_token, noise = self.diffusion.noise_samples(mask_token, t)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        outputs[-1] = self.norm(outputs[-1])
        
        mask_token = self.decoder_embed(mask_token)
        # mask_token += self.decoder_pos_embed[:, 1:, :]
        decoder_pos_embed = nn.Parameter(
            torch.gather(self.decoder_pos_embed[:, 1:, :].repeat(mask_token.shape[0], 1, 1), dim=1,
                         index=ids_masked.unsqueeze(-1).repeat(1, 1, mask_token.shape[-1])))
        mask_token += decoder_pos_embed
        for dec_block, enc_output in zip(self.dec_blocks, reversed(outputs)):
            mask_token = dec_block(mask_token, enc_output)
        x8 = self.decoder_norm(mask_token)
        x8 = self.decoder_pred(x8)

        return x8, ids_restore, mask, ids_masked, ids_keep
