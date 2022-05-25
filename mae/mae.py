import copy
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import DropPath, Mlp


class ViT(nn.Module):

    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        img_channels=3,
        patch_size=16,
        blocks=24,
        embed_dim=1024,
        heads=16,
        droppath=0.1,
        dropout=0.0,
        global_pool=True,
    ):
        super().__init__()
        self.global_pool = global_pool

        self.encoder = ViTEncoder(
            img_size=img_size,
            img_channels=img_channels,
            patch_size=patch_size,
            blocks=blocks,
            embed_dim=embed_dim,
            heads=heads,
            droppath=droppath,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        torch.nn.init.trunc_normal_(self.head.weight, std=2e-5)

        if global_pool:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.encoder.norm = nn.Identity()

    def forward(self, imgs):
        x, _, _ = self.encoder(imgs, mask_ratio=0.0)  # [N, L, C]

        if self.global_pool:
            x = x[:, 1:].mean(1)
            x = self.norm(x)
        else:
            x = x[:, 0]  # [N, C], cls token
        x = self.head(x)  # [N, num_classes]
        return x


class MaskedAutoEncoder(nn.Module):

    def __init__(
        self,
        # input image
        img_size=224,
        img_channels=3,
        patch_size=16,
        # encoder params
        encoder_blocks=24,
        encoder_embed_dim=1024,
        encoder_heads=16,
        encoder_dropout=0.0,
        # decoder params
        decoder_blocks=8,
        decoder_embed_dim=512,
        decoder_heads=16,
        decoder_dropout=0.0,
        norm_pix_loss=True,
    ):
        super().__init__()

        self.norm_pix_loss = norm_pix_loss

        ###### ENCODER ######
        self.encoder = ViTEncoder(
            img_size=img_size,
            img_channels=img_channels,
            patch_size=patch_size,
            blocks=encoder_blocks,
            embed_dim=encoder_embed_dim,
            heads=encoder_heads,
            dropout=encoder_dropout,
        )

        ###### DECODER ######
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder = ViTDecoder(
            img_channels=img_channels,
            num_patches=num_patches,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            blocks=decoder_blocks,
            embed_dim=decoder_embed_dim,
            heads=decoder_heads,
            dropout=decoder_dropout,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs, mask_ratio):
        """
        Args:
            imgs (Tensor): input images, size ``[N, C, H, W]``
            mask_ratio (float): random masking ratio between ``[0, 1]``

        Returns:
            (pred, mask, loss)

            pred (Tensor): predicted images, size ``[N, C, H, W]``
            mask (Tensor): mask, (0 for non-masked, 1 for masked), size ``[N, number of patches]``
            loss (Tensor): loss value
        """
        x, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(x, ids_restore)

        target = self.encoder.patch_embed.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            std = (var + 1.0e-6) ** 0.5
            target = (target - mean) / std

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # unpatchify to visualize
        if self.norm_pix_loss:
            pred = pred * std + mean
        pred = self.encoder.patch_embed.unpatchify(pred)

        return pred, mask, loss


class ViTEncoder(nn.Module):
    """
    ViT encoder
    """

    def __init__(
        self,
        img_size=224,
        img_channels=3,
        patch_size=16,
        blocks=24,
        embed_dim=1024,
        heads=16,
        droppath=0.0,
        dropout=0.0,
    ):
        super().__init__()

        ###### CLS TOKEN ######
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        ###### TRANASFORMER ######
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            img_channels=img_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, droppath, blocks)]  # stochastic depth decay rule
        self.transformer = nn.Sequential(
            *[
                Block(embed_dim, heads, mlp_ratio=4, qkv_bias=True, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(blocks)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # cos-sin positional embedding
        grid_size = int(num_patches ** 0.5)
        self.pos_embed = build_sincos2d(embed_dim, grid_size, use_cls_token=True)

    def forward(self, imgs, mask_ratio):
        """
        Args:
            x: input images, [N, C, H, W]
            mask_ratio: mask ratio

        Returns:
            encoded embeddings, [N, num_patches + 1, embed_dim]
        """

        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        # masking: [N, L, D] -> [N, L * mask_ratio, D]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # encode
        x = self.transformer(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Args:
            x: input sequence, [N, L, D],
            mask_ratio: mask ratio

        Returns:
            x_masked:    masked sequence, [N, L * mask_ratio, D]
            mask:        mask (0 for non-masked, 1 for masked), [N, L]
            ids_restore: ids to be restored, [N, L]
        """
        if mask_ratio == 0.0:
            return x, None, None

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class ViTDecoder(nn.Module):
    """
    ViT Decoder
    """

    def __init__(
        self,
        img_channels,
        patch_size,
        num_patches,
        encoder_embed_dim,
        blocks=8,
        embed_dim=512,
        heads=16,
        dropout=0.0,
    ):
        super().__init__()

        ###### MASK TOKEN ######
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)

        ###### TRANSFORMER ######
        self.proj = nn.Linear(encoder_embed_dim, embed_dim, bias=True)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        block = Block(embed_dim, heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
        self.transformer = nn.Sequential(*[copy.deepcopy(block) for i in range(blocks)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # cos-sin positional embedding
        grid_size = int(num_patches ** 0.5)
        self.pos_embed = build_sincos2d(embed_dim, grid_size, use_cls_token=True)

        ###### PREDICT HEAD ######
        self.pred_head = nn.Linear(embed_dim, img_channels * patch_size ** 2, bias=True)

    def forward(self, x, ids_restore):
        """
        Args:
            x: input embeddings, [N, num_patches + 1, encoder_embed_dim]
            ids_restore: mapping from original pos to masked pos, [N, num_patches]

        Returns:
            imgs: decoded images, [N, C, H, W]
        """
        # proj from encoder_embed_dim to decoder_embed_dim
        x = self.proj(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        tokens = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove cls token
        index = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        tokens = torch.gather(tokens, dim=1, index=index)  # unshuffle
        x = torch.cat([x[:, :1, :], tokens], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # decode
        x = self.transformer(x)
        x = self.norm(x)

        x = self.pred_head(x)
        # remove cls token
        x = x[:, 1:, :]
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=qkv_bias)

    def forward(self, x):
        output, attn_weights = self.attn(x, x, x)
        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        img_channels=3,
        patch_size=16,
        embed_dim=1024,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        self.img_channels = img_channels
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels=img_channels, out_channels=embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        """
        Args:
            x: input images, [N, in_c, H, W]

        Returns:
            flattened patch embeddings
        """
        x = self.proj(x)  # [N, embed_dim, grid_size, grid_size]
        x = torch.flatten(x, start_dim=2)  # [N, embed_dim, grid_size**2]
        x = x.transpose(1, 2)  # [N, grid_size**2, embed_dim]
        return x

    def patchify(self, imgs):
        """
        Args:
            imgs: (N, 3, H, W)

        Returns:
            x: (N, L, patch_size**2 * 3)
        """
        P = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % P == 0

        N, C = imgs.shape[0], imgs.shape[1]
        H = W = imgs.shape[2] // P
        x = imgs.reshape(N, 3, H, P, W, P)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(N, H * W, P ** 2 * C)
        return x

    def unpatchify(self, x):
        """
        Args:
            x: input embeddings, [N, grid_size**2, patch_size**2 * 3]

        Returns:
            2D images, [N, H, W, 3]
        """
        N, P, C = x.shape[0], self.patch_size, self.img_channels
        H, W = self.grid_size, self.grid_size

        x = x.reshape(N, H, W, P, P, C)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(N, C, H * P, W * P)

        return imgs


def build_sincos2d(embed_dim, grid_size, use_cls_token=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

    temperature = 10000.0
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)

    out_h = torch.einsum("m,d->md", grid_h.reshape(-1), omega)
    out_w = torch.einsum("m,d->md", grid_w.reshape(-1), omega)

    sincos = [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)]
    pos_embed = torch.cat(sincos, axis=1).float().unsqueeze(0)

    if use_cls_token:
        cls_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = torch.cat([cls_token, pos_embed], dim=1)

    pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)

    return pos_embed


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoEncoder(patch_size=16, encoder_blocks=12, encoder_embed_dim=768, encoder_heads=12, **kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoEncoder(patch_size=16, encoder_blocks=24, encoder_embed_dim=1024, encoder_heads=16, **kwargs)
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoEncoder(patch_size=14, encoder_blocks=32, encoder_embed_dim=1280, encoder_heads=16, **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = ViT(patch_size=16, blocks=12, embed_dim=768, heads=12, **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = ViT(patch_size=16, blocks=24, embed_dim=1024, heads=16, **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = ViT(patch_size=14, blocks=32, embed_dim=1280, heads=16, **kwargs)
    return model
