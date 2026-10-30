"""A simple 2D image regression network based on Palette U-Net."""

import torch
from torch import nn
from torch.nn import functional as F


def zero_module(module):
    """Initialize a layer to output zeros, as in the original Palette U-Net."""
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class GroupNorm32(nn.GroupNorm):
    """Compute GroupNorm in float32 for stable mixed-precision training."""

    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


class ResBlock(nn.Module):
    """GN -> SiLU -> 3x3 Conv -> GN -> SiLU -> Dropout -> 3x3 Conv + skip.

    Upsampling/downsampling happens on both branches before the first Conv.
    Downsampling: 2x average pooling. Upsampling: 2x nearest neighbor.
    """

    def __init__(self, in_channels, out_channels=None, dropout=0.2,
                 up=False, down=False):
        super().__init__()
        if up and down:
            raise ValueError("up and down cannot both be True")

        out_channels = out_channels or in_channels
        self.up = up
        self.down = down
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalization(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def _resize(self, x):
        if self.up:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            return F.avg_pool2d(x, kernel_size=2)
        return x

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(self._resize(h))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(self._resize(x)) + h


class AttentionBlock(nn.Module):
    """Global multi-head self-attention (32 channels per head)."""

    def __init__(self, channels, head_channels=32):
        super().__init__()
        if channels % head_channels:
            raise ValueError("channels must be divisible by head_channels")
        self.num_heads = channels // head_channels
        self.head_channels = head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, height, width = x.shape
        n = height * width
        qkv = self.qkv(self.norm(x).reshape(b, c, n))

        # Legacy Palette QKV order: [B, heads, (Q,K,V), head_dim, HW].
        qkv = qkv.reshape(b, self.num_heads, 3, self.head_channels, n)
        q, k, v = qkv.unbind(dim=2)

        # scaled_dot_product_attention expects [B, heads, HW, head_dim].
        h = F.scaled_dot_product_attention(
            q.transpose(-1, -2),
            k.transpose(-1, -2),
            v.transpose(-1, -2),
        )
        h = h.transpose(-1, -2).reshape(b, c, n)
        h = self.proj(h).reshape(b, c, height, width)
        return x + h


class Network(nn.Module):
    """Palette-style U-Net with no gamma embedding and no diffusion logic.

    Default: [B, 6, H, W] -> [B, 3, H, W].
    For RGB -> RGB regression, use Network(in_channels=3).
    """

    def __init__(self, in_channels=6, out_channels=3, base_channels=64,
                 dropout=0.2, use_attention=True):
        super().__init__()
        if base_channels % 32:
            raise ValueError("base_channels must be divisible by 32")

        channel_mults = (1, 2, 4, 8)
        res_blocks = 2
        ch = base_channels

        # Encoder: initial 3x3 Conv, then 2 ResBlocks at each resolution.
        # Downsampling is an additional ResBlock at each of the first 3 levels.
        self.encoder = nn.ModuleList([nn.Conv2d(in_channels, ch, 3, padding=1)])
        skip_channels = [ch]
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(res_blocks):
                self.encoder.append(ResBlock(ch, out_ch, dropout=dropout))
                ch = out_ch
                skip_channels.append(ch)
            if level < len(channel_mults) - 1:
                self.encoder.append(ResBlock(ch, ch, dropout=dropout, down=True))
                skip_channels.append(ch)

        # Only bottleneck attention is active with the public setting
        # attn_res=[16] and channel_mults=[1,2,4,8] (downsample factors 1,2,4,8).
        self.middle = nn.Sequential(
            ResBlock(ch, ch, dropout=dropout),
            AttentionBlock(ch, head_channels=32) if use_attention else nn.Identity(),
            ResBlock(ch, ch, dropout=dropout),
        )

        # Decoder: 3 ResBlocks at each level, each preceded by a concat skip.
        # The last block at each level (except the finest) also upsamples 2x.
        self.decoder = nn.ModuleList()
        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            for i in range(res_blocks + 1):
                skip_ch = skip_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, dropout=dropout)]
                ch = out_ch
                if level > 0 and i == res_blocks:
                    layers.append(ResBlock(ch, ch, dropout=dropout, up=True))
                self.decoder.append(nn.Sequential(*layers))

        assert not skip_channels
        self.head = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input shape [B, C, H, W] (2D images)")
        height, width = x.shape[-2:]

        # Three 2x downsamples need dimensions divisible by 8.
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips = []
        h = x
        for block in self.encoder:
            h = block(h)
            skips.append(h)

        h = self.middle(h)

        for block in self.decoder:
            h = block(torch.cat((h, skips.pop()), dim=1))

        out = self.head(h)
        return out[..., :height, :width]


