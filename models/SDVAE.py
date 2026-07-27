"""Inference-only EQ-VAE used by REPA.

The module mirrors the architecture and parameter names from the EQ-VAE
project, while intentionally omitting its discriminator, perceptual loss, and
DSM training dependencies.
"""

from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange


def nonlinearity(x):
    return x * torch.sigmoid(x)


def normalize(in_channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        _, _, height, width = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        output = torch.einsum("bhde,bhdn->bhen", context, q)
        output = rearrange(
            output,
            "b heads c (h w) -> b (heads c) h w",
            h=height,
            w=width,
        )
        return self.to_out(output)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        if self.with_conv:
            x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, self.out_channels, 3, 1, 1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, self.out_channels, 1, 1, 0
                )

    def forward(self, x, temb):
        hidden = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:
            hidden = hidden + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        hidden = self.conv2(self.dropout(nonlinearity(self.norm2(hidden))))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + hidden


class LinAttnBlock(LinearAttention):
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        hidden = self.norm(x)
        q, k, v = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * (int(channels) ** -0.5)
        weights = nn.functional.softmax(weights, dim=2).permute(0, 2, 1)
        hidden = torch.bmm(
            v.reshape(batch, channels, height * width), weights
        ).reshape(batch, channels, height, width)
        return x + self.proj_out(hidden)


def make_attn(in_channels, attn_type="vanilla"):
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    if attn_type == "none":
        return nn.Identity()
    if attn_type == "linear":
        return LinAttnBlock(in_channels)
    raise ValueError(f"Unknown attention type: {attn_type}")


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        del out_ch, ignore_kwargs
        if use_linear_attn:
            attn_type = "linear"
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.norm_out = normalize(block_in)
        output_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, output_channels, 3, 1, 1)

    def forward(self, x):
        hidden = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden = self.down[i_level].block[i_block](hidden, None)
                if len(self.down[i_level].attn) > 0:
                    hidden = self.down[i_level].attn[i_block](hidden)
            if i_level != self.num_resolutions - 1:
                hidden = self.down[i_level].downsample(hidden)
        hidden = self.mid.block_1(hidden, None)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden, None)
        return self.conv_out(nonlinearity(self.norm_out(hidden)))


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        del in_channels, ignore_kwargs
        if use_linear_attn:
            attn_type = "linear"
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, z):
        hidden = self.conv_in(z)
        hidden = self.mid.block_1(hidden, None)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden, None)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden, None)
                if len(self.up[i_level].attn) > 0:
                    hidden = self.up[i_level].attn[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        if self.give_pre_end:
            return hidden
        hidden = self.conv_out(nonlinearity(self.norm_out(hidden)))
        return torch.tanh(hidden) if self.tanh_out else hidden


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    """EQ-VAE encoder/decoder with the ImageNet-256 architecture."""

    def __init__(self):
        super().__init__()
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_resolutions": (),
            "dropout": 0.0,
        }
        embed_dim = 4
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        posterior = self.encode(x)
        latents = posterior.sample()
        return self.decode(latents), posterior, latents

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"EQ-VAE checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no model or state_dict entry"
            )
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "EQ-VAE checkpoint is missing model parameters: "
                + ", ".join(incompatible.missing_keys)
            )
        unsupported_keys = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("loss.")
        ]
        if unsupported_keys:
            raise RuntimeError(
                "EQ-VAE checkpoint has unsupported parameters: "
                + ", ".join(unsupported_keys)
            )
        return self

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        return cls().load_checkpoint(checkpoint_path)
