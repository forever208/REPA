"""Inference-only VA-VAE loader for REPA f16 latent training."""

from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn

from models.SDVAE import Decoder, DiagonalGaussianDistribution, Encoder


CONFIG_PATH = Path(__file__).with_name("VAVAE_config_imgnet256.yaml")


class AutoencoderKL(nn.Module):
    """Config-driven VA-VAE encoder/decoder compatible with released weights."""

    def __init__(self, ddconfig, embed_dim=32):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = nn.Conv2d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

    def encode(self, inputs):
        moments = self.quant_conv(self.encoder(inputs))
        return DiagonalGaussianDistribution(moments)

    def decode(self, latents):
        return self.decoder(self.post_quant_conv(latents))

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VA-VAE checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            raise TypeError("VA-VAE checkpoint must contain a state dictionary")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if state and all(key.startswith("module.") for key in state):
            state = {
                key.removeprefix("module."): value for key, value in state.items()
            }
        model_prefixes = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
        state = {
            key: value for key, value in state.items()
            if key.startswith(model_prefixes)
        }
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            preview = ", ".join(incompatible.missing_keys[:10])
            raise RuntimeError(f"Checkpoint is missing VA-VAE weights: {preview}")
        return self

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"VA-VAE config not found: {CONFIG_PATH}")
        try:
            OmegaConf = import_module("omegaconf").OmegaConf
        except (ImportError, AttributeError) as error:
            raise ImportError("VA-VAE config loading requires omegaconf") from error

        config = OmegaConf.load(CONFIG_PATH)
        params = OmegaConf.to_container(config.model.params, resolve=True)
        return cls(
            ddconfig=params["ddconfig"],
            embed_dim=params.get("embed_dim", params["ddconfig"]["z_channels"]),
        ).load_checkpoint(checkpoint_path)
