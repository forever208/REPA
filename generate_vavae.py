# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Generate VA-VAE samples with DDP and compute FID."""

import argparse
import math
import os
import shutil

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models.VAVAE import AutoencoderKL
from models.sit import SiT_models
from samplers import euler_maruyama_sampler, euler_sampler
from tools.fid_score import calculate_fid_given_paths
from utils import load_legacy_checkpoints


LATENT_CHANNELS = 32
VAE_DOWNSAMPLE_FACTOR = 16


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "DDP sampling requires at least one GPU."
    if args.resolution % VAE_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"Image size must be divisible by {VAE_DOWNSAMPLE_FACTOR} for VA-VAE."
        )
    if args.latents_scale <= 0:
        raise ValueError("--latents-scale must be positive.")
    if args.num_fid_samples <= 0:
        raise ValueError("--num-fid-samples must be positive.")
    if args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be at least 1.0.")

    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // VAE_DOWNSAMPLE_FACTOR
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=LATENT_CHANNELS,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(",")],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    state_dict = torch.load(
        args.ckpt,
        map_location=f"cuda:{device}",
        weights_only=False,
    )["ema"]
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_checkpoint(args.vae_checkpoint).to(device)
    vae.eval()
    vae.requires_grad_(False)
    stat_shape = (1, LATENT_CHANNELS, 1, 1)
    latents_scale = torch.full(
        stat_shape, args.latents_scale, device=device
    )
    latents_bias = torch.full(
        stat_shape, args.latents_bias, device=device
    )

    model_name = args.model.replace("/", "-")
    ckpt_name = os.path.basename(args.ckpt).removesuffix(".pt")
    vae_name = os.path.splitext(os.path.basename(args.vae_checkpoint))[0]
    folder_name = (
        f"{model_name}-{ckpt_name}-size-{args.resolution}-vae-{vae_name}-"
        f"scale-{args.latents_scale}-bias-{args.latents_bias}-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    )
    sample_folder = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        if os.path.isdir(sample_folder):
            shutil.rmtree(sample_folder)
        os.makedirs(sample_folder)
        print(f"Saving .png samples at {sample_folder}")
    dist.barrier()

    batch_size = args.per_proc_batch_size
    global_batch_size = batch_size * dist.get_world_size()
    total_samples = math.ceil(
        args.num_fid_samples / global_batch_size
    ) * global_batch_size
    samples_per_gpu = total_samples // dist.get_world_size()
    iterations = samples_per_gpu // batch_size
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            "projector Parameters: "
            f"{sum(p.numel() for p in model.projectors.parameters()):,}"
        )

    progress = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    for _ in progress:
        noise = torch.randn(
            batch_size,
            LATENT_CHANNELS,
            latent_size,
            latent_size,
            device=device,
        )
        labels = torch.randint(
            0, args.num_classes, (batch_size,), device=device
        )
        sampling_kwargs = {
            "model": model,
            "latents": noise,
            "y": labels,
            "num_steps": args.num_steps,
            "heun": args.heun,
            "cfg_scale": args.cfg_scale,
            "guidance_low": args.guidance_low,
            "guidance_high": args.guidance_high,
            "path_type": args.path_type,
        }
        if args.mode == "sde":
            samples = euler_maruyama_sampler(**sampling_kwargs).float()
        elif args.mode == "ode":
            samples = euler_sampler(**sampling_kwargs).float()
        else:
            raise ValueError(f"Unsupported sampling mode: {args.mode}")

        samples = vae.decode(samples / latents_scale + latents_bias)
        samples = ((samples + 1) / 2).clamp(0, 1)
        samples = (
            (255 * samples)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for index_in_batch, sample in enumerate(samples):
            index = index_in_batch * dist.get_world_size() + rank + total
            if index >= args.num_fid_samples:
                break
            Image.fromarray(sample).save(
                os.path.join(sample_folder, f"{index:06d}.png")
            )
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        fid = calculate_fid_given_paths((args.fid_stat, sample_folder))
        print(f"fid{args.num_fid_samples}={fid}")
        shutil.rmtree(sample_folder)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SiT_models.keys()),
        default="SiT-XL/1",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument(
        "--resolution", type=int, choices=[256, 512], default=256
    )
    parser.add_argument(
        "--fused-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vae-checkpoint",
        "--vae_checkpoint",
        dest="vae_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--latents-scale",
        "--latents_scale",
        dest="latents_scale",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--latents-bias",
        "--latents_bias",
        dest="latents_bias",
        type=float,
        required=True,
    )
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument(
        "--fid-stat",
        "--fid_stat",
        dest="fid_stat",
        type=str,
        required=True,
    )
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument(
        "--path-type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--heun",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument(
        "--legacy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
