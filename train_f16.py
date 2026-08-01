"""REPA training entry point for VA-VAE f16d32 latents."""

import argparse
import copy
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import shutil

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import CustomDataset
from loss import SILoss
from models.sit import SiT_models
from models.VAVAE import AutoencoderKL, DEFAULT_CONFIG
from samplers import euler_maruyama_sampler, euler_sampler
from tools.fid_score import calculate_fid_given_paths
from train import (
    create_logger,
    parse_args,
    preprocess_raw_image,
    requires_grad,
    sample_posterior,
    update_ema,
)
from utils import load_encoders


LATENT_CHANNELS = 32
VAE_DOWNSAMPLE_FACTOR = 16
DEFAULT_CHECKPOINT = (
    "/home/ningmang/eqvae/exps/imagenet256_VAVAE/"
    "vavae-imagenet256-f16d32-dinov2.pt"
)
SDVAE_DEFAULT_CHECKPOINT = (
    "/home/ningmang/eqvae/exps/imagenet256_SDVAE_ft/checkpoints/40000.ckpt"
)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    report_to = (
        None
        if args.report_to.lower() in {"none", "null", "false", "no"}
        else args.report_to
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=report_to,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as file:
            json.dump(vars(args), file, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        logger = None

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    if args.resolution % VAE_DOWNSAMPLE_FACTOR != 0:
        raise ValueError(
            f"Image size must be divisible by {VAE_DOWNSAMPLE_FACTOR} for VA-VAE."
        )
    if args.latents_scale <= 0:
        raise ValueError("--latents-scale must be positive.")
    latent_size = args.resolution // VAE_DOWNSAMPLE_FACTOR

    if args.enc_type is None:
        raise NotImplementedError()
    encoders, encoder_types, architectures = load_encoders(
        args.enc_type, device, args.resolution
    )
    z_dims = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=LATENT_CHANNELS,
        num_classes=args.num_classes,
        use_cfg=args.cfg_prob > 0,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    )
    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_checkpoint(
        args.vae_checkpoint, config_path=args.vae_config
    ).to(device)
    vae.eval()
    requires_grad(vae, False)
    requires_grad(ema, False)

    stat_shape = (1, LATENT_CHANNELS, 1, 1)
    latents_scale = torch.full(stat_shape, args.latents_scale, device=device)
    latents_bias = torch.full(stat_shape, args.latents_bias, device=device)
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.pre_fetch,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + ".pt"
        ckpt = torch.load(
            f"{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}",
            map_location="cpu",
        )
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["steps"]

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if accelerator.is_main_process and report_to is not None:
        accelerator.init_trackers(
            project_name="REPA",
            config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
        mininterval=5.0,
    )
    for _epoch in range(args.epochs):
        model.train()
        for raw_image, moments, labels in train_dataloader:
            raw_image = raw_image.to(device)
            moments = moments.squeeze(dim=1).to(device)
            labels = labels.to(device)
            if moments.shape[1] != 2 * LATENT_CHANNELS:
                raise ValueError(
                    "VA-VAE feature files must contain 64 channels "
                    f"(mean+std), got shape {tuple(moments.shape)}"
                )
            if args.legacy:
                drop_ids = torch.rand(labels.shape[0], device=device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, labels)

            with torch.no_grad():
                latents = sample_posterior(
                    moments,
                    latents_scale=latents_scale,
                    latents_bias=latents_bias,
                )
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, _arch in zip(
                        encoders, encoder_types, architectures
                    ):
                        encoder_input = preprocess_raw_image(raw_image, encoder_type)
                        representation = encoder.forward_features(encoder_input)
                        if "mocov3" in encoder_type:
                            representation = representation[:, 1:]
                        if "dinov2" in encoder_type:
                            representation = representation["x_norm_patchtokens"]
                        zs.append(representation)

            grad_norm = torch.tensor(0.0, device=device)
            with accelerator.accumulate(model):
                loss, proj_loss = loss_fn(
                    model, latents, {"y": labels}, zs=zs
                )
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                total_loss = loss_mean + proj_loss_mean * args.proj_coeff
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, model)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": (
                    accelerator.gather(proj_loss_mean).mean().detach().item()
                ),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
            }
            progress_bar.set_postfix(**logs, refresh=False)
            if accelerator.is_main_process and (
                global_step == 1 or global_step % args.log_every == 0
            ):
                logger.info(
                    f"step={global_step} loss={logs['loss']:.6f} "
                    f"proj_loss={logs['proj_loss']:.6f} "
                    f"grad_norm={logs['grad_norm']:.3f}"
                )
                if report_to is not None:
                    accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    os.makedirs(args.sample_path, exist_ok=True)
                    logger.info(
                        f"FID evaluation at steps {global_step}, "
                        f"samples are in {args.sample_path}"
                    )
                ema.eval()
                if args.cfg_scale < 1.0:
                    raise ValueError("--cfg-scale must be at least 1.0")

                num_iterations = int(args.num_samples // args.batch_size + 1)
                mini_batch_size = args.batch_size // accelerator.num_processes
                world_size = accelerator.state.num_processes
                local_rank = accelerator.state.local_process_index
                in_channels = getattr(ema, "in_channels")
                for index in tqdm(
                    range(num_iterations),
                    disable=not accelerator.is_main_process,
                    desc="sample2dir",
                ):
                    noise = torch.randn(
                        mini_batch_size,
                        in_channels,
                        latent_size,
                        latent_size,
                        device=device,
                    )
                    sample_labels = torch.randint(
                        0, args.num_classes, (mini_batch_size,), device=device
                    )
                    sampling_kwargs = {
                        "model": ema,
                        "latents": noise,
                        "y": sample_labels,
                        "num_steps": args.num_steps,
                        "heun": args.heun,
                        "cfg_scale": args.cfg_scale,
                        "guidance_low": args.guidance_low,
                        "guidance_high": args.guidance_high,
                        "path_type": args.path_type,
                    }
                    with torch.no_grad():
                        if args.mode == "sde":
                            samples = euler_maruyama_sampler(
                                **sampling_kwargs
                            ).float()
                        elif args.mode == "ode":
                            samples = euler_sampler(**sampling_kwargs).float()
                        else:
                            raise NotImplementedError()
                        samples = vae.decode(
                            samples / latents_scale + latents_bias
                        )
                        samples = ((samples + 1) / 2).clamp(0, 1)
                        samples = (
                            (255 * samples)
                            .permute(0, 2, 3, 1)
                            .to("cpu", dtype=torch.uint8)
                            .numpy()
                        )
                        for batch_index, sample in enumerate(samples):
                            image_id = (
                                index * mini_batch_size * world_size
                                + local_rank * mini_batch_size
                                + batch_index
                            )
                            if image_id >= args.num_samples:
                                break
                            Image.fromarray(sample).save(
                                f"{args.sample_path}/{image_id:06d}.png"
                            )

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    fid = calculate_fid_given_paths(
                        (args.fid_stat, args.sample_path)
                    )
                    logger.info(f"step={global_step} fid{args.num_samples}={fid}")
                    eval_log = os.path.join(
                        args.output_dir,
                        f"eval_{args.mode}_{args.num_samples}.log",
                    )
                    with open(eval_log, "a") as file:
                        print(
                            f"step={global_step} fid{args.num_samples}={fid}",
                            file=file,
                        )
                    shutil.rmtree(args.sample_path)
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                model.train()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_f16_args():
    f16_parser = argparse.ArgumentParser(add_help=False)
    f16_parser.add_argument("--vae-config", default=DEFAULT_CONFIG)
    f16_args, remaining = f16_parser.parse_known_args()
    args = parse_args(remaining)
    args.vae_config = f16_args.vae_config
    if args.vae_checkpoint == SDVAE_DEFAULT_CHECKPOINT:
        args.vae_checkpoint = DEFAULT_CHECKPOINT
    return args


if __name__ == "__main__":
    main(parse_f16_args())
