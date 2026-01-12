import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path
import json

import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
import wandb

wandb.init(mode="offline")
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from utils import gaussian_blur, downsample_to, rmsc, rmsc_tokens

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
from utils import SpatialNormalization, ALL_SPNORM_METHODS


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  evaluation Loop                                #
#################################################################################

def rmsc_evaluation(args):
    # set accelerator and log
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")  # Save to a JSON file
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    else:
        logger = None

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create DINO, SiT, VAE:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
    else:
        raise NotImplementedError()

    for e in encoders:
        e.eval()
        requires_grad(e, False)

    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    requires_grad(model, False)

    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )

    # load ckpt
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    ema.load_state_dict(ckpt["ema"], strict=True)

    if accelerator.is_main_process:
        logger.info(f"SiT model loaded from {args.ckpt_path}")

    eval_model = ema if args.use_ema else model
    eval_model.eval()

    # Prepare with accelerator (no optimizer needed)
    eval_model = accelerator.prepare(eval_model)

    # dataloader
    dataset = CustomDataset(args.data_dir)
    assert args.batch_size % accelerator.num_processes == 0, f"--batch_size must be divisible by num_processes ({accelerator.num_processes})"
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    loader = accelerator.prepare(loader)

    if accelerator.is_main_process:
        logger.info(f"Dataset: {len(dataset):,} images | eval_num_samples={args.eval_num_samples} | "
                    f"global_batch={args.batch_size} (local={local_batch_size}) | use_ema={args.use_ema} | use_blur={args.gaussian_blur}")

    # evaluation
    total_seen = 0
    sum_rmsc_x = 0.0
    sum_rmsc_z = 0.0
    sum_rmsc_dino = 0.0
    sum_rmsc_loss = 0.0
    spnorm = SpatialNormalization("zscore")

    progress = tqdm(disable=not accelerator.is_local_main_process, desc="Eval", dynamic_ncols=True)
    it = iter(loader)

    while total_seen < args.eval_num_samples:
        try:
            raw_image, x, y = next(it)
        except StopIteration:
            it = iter(loader)
            raw_image, x, y = next(it)

        raw_image = raw_image.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        labels = y

        with accelerator.autocast():
            x = x.squeeze(dim=1)
            x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
            zs = []
            for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                features = encoder.forward_features(raw_image_)  # (B, tokens, dim) or dict for dinov2

                # normalize spatial features
                spnorm_kwargs = {
                    'feat': features['x_norm_patchtokens'],
                    'cls': features['x_norm_clstoken'],
                    'cls_weight': 0.2,
                    'zscore_alpha': 0.6,
                    'zscore_proj_skip_std': False,
                }
                z = spnorm(**spnorm_kwargs)
                zs.append(z)

            rmsc_dino = rmsc_tokens(zs[0])  # (B,)

        # rmsc_x exactly like training
        _, _, hx, wx = raw_image.shape  # uint8 [0,255]
        x_img = raw_image.float() / 255.0  # value range [0, 1] before rmsc eval
        if args.gaussian_blur:
            x_img = gaussian_blur(x_img, kernel_size=7, sigma=1.2)
        downsam_x = downsample_to(x_img, (hx // 8, wx // 8))
        rmsc_x = rmsc(downsam_x, patch_sz=2)  # patch size should match the diffusion patch size, (B,)

        # get rmsc_z from SILoss path (same as training)
        model_kwargs = dict(y=labels)
        loss, proj_loss, rmsc_z = loss_fn(eval_model, x, model_kwargs, zs=zs)  # rmsc_z: (B,)
        rmsc_loss = F.mse_loss(rmsc_x, rmsc_z, reduction="none")  # (B,) per-sample

        # gather to get global accurate sums
        gx = accelerator.gather(rmsc_x.detach()).float().view(-1)
        gz = accelerator.gather(rmsc_z.detach()).float().view(-1)
        gd = accelerator.gather(rmsc_dino.detach()).float().view(-1)
        gl = accelerator.gather(rmsc_loss.detach()).float().view(-1)

        # cap to not exceed eval_num_samples
        remaining = args.eval_num_samples - total_seen
        if gx.numel() > remaining:
            gx = gx[:remaining]
            gz = gz[:remaining]
            gd = gd[:remaining]
            gl = gl[:remaining]

        bs = gx.numel()  # total batch size
        total_seen += bs

        sum_rmsc_x += gx.sum().item()
        sum_rmsc_z += gz.sum().item()
        sum_rmsc_dino += gd.sum().item()
        sum_rmsc_loss += gl.sum().item()

        if accelerator.is_local_main_process:
            progress.update(bs)
            progress.set_postfix(
                seen=total_seen,
                rmsc_x=sum_rmsc_x / max(total_seen, 1),
                rmsc_z=sum_rmsc_z / max(total_seen, 1),
                rmsc_dino=sum_rmsc_dino / max(total_seen, 1),
                rmsc_loss=sum_rmsc_loss / max(total_seen, 1),
            )

    if accelerator.is_local_main_process:
        progress.close()

    accelerator.wait_for_everyone()

    means = {
        "rmsc_x": sum_rmsc_x / args.eval_num_samples,
        "rmsc_z": sum_rmsc_z / args.eval_num_samples,
        "rmsc_dino": sum_rmsc_dino / args.eval_num_samples,
        "rmsc_loss": sum_rmsc_loss / args.eval_num_samples,
        "num_samples": args.eval_num_samples,
        "use_ema": bool(args.use_ema),
        "use_blur": bool(args.gaussian_blur),
    }

    # log/save only on main process
    if accelerator.is_main_process:
        out_path = os.path.join(args.output_dir, f"eval_rmsc.txt")

        # Pretty text format
        lines = [
            f"ckpt_path: {args.ckpt_path}",
            f"rmsc_x: {means['rmsc_x']}",
            f"rmsc_z: {means['rmsc_z']}",
            f"rmsc_dino: {means['rmsc_dino']}",
            f"rmsc_loss: {means['rmsc_loss']}",
            f"num_samples: {means['num_samples']}",
            f"use_blur: {means['use_blur']}",
        ]
        # append if exists, otherwise create
        mode = "a" if os.path.exists(out_path) else "w"
        with open(out_path, mode) as f:
            if mode == "a":
                f.write("\n")
            f.write("\n".join(lines) + "\n")

        logger.info("Evaluation done.")
        logger.info("\n" + "\n".join(lines))
        logger.info(f"Saved: {out_path}")
        logger.info(f"rmsc results: {means}")

    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output_dir", type=str, default="exps")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--fused_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data_dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch_size", type=int, default=256)

    # precision, cpu, seed
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)

    # loss
    parser.add_argument("--path_type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])  # currently we only support v-prediction
    parser.add_argument("--cfg_prob", type=float, default=0.1)
    parser.add_argument("--enc_type", type=str, default='dinov2-vit-b')
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")

    # rmsc evaluation
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--eval_num_samples", type=int, default=50000)
    parser.add_argument("--gaussian_blur", action=argparse.BooleanOptionalAction, default=False,
                        help="gaussian blurring for downsampled image before rmsc eval")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    rmsc_evaluation(args)

    # accelerate launch --multi_gpu --num_machines 1 --num_processes 4 eval_rmsc.py \
    # --mixed_precision fp16 --exp_name linear-dinov2-b-enc4 \
    # --output_dir /data/clusterfs/mld/users/lanliu/mang/REPA_exps/imgnet256_repa \
    # --ckpt_path /data/clusterfs/mld/users/lanliu/mang/REPA_exps/imgnet256_repa/linear-dinov2-b-enc4/checkpoints/0300000.pt \
    # --model SiT-B/2 --data_dir /data/scratch/datasets/imagenet256 \
    # --eval_num_samples 50000 --batch_size 256 --use_ema

