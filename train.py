import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
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
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
wandb.init(mode="offline")
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from samplers import euler_sampler, euler_maruyama_sampler
from PIL import Image
import shutil
from tools.fid_score import calculate_fid_given_paths
from utils import gaussian_blur, downsample_to, rmsc, rmsc_tokens

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

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


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


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
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator and log
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json") # Save to a JSON file
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
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
    
    # Create DINO, SiT, VAE:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, args.resolution)
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
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
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
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
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}', map_location='cpu',
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )
        
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)  # (batch, tokens, dim)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

                rmsc_dino = rmsc_tokens(zs[0])

            grad_norm = torch.tensor(0.0, device=device)
            with accelerator.accumulate(model):
                # rmsc
                _, _, hx, wx = raw_image.shape  # (batch, c, h, w) unit8, value range [0, 255]
                raw_image = raw_image.float() / 127.5 - 1.0  # value range [-1, 1] before rmsc eval
                if args.gaussian_blur:
                    raw_image = gaussian_blur(raw_image, kernel_size=7, sigma=1.2)
                downsam_x = downsample_to(raw_image, (hx//8, wx//8))
                rmsc_x = rmsc(downsam_x, patch_sz=1)  # (batch, )

                model_kwargs = dict(y=labels)
                loss, proj_loss, rmsc_z = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                rmsc_loss = F.mse_loss(rmsc_x, rmsc_z)  # scaler
                loss = loss_mean + proj_loss_mean * args.proj_coeff + rmsc_loss * args.rmsc_weight
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### save ckpt
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

            ### logging info
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "rmsc_x": accelerator.gather(rmsc_x).mean().detach().item(),
                "rmsc_z": accelerator.gather(rmsc_z).mean().detach().item(),
                "rmsc_dino": accelerator.gather(rmsc_dino).mean().detach().item(),
                "rmsc_loss": accelerator.gather(rmsc_loss).mean().detach().item(),
            }
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process and (global_step == 1 or global_step % args.log_every == 0):
                logger.info(
                    f"step={global_step} "
                    f"loss={logs['loss']:.6f} proj_loss={logs['proj_loss']:.6f} grad_norm={logs['grad_norm']:.3f} "
                    f"rmsc_x={logs['rmsc_x']:.6f} rmsc_z={logs['rmsc_z']:.6f} rmsc_dino={logs['rmsc_dino']:.6f} rmsc_loss={logs['rmsc_loss']:.6f}"
                )
                accelerator.log(logs, step=global_step)  # wandb log

            ### FID evaluation
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    os.makedirs(args.sample_path, exist_ok=True)
                    logger.info(f"FID evaluation at steps {global_step}, samples are in {args.sample_path}")

                ema.eval()
                assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

                # generate samples and save into the folder
                num_iterations = int(args.num_samples // args.batch_size + 1)
                mini_batch_size = args.batch_size // accelerator.num_processes
                world_size = accelerator.state.num_processes
                local_rank = accelerator.state.local_process_index
                in_channels = getattr(ema, "in_channels")

                for i in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
                    z = torch.randn(mini_batch_size, in_channels, latent_size, latent_size, device=device)
                    y = torch.randint(0, args.num_classes, (mini_batch_size,), device=device)
                    sampling_kwargs = dict(
                        model=ema,  # use ema model for sampling
                        latents=z,
                        y=y,
                        num_steps=args.num_steps,
                        heun=args.heun,
                        cfg_scale=args.cfg_scale,
                        guidance_low=args.guidance_low,
                        guidance_high=args.guidance_high,
                        path_type=args.path_type,
                    )

                    with torch.no_grad():
                        if args.mode == "sde":
                            samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                        elif args.mode == "ode":
                            samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                        else:
                            raise NotImplementedError()

                        latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215, ]).view(1, 4, 1, 1).to(device)
                        latents_bias = -torch.tensor([0., 0., 0., 0., ]).view(1, 4, 1, 1).to(device)
                        samples = vae.decode((samples - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                        for b_id, sample in enumerate(samples):  # distributed image save
                            img_id = i * mini_batch_size * world_size + local_rank * mini_batch_size + b_id
                            if img_id >= args.num_samples:
                                break
                            Image.fromarray(sample).save(f"{args.sample_path}/{img_id:06d}.png")

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _fid = calculate_fid_given_paths((args.fid_stat, args.sample_path))
                    logger.info(f'step={global_step} fid{args.num_samples}={_fid}')
                    with open(os.path.join(args.output_dir, f'eval_{args.mode}_{args.num_samples}.log'), 'a') as f:
                        print(f'step={global_step} fid{args.num_samples}={_fid}', file=f)
                    shutil.rmtree(args.sample_path)  # remove all generated images

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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output_dir", type=str, default="exps")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--sampling_steps", type=int, default=10000)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--fused_attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk_norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data_dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch_size", type=int, default=256)

    # precision
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max_train_steps", type=int, default=400000)
    parser.add_argument("--checkpointing_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num_workers", type=int, default=8)

    # loss
    parser.add_argument("--path_type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg_prob", type=float, default=0.1)
    parser.add_argument("--enc_type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj_coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rmsc_weight", type=float, default=1.0)
    parser.add_argument("--gaussian_blur", action=argparse.BooleanOptionalAction, default=False, help="gaussian blurring for downsampled image before rmsc eval")

    # FID evaluation
    parser.add_argument("--sample_path", type=str, default=None, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg_scale", type=float, default=1.8)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)  # only for ode
    parser.add_argument("--guidance_low", type=float, default=0.)
    parser.add_argument("--guidance_high", type=float, default=0.7)
    parser.add_argument("--fid_stat", type=str, default=None, required=True)


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
