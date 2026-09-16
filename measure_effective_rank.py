import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CustomDataset
from models.sit import SiT_models


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Measure all-stage SiT patch-token spatial effective rank"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--in-out",
        "--in_out",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--model", choices=tuple(SiT_models), default="SiT-XL/2")
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument("--encoder-depth", default=8, type=int)
    parser.add_argument("--projector-embed-dims", default="768")
    parser.add_argument(
        "--fused-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--latents-scale", default=0.22506, type=float)
    parser.add_argument("--latents-bias", default=0.3279, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-images", default=1000, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--cls-idx",
        type=int,
        default=None,
        help="Optional ImageNet class index; sample all classes when omitted.",
    )
    parser.add_argument("--num-timesteps", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--heatmap-pr-max",
        "--heatmap_pr_max",
        type=float,
        default=None,
        help="Heatmap PR upper bound; defaults to the measured maximum.",
    )
    parser.add_argument(
        "--heatmap-gamma",
        "--heatmap_gamma",
        type=float,
        default=0.8,
        help="PowerNorm gamma used by the heatmap.",
    )
    parser.add_argument("--output", default="token_spatial_effective_rank.png")
    return parser


def _init_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return distributed, rank, world_size, local_rank


def _spatial_effective_rank(tokens):
    """Return centered spatial participation ratio for each image."""
    centered = tokens - tokens.mean(dim=1, keepdim=True)
    gram = centered @ centered.transpose(1, 2)
    numerator = centered.square().sum(dim=(1, 2)).square()
    denominator = gram.square().sum(dim=(1, 2))
    ranks = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(denominator),
    )
    return ranks.clamp(min=0.0, max=float(tokens.size(1) - 1))


def _latent_patch_tokens(latents, patch_size):
    """Convert BCHW latents to row-major flattened patch tokens."""
    patch_height, patch_width = patch_size
    batch_size, channels, _, _ = latents.shape
    patches = latents.unfold(
        2, patch_height, patch_height
    ).unfold(3, patch_width, patch_width)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size, -1, channels * patch_height * patch_width
    )


def _sample_indices(dataset, num_images, cls_idx, seed):
    if cls_idx is None:
        candidate_indices = list(range(len(dataset)))
        sample_source = "all classes"
    else:
        if not 0 <= cls_idx < 1000:
            raise ValueError(f"--cls-idx must be in [0, 999], got {cls_idx}")
        candidate_indices = np.flatnonzero(dataset.labels == cls_idx).tolist()
        sample_source = f"class {cls_idx}"

    selected_count = min(num_images, len(candidate_indices))
    if selected_count == 0:
        raise RuntimeError(f"No samples found for {sample_source}")
    selected_indices = random.Random(seed).sample(
        candidate_indices, k=selected_count
    )
    return selected_indices, sample_source


def _save_rank_heatmap(
    timesteps,
    stage_labels,
    stage_ranks,
    output_path,
    pr_max=None,
    gamma=0.8,
):
    """Save one PowerNorm-scaled t-by-stage spatial-rank heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = np.asarray(stage_ranks, dtype=np.float64).T
    color_max = float(heatmap.max()) if pr_max is None else pr_max
    color_max = max(color_max, np.finfo(np.float64).eps)
    figure_width = max(10.0, 0.48 * len(stage_labels))
    fig, axis = plt.subplots(figsize=(figure_width, 6.5))
    image = axis.imshow(
        heatmap,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=color_max),
    )

    axis.set_title("All-stage patch-token spatial effective rank")
    axis.set_xlabel("Stage")
    axis.set_ylabel("t")
    axis.set_xticks(np.arange(len(stage_labels)))
    axis.set_xticklabels(stage_labels, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(timesteps)))
    axis.set_yticklabels([f"{timestep:.2f}" for timestep in timesteps])
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(
        f"Mean spatial participation ratio (PowerNorm γ={gamma:g})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SiT-XL/2 analysis")
    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")
    if args.latents_scale <= 0:
        raise ValueError("--latents-scale must be positive")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.num_timesteps < 2:
        raise ValueError("--num-timesteps must be at least 2")
    if args.heatmap_pr_max is not None and args.heatmap_pr_max <= 0:
        raise ValueError("--heatmap-pr-max must be positive")
    if args.heatmap_gamma <= 0:
        raise ValueError("--heatmap-gamma must be positive")

    distributed, rank, world_size, local_rank = _init_distributed()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    latent_size = args.resolution // 8
    z_dims = [int(value) for value in args.projector_embed_dims.split(",")]
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        fused_attn=args.fused_attn,
        qk_norm=args.qk_norm,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if args.weights not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {args.weights!r}; available keys: "
            f"{sorted(checkpoint)}"
        )
    model.load_state_dict(checkpoint[args.weights], strict=True)
    del checkpoint
    model.to(device).eval().requires_grad_(False)

    dataset = CustomDataset(args.data_dir)
    selected_indices, sample_source = _sample_indices(
        dataset, args.num_images, args.cls_idx, args.seed
    )
    local_indices = selected_indices[rank::world_size]
    data_loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    stage_specs = [
        ("latent_patches", "Noisy latent patches", "Latent"),
        ("x_embedder", "x_embedder(x)", "Embed"),
        ("x_embedder_pos", "x_embedder(x) + pos_embed", "Embed"),
        *[
            (
                f"layer{layer}",
                f"Transformer block {layer} output",
                f"L{layer}",
            )
            for layer in range(len(model.blocks))
        ],
        ("final_norm", "final_layer.norm_final(x)", "Final norm"),
        ("final_modulate", "final_layer modulate(...)", "Final mod"),
        ("final_linear", "final_layer.linear(x)", "Linear"),
    ]
    spatial_dim = model.x_embedder.num_patches
    captured_ranks = {}

    def record_rank(stage_key, tokens):
        if tokens.size(1) != spatial_dim:
            raise RuntimeError(
                f"{stage_key} produced {tokens.size(1)} spatial tokens, "
                f"expected {spatial_dim}"
            )
        with torch.autocast(device_type="cuda", enabled=False):
            captured_ranks[stage_key] = _spatial_effective_rank(tokens.float())

    def capture_latent_patches(_module, inputs):
        tokens = _latent_patch_tokens(
            inputs[0], model.x_embedder.patch_size
        )
        record_rank("latent_patches", tokens)

    def capture_x_embedder(_module, _inputs, output):
        record_rank("x_embedder", output)

    def capture_x_embedder_pos(_module, inputs):
        record_rank("x_embedder_pos", inputs[0])

    def capture_final_norm(_module, _inputs, output):
        record_rank("final_norm", output)

    def capture_final_modulate(_module, inputs):
        record_rank("final_modulate", inputs[0])

    def capture_final_linear(_module, _inputs, output):
        record_rank("final_linear", output)

    def make_capture_block(layer):
        stage_key = f"layer{layer}"

        def capture_block(_module, _inputs, output):
            record_rank(stage_key, output)

        return capture_block

    hooks = [
        model.x_embedder.register_forward_pre_hook(capture_latent_patches),
        model.x_embedder.register_forward_hook(capture_x_embedder),
        model.blocks[0].register_forward_pre_hook(capture_x_embedder_pos),
        *[
            block.register_forward_hook(make_capture_block(layer))
            for layer, block in enumerate(model.blocks)
        ],
        model.final_layer.norm_final.register_forward_hook(
            capture_final_norm
        ),
        model.final_layer.linear.register_forward_pre_hook(
            capture_final_modulate
        ),
        model.final_layer.linear.register_forward_hook(capture_final_linear),
    ]

    timesteps = torch.linspace(
        0.0, 1.0, args.num_timesteps, device=device
    )
    rank_sums = torch.zeros(
        len(stage_specs),
        args.num_timesteps,
        dtype=torch.float64,
        device=device,
    )
    image_counts = torch.zeros(
        args.num_timesteps, dtype=torch.int64, device=device
    )
    autocast_enabled = args.mixed_precision != "no"
    autocast_dtype = (
        torch.float16
        if args.mixed_precision == "fp16"
        else torch.bfloat16
    )

    if rank == 0:
        print(
            "Starting all-stage spatial effective-rank measurement: "
            f"weights={args.weights}, images={len(selected_indices)}, "
            f"source={sample_source}, timesteps={args.num_timesteps}, "
            f"stages={len(stage_specs)}, spatial_dim={spatial_dim}, "
            f"world_size={world_size}, "
            f"batches_per_rank={len(data_loader)}",
            flush=True,
        )

    latents_scale = torch.full(
        (1, 4, 1, 1), args.latents_scale, device=device
    )
    latents_bias = torch.full(
        (1, 4, 1, 1), args.latents_bias, device=device
    )

    try:
        with torch.inference_mode():
            progress = tqdm(
                data_loader,
                desc="Measuring spatial effective rank",
                unit="batch",
                dynamic_ncols=True,
                disable=rank != 0,
            )
            for _raw_images, moments, labels in progress:
                moments = moments.squeeze(dim=1).to(
                    device, non_blocking=True
                )
                labels = labels.to(device, non_blocking=True)
                mean, std = torch.chunk(moments, 2, dim=1)
                clean_latents = (
                    mean + std * torch.randn_like(mean) - latents_bias
                ) * latents_scale
                noise = torch.randn_like(clean_latents)

                for timestep_index, timestep in enumerate(timesteps):
                    noisy_latents = (
                        (1.0 - timestep) * clean_latents + timestep * noise
                    )
                    captured_ranks.clear()
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        model(
                            noisy_latents,
                            timestep.expand(noisy_latents.size(0)),
                            labels,
                        )

                    missing_stages = [
                        stage_key
                        for stage_key, _, _ in stage_specs
                        if stage_key not in captured_ranks
                    ]
                    if missing_stages:
                        raise RuntimeError(
                            f"Hooks did not capture stages: {missing_stages}"
                        )
                    for stage_index, (stage_key, _, _) in enumerate(
                        stage_specs
                    ):
                        ranks = captured_ranks[stage_key]
                        rank_sums[stage_index, timestep_index] += (
                            ranks.double().sum()
                        )
                    image_counts[timestep_index] += noisy_latents.size(0)
    finally:
        for hook in hooks:
            hook.remove()

    if distributed:
        dist.all_reduce(rank_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(image_counts, op=dist.ReduceOp.SUM)

    if rank == 0:
        if (image_counts == 0).any():
            raise RuntimeError("No images measured at one or more timesteps")
        stage_ranks = (
            rank_sums / image_counts.unsqueeze(0)
        ).cpu().tolist()
        timestep_values = timesteps.cpu().tolist()
        print(
            f"mode=all_stages weights={args.weights} "
            f"images={len(selected_indices)} spatial_dim={spatial_dim} "
            f"source={sample_source} "
            "aggregation=mean_per_image_spatial_pr"
        )
        for (stage_key, stage_label, _), ranks in zip(
            stage_specs, stage_ranks
        ):
            print(f"stage={stage_key} label={stage_label}")
            for timestep, effective_rank in zip(timestep_values, ranks):
                print(
                    f"t={timestep:.6f} "
                    f"effective_rank={effective_rank:.6f}"
                )

        plotted_stages = [
            (short_label, ranks)
            for (stage_key, _, short_label), ranks in zip(
                stage_specs, stage_ranks
            )
            if stage_key
            not in {"x_embedder", "final_norm", "final_modulate"}
        ]
        output_path = Path(args.output)
        _save_rank_heatmap(
            timestep_values,
            [label for label, _ in plotted_stages],
            [ranks for _, ranks in plotted_stages],
            output_path,
            pr_max=args.heatmap_pr_max,
            gamma=args.heatmap_gamma,
        )
        print(f"Saved all-stage heatmap to {output_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
