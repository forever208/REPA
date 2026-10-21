# REPA Supporting Multiple VAEs and Online FID Evaluation

This repository is built on the official [REPA implementation](https://github.com/sihyun-yu/REPA) of
[Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940).
REPA aligns diffusion transformer features with representations from pretrained visual encoders to improve diffusion training.

This fork adds support for **SD-VAE, EQ-VAE, DSM, and VA-VAE**, along with **online FID evaluation during training** and standalone distributed FID evaluation. The examples below train and evaluate class-conditional SiT models on ImageNet at 256 × 256 resolution.

| VAE | Latent shape at 256 × 256 (C × H × W) | Training entry point | Evaluation entry point | Example SiT models |
| --- | --- | --- | --- | --- |
| SD-VAE | 4 × 32 × 32 | `train.py` | `generate.py` | `SiT-B/2`, `SiT-XL/2` |
| EQ-VAE | 4 × 32 × 32 | `train.py` | `generate.py` | `SiT-B/2`, `SiT-XL/2` |
| DSM | 4 × 32 × 32 | `train.py` | `generate.py` | `SiT-B/2`, `SiT-XL/2` |
| VA-VAE (f16d32) | 32 × 16 × 16 | `train_vavae.py` | `generate_vavae.py` | `SiT-B/1`, `SiT-XL/1` |

The VAE is frozen during REPA training. VAE training, checkpoint preparation, and latent extraction are covered in [forever208/vae](https://github.com/forever208/vae).



## 1. Environment setup

```bash
git clone https://github.com/forever208/REPA.git
cd REPA

conda create -n repa python=3.13 -y
conda activate repa
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins PyTorch 2.12.0 and torchvision 0.27.0 with CUDA 13.0 wheels. Use an NVIDIA driver compatible with these wheels. The training examples use four or eight GPUs; the evaluation example uses eight. Set the visible devices and process count to match your machine.

The examples use DINOv2 ViT-B, downloaded through `torch.hub` on first use. FID evaluation also downloads Inception weights on first use. Allow network access for these downloads or prepare the Torch cache in advance. Logging to Weights & Biases is optional: the commands use `--report_to none`; use `--report_to wandb` after `wandb login` to enable it.

Run the training and evaluation commands from the REPA repository root. The checked-in shell scripts contain machine-specific paths and environment settings; the commands below expose the paths you need to replace.



## 2. Dataset preparation

Follow the [data preparation guide in forever208/vae](https://github.com/forever208/vae#3-data-preparation) to download and unpack ImageNet. Latent extraction expects the training images in class subdirectories:

```text
/path/to/imagenet/train/
├── n01440764/
│   ├── n01440764_10026.JPEG
│   └── ...
├── n01443537/
└── ...
```

The extractor resizes and center-crops the original images to 256 × 256. Images do not need to be resized on disk beforehand.

### Extract images and VAE posterior statistics

Use `extract_REPA_latents.py` from [forever208/vae](https://github.com/forever208/vae#extract-repa-latents), with that repository's environment. Starting from the REPA repository root:

```bash
git clone https://github.com/forever208/vae.git ../vae
cd ../vae
conda env create -f environment.yaml
conda activate vae

IMAGENET_TRAIN="/path/to/imagenet/train"
DATA_DIR="/path/to/imagenet256_DSM_40k"
VAE_CONFIG="/path/to/vae_run/resolved_config.yaml"
VAE_CHECKPOINT="/path/to/vae_run/checkpoints/40000.ckpt"

torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
  --task=extract \
  --source="$IMAGENET_TRAIN" \
  --dest="$DATA_DIR/vae-sd" \
  --dest-images="$DATA_DIR/images" \
  --config="$VAE_CONFIG" \
  --checkpoint="$VAE_CHECKPOINT" \
  --resolution=256 --transform=center-crop-dhariwal \
  --batch-size=128
```

Select the configuration and checkpoint for your VAE. A trained VAE run's `resolved_config.yaml` records its configuration; pretrained models can use their matching configuration from the VAE repository. Extraction requires CUDA and uses a batch size of 128 per GPU in this example. For one GPU, use `torchrun --standalone --nproc_per_node=1`.

The resulting dataset has this layout:

```text
imagenet256_DSM_40k/
├── images/
│   ├── dataset.json
│   └── 00000/
│       └── img00000000.png
└── vae-sd/
    ├── dataset.json
    └── 00000/
        └── img-mean-std-00000000.npy
```

Pass the **parent directory containing both `images/` and `vae-sd/`** to `--data_dir`. Keep the name `vae-sd/` for every VAE variant: the dataset loader uses this fixed directory name. Each `.npy` file contains the posterior **mean and standard deviation**, concatenated along the channel dimension. Images and latent files must remain paired, and `vae-sd/dataset.json` supplies their class labels. REPA uses the images for representation alignment and samples diffusion inputs from the saved posterior statistics.


### Measure latent normalization

In the VAE repository, using the same configuration and checkpoint, estimate the global statistics over sampled latents:

```bash
torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
  --task=stats \
  --source="$IMAGENET_TRAIN" \
  --config="$VAE_CONFIG" \
  --checkpoint="$VAE_CHECKPOINT" \
  --resolution=256 --transform=center-crop-dhariwal \
  --batch-size=128 --num-stat-samples=200000

cd ../REPA
conda activate repa
```

Use the reported `mean` as `latents_bias` and `scaling_factor` (`1 / std`) as `latents_scale`. REPA normalizes latents as `z_normalized = (z - latents_bias) * latents_scale` and reverses this transformation before VAE decoding. **Use the same VAE checkpoint for extraction, training, and evaluation, and keep normalization identical between training and evaluation.** The exported posterior statistics are unnormalized. Extract a separate latent dataset when changing the VAE checkpoint.

### FID reference statistics (included)

The repository includes [asset/fid_stats_imagenet256_guided_diffusion.npz](asset/fid_stats_imagenet256_guided_diffusion.npz) for both online and standalone FID evaluation. No separate statistics download is needed. All commands below use this path relative to the REPA repository root.

The file contains the original Inception feature mean `mu` (2048,) and covariance `sigma` (2048, 2048), both `float64`, extracted from the [official guided-diffusion ImageNet-256 reference archive](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) listed in its [evaluation guide](https://github.com/openai/guided-diffusion/tree/main/evaluations). These two `.npy` entries were repackaged into a compressed `.npz`, preserving their original contents byte for byte. Use this same reference file when comparing VAE variants and training runs.

File size: 31,918,745 bytes (30.44 MiB). SHA-256:

```text
99aaa9d701b732a7aabf2751c514a20babf16f6cebbd04a7816b8ddea012b9ea
```

## 3. Training

The commands below retain the core settings from [imagenet256_DSM.sh](imagenet256_DSM.sh) and [imagenet256_DSM_XL.sh](imagenet256_DSM_XL.sh). The companion [imagenet256_DSM_XL_generate.sh](imagenet256_DSM_XL_generate.sh) evaluates the XL checkpoint; its settings are described in the evaluation section.

The B and XL scripts use **different DSM VAE checkpoints**. Their normalization values below belong to those specific checkpoints. Replace the dataset and checkpoint paths, and use your measured normalization if training with another checkpoint.

### DSM + SiT-B/2

This example uses four GPUs, DINOv2 ViT-B alignment at depth 4, and the DSM VAE checkpoint:

```bash
DATA_DIR="/path/to/imagenet256_DSM_40k"
VAE_CHECKPOINT="/path/to/VAE_ckpt"
FID_STAT="asset/fid_stats_imagenet256_guided_diffusion.npz"
OUTPUT_DIR="exps/imagenet256_DSM_B"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m accelerate.commands.launch \
  --multi_gpu --num_machines=1 --num_processes=4 --main_process_port=29003 \
  train.py \
  --model="SiT-B/2" --enc_type="dinov2-vit-b" --encoder_depth=4 \
  --exp_name="linear-dinov2-b-enc4" --output_dir="$OUTPUT_DIR" \
  --data_dir="$DATA_DIR" --vae-checkpoint="$VAE_CHECKPOINT" \
  --latents_scale=0.2124 --latents_bias=-0.3952 \
  --path_type=linear --prediction=v --weighting=uniform --proj_coeff=0.5 \
  --batch_size=256 --num_workers=4 --pre_fetch=2 \
  --allow_tf32 --mixed_precision=fp16 --seed=0 --report_to=none \
  --checkpointing_steps=100000 --max_train_steps=400001 \
  --num_samples=50000 --num_steps=100 --mode=sde \
  --cfg_scale=2.0 --guidance_high=0.7 \
  --fid_stat="$FID_STAT" --sample_path="$OUTPUT_DIR/fid_samples"
```

### DSM + SiT-XL/2

This example uses eight GPUs, alignment at depth 8, and the DSM VAE checkpoint. It matches the setup evaluated by `imagenet256_DSM_XL_generate.sh`:

```bash
DATA_DIR="/path/to/imagenet256_DSM_40k"
VAE_CHECKPOINT="/path/to/VAE_ckpt"
FID_STAT="asset/fid_stats_imagenet256_guided_diffusion.npz"
OUTPUT_DIR="exps/imagenet256_DSM_XL"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch \
  --multi_gpu --num_machines=1 --num_processes=8 --main_process_port=29507 \
  train.py \
  --model="SiT-XL/2" --enc_type="dinov2-vit-b" --encoder_depth=8 \
  --exp_name="linear-dinov2-b-enc8" --output_dir="$OUTPUT_DIR" \
  --data_dir="$DATA_DIR" --vae-checkpoint="$VAE_CHECKPOINT" \
  --latents_scale=0.21993266813786302 --latents_bias=-0.3361667795808232 \
  --path_type=linear --prediction=v --weighting=uniform --proj_coeff=0.5 \
  --batch_size=256 --num_workers=4 --pre_fetch=2 \
  --allow_tf32 --mixed_precision=fp16 --seed=0 --report_to=none \
  --checkpointing_steps=500000 --max_train_steps=1000001 \
  --num_samples=50000 --num_steps=100 --mode=sde \
  --cfg_scale=2.0 --guidance_high=0.7 \
  --fid_stat="$FID_STAT" --sample_path="$OUTPUT_DIR/fid_samples"
```

`--batch_size` is the global batch size and should be divisible by the number of processes. Training uses fused attention and disables QK normalization by default. For SD-VAE or EQ-VAE, use `train.py` with the corresponding latent dataset, VAE checkpoint, and normalization. For VA-VAE, use `train_vavae.py` with a `/1` model, as in [imagenet256_VAVAE.sh](imagenet256_VAVAE.sh) and [imagenet256_VAVAE_XL.sh](imagenet256_VAVAE_XL.sh).

### Online FID and saved outputs

At every `--checkpointing_steps` interval, training saves a checkpoint, generates `--num_samples` images with the **EMA model**, decodes them with the selected VAE, computes FID, and resumes training. The examples evaluate 50,000 images with 100 SDE steps, CFG 2.0, and a guidance time window of [0.0, 0.7].

- Run configuration: `$OUTPUT_DIR/$EXP_NAME/args.json`.
- Checkpoints: `$OUTPUT_DIR/$EXP_NAME/checkpoints/<step padded to 7 digits>.pt`.
- Training log: `$OUTPUT_DIR/$EXP_NAME/log.txt`.
- Online FID results: `$OUTPUT_DIR/eval_sde_50000.log` for these examples.

Here, `$EXP_NAME` is the value of `--exp_name`. The B example saves through `0400000.pt`; the XL example saves through `1000000.pt`. The step limits above reproduce the shell scripts; checkpoint saving and FID follow the checkpoint interval.

Use a dedicated `--sample_path` for each training run: this temporary image directory is removed after each FID evaluation.


## 4. Evaluation

The core command below comes from [imagenet256_DSM_generate.sh](imagenet256_DSM_generate.sh). It loads the **EMA weights** from the B model's 400k checkpoint and computes FID over 50,000 generated images at one CFG setting:

```bash
CKPT="exps/imagenet256_DSM_B/linear-dinov2-b-enc4/checkpoints/0400000.pt"
VAE_CHECKPOINT="/path/to/VAE_ckpt"
FID_STAT="asset/fid_stats_imagenet256_guided_diffusion.npz"
SAMPLE_DIR="samples/imagenet256_DSM_B"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=8 --master_port=29508 \
  generate.py \
  --model="SiT-B/2" --encoder-depth=4 --projector-embed-dims=768 \
  --fused-attn --no-qk-norm --tf32 --resolution=256 --num-classes=1000 \
  --ckpt="$CKPT" --vae-checkpoint="$VAE_CHECKPOINT" \
  --latents-scale=0.2124 --latents-bias=-0.3952 \
  --global-seed=0 --num-fid-samples=50000 --per-proc-batch-size=32 \
  --num-steps=100 --mode=sde --path-type=linear \
  --cfg-scale=2.0 --guidance-low=0.0 --guidance-high=0.7 \
  --fid-stat="$FID_STAT" --sample-dir="$SAMPLE_DIR"
```

The command prints `fid50000=<value>` and removes its generated image subdirectory after computing FID. It does not retain an image archive. `--per-proc-batch-size` is the batch size per GPU. Standalone evaluation needs the SiT checkpoint, matching VAE checkpoint, and FID reference statistics; it does not load the training dataset.

The full `imagenet256_DSM_generate.sh` script sweeps CFG scales `1.0, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0` and records results in `$LOG_DIR/FID_cond.log`. Adjust its machine-specific paths before running it. The core command above evaluates only CFG 2.0.



## Acknowledgements and citation

We thank the authors of [REPA](https://github.com/sihyun-yu/REPA) for the original implementation, VAE preparation and latent extraction use [forever208/vae](https://github.com/forever208/vae).

If you find this repo helpful, please cite:

```bibtex
@article{ning2026spectrum,
  title={Spectrum matching: a unified perspective for superior diffusability in latent diffusion},
  author={Ning, Mang and Li, Mingxiao and Zhang, Le and Liu, Lanmiao and Blaschko, Matthew B and Salah, Albert Ali and Ertugrul, Itir Onal},
  journal={arXiv preprint arXiv:2603.14645},
  year={2026}
}
```
