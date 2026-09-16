#!/usr/bin/env bash

set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate repa

export HF_HOME=/home/ningmang/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ningmang/torch_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29509

cd /home/ningmang/REPA

EXP_DIR=/home/ningmang/REPA/REPA_exps/imagenet256_SDVAE_40k_stdx1_XL
CKPT="$EXP_DIR/linear-dinov2-b-enc8/checkpoints/1000000.pt"
DATA_DIR=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_SDVAE_40k
CLS_IDX=0
NUM_IMAGES=1000

TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$EXP_DIR/analysis/$TIME"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/stdout.log") \
  2> >(tee -a "$OUTPUT_DIR/stderr.log" >&2)

echo "stdout log: $OUTPUT_DIR/stdout.log"
echo "stderr log: $OUTPUT_DIR/stderr.log"

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  measure_effective_rank.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" \
  --model "SiT-XL/2" \
  --encoder-depth 8 \
  --projector-embed-dims 768 \
  --fused-attn \
  --no-qk-norm \
  --weights ema \
  --resolution 256 \
  --latents-scale 0.2474 \
  --latents-bias -0.2229 \
  --num-images "$NUM_IMAGES" \
  --cls-idx "$CLS_IDX" \
  --batch-size 16 \
  --num-workers 4 \
  --mixed-precision fp16 \
  --output "$OUTPUT_DIR/token_spatial_PR_cls${CLS_IDX}_all_stages.png" \
  --heatmap-pr-max 120 --heatmap-gamma 0.4

# sslaunch submit -j rank -c zhongwei -q stream -n 1 --no-log -- bash measure_effective_rank.sh
# sslaunch submit -j rank -q stream -n 1 --no-log -- bash measure_effective_rank.sh
