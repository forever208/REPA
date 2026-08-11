#!/bin/bash

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate repa

export HF_HOME=/home/ningmang/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ningmang/torch_cache

#############################
# FID Evaluation
#############################

cd /home/ningmang/REPA
LOG_DIR=/home/ningmang/REPA/REPA_exps/imagenet256_DSM_40k_mean_XL
mkdir -p "${LOG_DIR}"
TIME=$(date +%Y%m%d_%H%M%S)
exec > "${LOG_DIR}/generate_${TIME}.log" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508

torchrun --nnodes=1 --nproc_per_node=8 --master_port="${MASTER_PORT}" generate.py \
--model="SiT-XL/2" --encoder-depth=8 --projector-embed-dims=768 --fused-attn \
--ckpt="${LOG_DIR}/linear-dinov2-b-enc8/checkpoints/1000000.pt" \
--vae-checkpoint="/home/ningmang/vae/exps/imagenet256_DSM_ft/checkpoints/40000.ckpt" \
--latents-scale=0.2228 --latents-bias=-0.2901 \
--global-seed=0 --num-fid-samples=50000 --per-proc-batch-size=32 \
--num-steps=250 --mode=sde --path-type=linear --cfg-scale=1.8 --guidance-high=0.7 \
--fid-stat="/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz" \
--sample-dir="${LOG_DIR}/FID_samples"


# sslaunch submit -j 05lpipsxl-fid -c zhongwei -q stream -n 1 --no-log -- \
# bash imagenet256_VAE_XL_generate.sh
