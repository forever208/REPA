#!/bin/bash

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate repa

export HF_HOME=/home/ningmang/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ningmang/torch_cache

#############################
# Training
#############################

cd /home/ningmang/REPA
LOG_DIR=/home/ningmang/REPA/REPA_exps/imagenet256_repa_sdvae_40k
mkdir -p ${LOG_DIR}
TIME=$(date +%Y%m%d_%H%M%S)
exec > ${LOG_DIR}/${TIME}.log 2>&1

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29501

accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
train.py --allow_tf32 --mixed_precision="fp16" --seed=0 --report_to none \
--path_type="linear" --prediction="v" --weighting="uniform" \
--model="SiT-B/2" --enc_type="dinov2-vit-b" --encoder_depth=4 \
--checkpointing_steps=100000 --output_dir="$LOG_DIR" \
--exp_name="linear-dinov2-b-enc4" --data_dir="/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_sdvae_40k" \
--num_samples=50000 --num_steps=100 --mode=sde --cfg_scale=1.8 --guidance_high=0.7 \
--fid_stat="/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz" \
--sample_path="$LOG_DIR/FID_samples" \
--num_workers=8 --proj_coeff=0.5

# nohup bash imagenet256_repa.sh > /dev/null 2>&1 &


# data preprocessing
torchrun --nproc_per_node=4 preprocess.py \
--source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
--dest=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_repa/vae-sd \
--dest-images=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_repa/images \
--batch-size=128 \
--resolution=256 \
--transform=center-crop-dhariwal