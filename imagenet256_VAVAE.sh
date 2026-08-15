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
LOG_DIR=/home/ningmang/REPA/REPA_exps/imagenet256_VAVAE_mean
mkdir -p ${LOG_DIR}
TIME=$(date +%Y%m%d_%H%M%S)
exec > ${LOG_DIR}/${TIME}.log 2>&1

export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29047

accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
train_vavae.py --allow_tf32 --mixed_precision="fp16" --seed=0 --report_to none \
--path_type="linear" --prediction="v" --weighting="uniform" \
--model="SiT-B/1" --enc_type="dinov2-vit-b" --encoder_depth=4 --exp_name="linear-dinov2-b-enc4" \
--checkpointing_steps=100000 --output_dir="$LOG_DIR" \
--num_samples=50000 --num_steps=100 --mode=sde --cfg_scale=2.0 --guidance_high=0.7 \
--fid_stat="/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz" \
--sample_path="$LOG_DIR/FID_samples" --num_workers 4 --pre_fetch 2 \
--data_dir="/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_VAVAE" \
--vae-checkpoint="/home/ningmang/vae/exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt" \
--proj_coeff=0.5 --latents_scale=0.2837 --latents_bias=0.1900 \

# sslaunch submit -j ningmang -n 1 -c zhongwei -q stream -- sleep infinity
# kubectl --kubeconfig /mnt/vepfs/base/infra/.kube/zhongweiconfig \
# -n default exec -it ningmang-ningmang-n1-260731090654-master-0 -- bash

# nohup bash imagenet256_VAVAE.sh > /dev/null 2>&1 &
# pkill -TERM -f stage1_vae_trainer.py
# sudo chown -R ningmang:ningmang /home/ningmang/REPA