#!/bin/bash
set -eo pipefail

# The Python user packages belong to ningmang, including in kubectl root shells.
if [[ "$(id -u)" -eq 0 ]]; then
    exec runuser -u ningmang -- bash "${BASH_SOURCE[0]}" "$@"
fi

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate /home/ningmang/miniconda3/envs/REPA

export HF_HOME=/home/ningmang/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ningmang/torch_cache
export PYTHONUNBUFFERED=1

#############################
# Training
#############################

cd /home/ningmang/REPA
DATA_DIR="${DATA_DIR:-/tmp/ningmang/imagenet256_DSM_ft_05lpips_40k}"
VAE_CHECKPOINT=/mnt/vepfs/base2/stream/ningmang/vae_exps/imagenet256_DSM_ft_05lpips/checkpoints/40000.ckpt
FID_STAT=/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz
# Measured on 200,000 samples in extract_logs/20261010_100842/stdout.log.
LATENTS_SCALE=0.21993266813786302
LATENTS_BIAS=-0.3361667795808232

for REQUIRED_FILE in "${DATA_DIR}/images/dataset.json" "${DATA_DIR}/vae-sd/dataset.json" "${VAE_CHECKPOINT}" "${FID_STAT}"; do
    if [[ ! -r "${REQUIRED_FILE}" ]]; then
        echo "Error: required file is not readable: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

LOG_DIR="${LOG_DIR:-/mnt/vepfs/base2/stream/ningmang/REPA_exps/sit_xl/imagenet256_DSM_05lpips_40k_repa0p5}"
mkdir -p "${LOG_DIR}"
TIME=$(date +%Y%m%d_%H%M%S)
echo "Training log: ${LOG_DIR}/${TIME}.log"
exec > "${LOG_DIR}/${TIME}.log" 2>&1
SAMPLE_DIR="/tmp/imagenet256_DSM_XL_train_${TIME}_$$"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29507

python -m accelerate.commands.launch --multi_gpu --num_machines 1 --num_processes 8 \
--main_process_port="${MASTER_PORT}" \
train.py --allow_tf32 --mixed_precision="fp16" --seed=0 --report_to none \
--path_type="linear" --prediction="v" --weighting="uniform" \
--model="SiT-XL/2" --enc_type="dinov2-vit-b" --encoder_depth=8 --exp_name="linear-dinov2-b-enc8" \
--checkpointing_steps=500000 --max_train_steps=1000001 --output_dir="$LOG_DIR" \
--num_samples=50000 --num_steps=100 --mode=sde --cfg_scale=2.0 --guidance_high=0.7 \
--fid_stat="${FID_STAT}" \
--sample_path="${SAMPLE_DIR}" --num_workers 4 --pre_fetch 2 \
--data_dir="${DATA_DIR}" --vae-checkpoint="${VAE_CHECKPOINT}" \
--proj_coeff=0.5 --latents_scale="${LATENTS_SCALE}" --latents_bias="${LATENTS_BIAS}"

# Run in the container holding DATA_DIR (or set DATA_DIR to a shared copy).
# nohup bash /home/ningmang/REPA/imagenet256_DSM_XL.sh > /tmp/imagenet256_DSM_XL_launch.log 2>&1 &
