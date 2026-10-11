#!/bin/bash
set -eo pipefail

# kubectl exec opens a root shell; the Python user packages belong to ningmang.
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
# Parent of the images/ and vae-sd/ outputs in /home/ningmang/vae/extract_REPA_latents.sh.
DATA_DIR="${DATA_DIR:-/tmp/ningmang/imagenet256_DSM_ft_05lpips_10epochs_76k}"
VAE_CHECKPOINT=/mnt/vepfs/base2/stream/ningmang/vae_exps/imagenet256_DSM_ft_05lpips_10epochs/checkpoints/76000.ckpt
FID_STAT=/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz
# Same VAE checkpoint and normalization as imagenet256_DSM_05lpips_76k_stdx1_XL/linear-dinov2-b-enc8/args.json.
LATENTS_SCALE=0.2124
LATENTS_BIAS=-0.3952

if [[ ! -d "${DATA_DIR}/images" || ! -r "${DATA_DIR}/vae-sd/dataset.json" ]]; then
    echo "Error: extracted images/ and vae-sd/dataset.json are required under ${DATA_DIR}." >&2
    echo "Run after extraction in the same container, or set DATA_DIR to a shared copy of both directories." >&2
    exit 1
fi
for REQUIRED_FILE in "${VAE_CHECKPOINT}" "${FID_STAT}"; do
    if [[ ! -r "${REQUIRED_FILE}" ]]; then
        echo "Error: required file is not readable: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

LOG_DIR="${LOG_DIR:-/mnt/vepfs/base2/stream/ningmang/REPA_exps/sit_b/imagenet256_DSM_05lpips_76k_stdx1}"
mkdir -p "${LOG_DIR}"
TIME=$(date +%Y%m%d_%H%M%S)
echo "Training log: ${LOG_DIR}/${TIME}.log"
exec > "${LOG_DIR}/${TIME}.log" 2>&1
SAMPLE_DIR="/tmp/imagenet256_DSM_train_${TIME}_$$"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29003

python -m accelerate.commands.launch --multi_gpu --num_machines 1 --num_processes 4 \
--main_process_port="${MASTER_PORT}" \
train.py --allow_tf32 --mixed_precision="fp16" --seed=0 --report_to none \
--path_type="linear" --prediction="v" --weighting="uniform" \
--model="SiT-B/2" --enc_type="dinov2-vit-b" --encoder_depth=4 --exp_name="linear-dinov2-b-enc4" \
--checkpointing_steps=100000 --max_train_steps=400001 --output_dir="${LOG_DIR}" \
--num_samples=50000 --num_steps=100 --mode=sde --cfg_scale=2.0 --guidance_high=0.7 \
--fid_stat="${FID_STAT}" --sample_path="${SAMPLE_DIR}" --num_workers 4 --pre_fetch 2 \
--data_dir="${DATA_DIR}" --vae-checkpoint="${VAE_CHECKPOINT}" \
--proj_coeff=0.5 --latents_scale="${LATENTS_SCALE}" --latents_bias="${LATENTS_BIAS}"


# sslaunch submit -j ningmang -n 1 -q stream -- sleep infinity
# kubectl --kubeconfig /mnt/vepfs/base/infra/.kube/aliconfig \
#   -n default exec -it ningmang-ningmang-n1-261010100216-master-0 -- bash

# nohup bash imagenet256_DSM.sh > /dev/null 2>&1 &
# pkill -TERM -f stage1_vae_trainer.py
# sudo chown -R ningmang:ningmang /home/ningmang/REPA
