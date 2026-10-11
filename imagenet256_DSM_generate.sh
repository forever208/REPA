#!/bin/bash
set -eo pipefail

# The Python user packages belong to ningmang, including when launched from a root shell.
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
# FID Evaluation
#############################

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508


cd /home/ningmang/REPA
LOG_DIR=/mnt/vepfs/base2/stream/ningmang/REPA_exps/sit_b/imagenet256_DSM_05lpips_76k_stdx1
# generate.py loads the EMA weights from the final 400k-step checkpoint.
CKPT="${LOG_DIR}/linear-dinov2-b-enc4/checkpoints/0400000.pt"
# Match the VAE and normalization recorded in the training run's args.json.
VAE_CHECKPOINT=/mnt/vepfs/base2/stream/ningmang/vae_exps/imagenet256_DSM_ft_05lpips_10epochs/checkpoints/76000.ckpt
LATENTS_SCALE=0.2124
LATENTS_BIAS=-0.3952
FID_STAT=/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz

for REQUIRED_FILE in "${CKPT}" "${VAE_CHECKPOINT}" "${FID_STAT}"; do
    if [[ ! -r "${REQUIRED_FILE}" ]]; then
        echo "Error: required file is not readable: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"
TIME=$(date +%Y%m%d_%H%M%S)
echo "Evaluation log: ${LOG_DIR}/generate_${TIME}.log"
exec > "${LOG_DIR}/generate_${TIME}.log" 2>&1
FID_LOG="${LOG_DIR}/FID_cond.log"
# Use the same CFG sweep in imagenet256_SDVAE_generate.sh.
CFG_SCALES=(1.0 2.0 2.5 2.6 2.7 2.8 2.9 3.0)
NUM_STEPS=100
GUIDANCE_HIGH=0.7

if [[ ! -s "${FID_LOG}" ]]; then
    printf "time\tckpt\tnum_steps\tcfg_scale\tguidance_high\tfid\n" > "${FID_LOG}"
fi

for CFG_SCALE in "${CFG_SCALES[@]}"; do
    echo "Generating samples with cfg-scale=${CFG_SCALE}"
    SAMPLE_DIR=$(mktemp -d "/tmp/imagenet256_DSM_cfg${CFG_SCALE}_XXXXXX") || exit 1
    RUN_LOG=$(mktemp "${LOG_DIR}/.fid_run_XXXXXX")
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port="${MASTER_PORT}" generate.py \
    --model="SiT-B/2" --encoder-depth=4 --projector-embed-dims=768 --fused-attn \
    --resolution=256 --num-classes=1000 \
    --ckpt="${CKPT}" \
    --vae-checkpoint="${VAE_CHECKPOINT}" \
    --latents-scale="${LATENTS_SCALE}" --latents-bias="${LATENTS_BIAS}" \
    --global-seed=0 --num-fid-samples=50000 --per-proc-batch-size=32 \
    --num-steps="${NUM_STEPS}" --mode=sde --path-type=linear --cfg-scale="${CFG_SCALE}" --guidance-high="${GUIDANCE_HIGH}" \
    --fid-stat="${FID_STAT}" \
    --sample-dir="${SAMPLE_DIR}" 2>&1 | tee "${RUN_LOG}"

    FID_RESULT=$(awk '/^fid[0-9]+=/ {result=$0} END {print result}' "${RUN_LOG}")
    if [[ -n "${FID_RESULT}" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date +%Y-%m-%dT%H:%M:%S)" "${CKPT}" "${NUM_STEPS}" \
            "${CFG_SCALE}" "${GUIDANCE_HIGH}" "${FID_RESULT}" >> "${FID_LOG}"
        # generate.py deletes the images after FID; remove the empty parent.
        rmdir -- "${SAMPLE_DIR}"
    else
        echo "Error: no FID result found for cfg-scale=${CFG_SCALE}" >&2
        exit 1
    fi
    rm -f "${RUN_LOG}"
done

# sslaunch submit -j dsmsample -q stream -n 1 -w /home/ningmang/REPA --no-log -- bash /home/ningmang/REPA/imagenet256_DSM_generate.sh
