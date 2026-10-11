#!/bin/bash
set -eo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate /home/ningmang/miniconda3/envs/REPA

export HF_HOME=/home/ningmang/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ningmang/torch_cache

#############################
# FID Evaluation
#############################

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29508


cd /home/ningmang/REPA
LOG_DIR=/mnt/vepfs/base2/stream/ningmang/REPA_exps/imagenet256_SDVAE_stdx1_XL
mkdir -p "${LOG_DIR}"
TIME=$(date +%Y%m%d_%H%M%S)
exec > "${LOG_DIR}/generate_${TIME}.log" 2>&1
FID_LOG="${LOG_DIR}/FID_cond.log"

CFG_SCALES=(1.0 1.8 1.9 2.0 2.1 2.2 2.3)
CKPT="${LOG_DIR}/linear-dinov2-b-enc8/checkpoints/1000000.pt"
NUM_STEPS=100
GUIDANCE_HIGH=0.7

if [[ ! -s "${FID_LOG}" ]]; then
    printf "time\tckpt\tnum_steps\tcfg_scale\tguidance_high\tfid\n" > "${FID_LOG}"
fi

for CFG_SCALE in "${CFG_SCALES[@]}"; do
    echo "Generating samples with cfg-scale=${CFG_SCALE}"
    SAMPLE_DIR=$(mktemp -d "/tmp/imagenet256_SDVAE_XL_cfg${CFG_SCALE}_XXXXXX") || exit 1
    RUN_LOG=$(mktemp "${LOG_DIR}/.fid_run_XXXXXX")
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port="${MASTER_PORT}" generate.py \
    --model="SiT-XL/2" --encoder-depth=8 --projector-embed-dims=768 --fused-attn \
    --ckpt="${CKPT}" \
    --vae-checkpoint="/mnt/vepfs/base2/stream/ningmang/vae_exps/imagenet256_SDVAE/model.ckpt" \
    --latents-scale=0.22506 --latents-bias=0.3279 \
    --global-seed=0 --num-fid-samples=50000 --per-proc-batch-size=32 \
    --num-steps="${NUM_STEPS}" --mode=sde --path-type=linear --cfg-scale="${CFG_SCALE}" --guidance-high="${GUIDANCE_HIGH}" \
    --fid-stat="/home/ningmang/REPA/asset/fid_stats_imagenet256_guided_diffusion.npz" \
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

# sslaunch submit -j sdsample -q stream -n 1 -w /home/ningmang/REPA --no-log -- bash /home/ningmang/REPA/imagenet256_SDVAE_XL_generate.sh
