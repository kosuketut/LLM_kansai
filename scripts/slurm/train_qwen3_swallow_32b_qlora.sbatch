#!/usr/bin/env bash
#SBATCH --job-name=osaka-qwen3-32b-qlora
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs outputs hf_cache

SIF_IMAGE="${SIF_IMAGE:-${PROJECT_DIR}/singularity/qwen3_qlora.sif}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/osaka_data_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/qwen3_swallow_32b_cpt_qlora}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-tokyotech-llm/Qwen3-Swallow-32B-CPT-v0.2}"
HF_HOME_DIR="${HF_HOME_DIR:-${PROJECT_DIR}/hf_cache}"

if [[ ! -f "${SIF_IMAGE}" ]]; then
  echo "Singularity image not found: ${SIF_IMAGE}" >&2
  echo "Build it with: singularity build ${PROJECT_DIR}/singularity/qwen3_qlora.sif ${PROJECT_DIR}/singularity/qwen3_qlora.def" >&2
  exit 1
fi

singularity exec --nv \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --bind "${HF_HOME_DIR}:${HF_HOME_DIR}" \
  --pwd "${PROJECT_DIR}" \
  --env HF_HOME="${HF_HOME_DIR}" \
  --env HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub" \
  --env TRANSFORMERS_CACHE="${HF_HOME_DIR}/transformers" \
  "${SIF_IMAGE}" \
  python3 scripts/train_qwen3_swallow_32b_qlora.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
    --max-steps "${MAX_STEPS:-2000}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning-rate "${LEARNING_RATE:-1e-5}" \
    --save-steps "${SAVE_STEPS:-100}" \
    --eval-steps "${EVAL_STEPS:-200}" \
    --logging-steps "${LOGGING_STEPS:-10}" \
    --lora-r "${LORA_R:-8}" \
    --lora-alpha "${LORA_ALPHA:-20}" \
    --lora-dropout "${LORA_DROPOUT:-0.0}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
