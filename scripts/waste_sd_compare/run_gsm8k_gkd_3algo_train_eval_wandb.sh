#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

# GPU / runtime
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TMPDIR="${TMPDIR:-/work5/jingwut/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}"
export WASTE_SD_RAY_TMP_DIR="${WASTE_SD_RAY_TMP_DIR:-/work5/jingwut/rwcmp_${USER}_${TS}}"

IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
export WASTE_SD_N_GPUS_PER_NODE="${WASTE_SD_N_GPUS_PER_NODE:-${#_GPU_ARR[@]}}"

# Experiment IO
export WASTE_SD_COMPARE_OUT_ROOT="${WASTE_SD_COMPARE_OUT_ROOT:-${REPO_ROOT}/outputs/waste_sd_compare/${TS}_gsm8k_gkd_qwen3_17b}"
export WASTE_SD_COMPARE_LOG_ROOT="${WASTE_SD_COMPARE_LOG_ROOT:-/work5/jingwut/logs/waste_sd_compare/${TS}_gsm8k_gkd_qwen3_17b}"

# Models / data
export WASTE_SD_STUDENT_MODEL="${WASTE_SD_STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
export WASTE_SD_TEACHER_MODEL="${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen3-1.7B-Base}"
export WASTE_SD_GSM8K_TRAIN_FILE="${WASTE_SD_GSM8K_TRAIN_FILE:-/work5/jingwut/On-Policy-Distillation/data/gsm8k_gkd/train.parquet}"
export WASTE_SD_GSM8K_TEST_FILE="${WASTE_SD_GSM8K_TEST_FILE:-/work5/jingwut/On-Policy-Distillation/data/gsm8k_gkd/test.parquet}"

# Unified hyperparameters for all 3 algorithms
export WASTE_SD_ALGORITHMS="${WASTE_SD_ALGORITHMS:-off_policy,on_policy,waste_sd}"
export WASTE_SD_TEMPERATURE="${WASTE_SD_TEMPERATURE:-1.0}"
export WASTE_SD_GAMMA="${WASTE_SD_GAMMA:-6}"
export WASTE_SD_TRAIN_LR="${WASTE_SD_TRAIN_LR:-5e-6}"
export WASTE_SD_TRAIN_STEPS="${WASTE_SD_TRAIN_STEPS:-2000}"
export WASTE_SD_EVAL_STEPS="${WASTE_SD_EVAL_STEPS:-500}"
export WASTE_SD_EVAL_BATCH_SIZE="${WASTE_SD_EVAL_BATCH_SIZE:-1}"
export WASTE_SD_SAVE_FREQ="${WASTE_SD_SAVE_FREQ:-500}"

# Training stability / throughput knobs
export WASTE_SD_TRAIN_BATCH_SIZE="${WASTE_SD_TRAIN_BATCH_SIZE:-64}"
export WASTE_SD_PPO_MINI_BATCH_SIZE="${WASTE_SD_PPO_MINI_BATCH_SIZE:-16}"
export WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU="${WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export WASTE_SD_TEACHER_FORWARD_BACKEND="${WASTE_SD_TEACHER_FORWARD_BACKEND:-local_replica}"
export WASTE_SD_MAX_PROMPT_LENGTH="${WASTE_SD_MAX_PROMPT_LENGTH:-256}"
export WASTE_SD_ROLLOUT_PROMPT_LENGTH="${WASTE_SD_ROLLOUT_PROMPT_LENGTH:-256}"
export WASTE_SD_MAX_RESPONSE_LENGTH="${WASTE_SD_MAX_RESPONSE_LENGTH:-256}"
export WASTE_SD_ROLLOUT_RESPONSE_LENGTH="${WASTE_SD_ROLLOUT_RESPONSE_LENGTH:-256}"
export WASTE_SD_FILTER_OVERLONG_PROMPTS="${WASTE_SD_FILTER_OVERLONG_PROMPTS:-true}"

# Logging / checkpoint
export WASTE_SD_TRAINER_LOGGER="${WASTE_SD_TRAINER_LOGGER:-[console,wandb]}"
export WASTE_SD_PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-waste_sd_gsm8k_compare}"
export WASTE_SD_RUN_TRAIN="${WASTE_SD_RUN_TRAIN:-1}"
export WASTE_SD_RUN_EVAL="${WASTE_SD_RUN_EVAL:-1}"
export WASTE_SD_FORCE_RAY_STOP="${WASTE_SD_FORCE_RAY_STOP:-0}"

mkdir -p "${WASTE_SD_COMPARE_OUT_ROOT}" "${WASTE_SD_COMPARE_LOG_ROOT}" "${TMPDIR}"

echo "[compare] repo=${REPO_ROOT}"
echo "[compare] gpus=${CUDA_VISIBLE_DEVICES} n_gpus_per_node=${WASTE_SD_N_GPUS_PER_NODE}"
echo "[compare] out=${WASTE_SD_COMPARE_OUT_ROOT}"
echo "[compare] logs=${WASTE_SD_COMPARE_LOG_ROOT}"
echo "[compare] algos=${WASTE_SD_ALGORITHMS}"
echo "[compare] train_steps=${WASTE_SD_TRAIN_STEPS} eval_steps=${WASTE_SD_EVAL_STEPS}"
echo "[compare] temp=${WASTE_SD_TEMPERATURE} gamma=${WASTE_SD_GAMMA}"

bash "${REPO_ROOT}/scripts/waste_sd_compare/run_gsm8k_qwen_compare.sh"
