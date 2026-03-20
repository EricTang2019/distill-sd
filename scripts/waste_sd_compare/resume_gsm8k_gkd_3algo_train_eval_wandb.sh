#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${1:-${WASTE_SD_COMPARE_OUT_ROOT:-}}"
if [[ -z "${OUT_ROOT}" ]]; then
  echo "[resume] Missing compare output root." >&2
  echo "[resume] Usage: $0 <existing_compare_out_root>" >&2
  echo "[resume] or set WASTE_SD_COMPARE_OUT_ROOT=<existing_compare_out_root>" >&2
  exit 1
fi
if [[ ! -d "${OUT_ROOT}" ]]; then
  echo "[resume] Compare output root not found: ${OUT_ROOT}" >&2
  exit 1
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${WASTE_SD_COMPARE_LOG_ROOT:-/work5/jingwut/On-Policy-Distillation/logs/waste_sd_compare_resume_${TS}}"
mkdir -p "${LOG_ROOT}"

ALGO_CSV="${WASTE_SD_ALGORITHMS:-off_policy,on_policy,waste_sd}"
TRAIN_STEPS="${WASTE_SD_TRAIN_STEPS:-2000}"
EVAL_STEPS="${WASTE_SD_EVAL_STEPS:-500}"

TRAIN_BATCH_SIZE="${WASTE_SD_TRAIN_BATCH_SIZE:-64}"
TRAIN_MINI_BATCH_SIZE="${WASTE_SD_PPO_MINI_BATCH_SIZE:-64}"
TRAIN_MICRO_BATCH_SIZE_PER_GPU="${WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"

EVAL_BATCH_SIZE="${WASTE_SD_EVAL_BATCH_SIZE:-8}"
EVAL_ROLLOUT_GPU_MEM_UTIL="${WASTE_SD_EVAL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
EVAL_ROLLOUT_MAX_BATCHED_TOKENS="${WASTE_SD_EVAL_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}"
EVAL_ROLLOUT_MAX_NUM_SEQS="${WASTE_SD_EVAL_ROLLOUT_MAX_NUM_SEQS:-64}"
EVAL_ROLLOUT_ENFORCE_EAGER="${WASTE_SD_EVAL_ROLLOUT_ENFORCE_EAGER:-true}"

latest_ckpt() {
  local ckpt_dir="$1"
  ls -dt "${ckpt_dir}"/global_step_* 2>/dev/null | head -n1 || true
}

has_target_ckpt() {
  local ckpt_dir="$1"
  [[ -d "${ckpt_dir}/global_step_${TRAIN_STEPS}" ]]
}

to_upper() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

echo "[resume] repo=${REPO_ROOT}"
echo "[resume] out_root=${OUT_ROOT}"
echo "[resume] log_root=${LOG_ROOT}"
echo "[resume] algos=${ALGO_CSV}"
echo "[resume] train_steps=${TRAIN_STEPS} eval_steps=${EVAL_STEPS}"

IFS=',' read -r -a ALGO_LIST <<< "${ALGO_CSV}"
for algo in "${ALGO_LIST[@]}"; do
  algo_root="${OUT_ROOT}/${algo}"
  ckpt_dir="${algo_root}/checkpoints"
  mkdir -p "${algo_root}" "${ckpt_dir}"

  train_log="${LOG_ROOT}/${algo}_resume_train.log"
  eval_log="${LOG_ROOT}/${algo}_resume_eval.log"

  if has_target_ckpt "${ckpt_dir}"; then
    echo "[resume] algo=${algo} train already reached global_step_${TRAIN_STEPS}, skip train."
  else
    echo "[resume] algo=${algo} training to global_step_${TRAIN_STEPS}..."
    WASTE_SD_COMPARE_OUT_ROOT="${OUT_ROOT}" \
    WASTE_SD_COMPARE_LOG_ROOT="${LOG_ROOT}" \
    WASTE_SD_ALGORITHMS="${algo}" \
    WASTE_SD_RUN_TRAIN=1 \
    WASTE_SD_RUN_EVAL=0 \
    WASTE_SD_TRAIN_STEPS="${TRAIN_STEPS}" \
    WASTE_SD_SAVE_FREQ="${WASTE_SD_SAVE_FREQ:-500}" \
    WASTE_SD_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    WASTE_SD_PPO_MINI_BATCH_SIZE="${TRAIN_MINI_BATCH_SIZE}" \
    WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU="${TRAIN_MICRO_BATCH_SIZE_PER_GPU}" \
    WASTE_SD_DEBUG="${WASTE_SD_DEBUG:-0}" \
    bash "${REPO_ROOT}/scripts/waste_sd_compare/run_gsm8k_gkd_3algo_train_eval_wandb.sh" \
      2>&1 | tee "${train_log}"
  fi

  ckpt_path="$(latest_ckpt "${ckpt_dir}")"
  if [[ -z "${ckpt_path}" ]]; then
    echo "[resume] algo=${algo} no checkpoint found under ${ckpt_dir}, skip eval." >&2
    continue
  fi

  upper_algo="$(to_upper "${algo}")"
  ckpt_env_name="WASTE_SD_${upper_algo}_CKPT_PATH"
  echo "[resume] algo=${algo} eval from checkpoint=${ckpt_path}"
  env "${ckpt_env_name}=${ckpt_path}" \
    WASTE_SD_COMPARE_OUT_ROOT="${OUT_ROOT}" \
    WASTE_SD_COMPARE_LOG_ROOT="${LOG_ROOT}" \
    WASTE_SD_ALGORITHMS="${algo}" \
    WASTE_SD_RUN_TRAIN=0 \
    WASTE_SD_RUN_EVAL=1 \
    WASTE_SD_EVAL_STEPS="${EVAL_STEPS}" \
    WASTE_SD_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    WASTE_SD_SAVE_FREQ=-1 \
    WASTE_SD_ROLLOUT_GPU_MEMORY_UTILIZATION="${EVAL_ROLLOUT_GPU_MEM_UTIL}" \
    WASTE_SD_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${EVAL_ROLLOUT_MAX_BATCHED_TOKENS}" \
    WASTE_SD_ROLLOUT_MAX_NUM_SEQS="${EVAL_ROLLOUT_MAX_NUM_SEQS}" \
    WASTE_SD_ROLLOUT_ENFORCE_EAGER="${EVAL_ROLLOUT_ENFORCE_EAGER}" \
    WASTE_SD_DEBUG="${WASTE_SD_DEBUG:-0}" \
    bash "${REPO_ROOT}/scripts/waste_sd_compare/run_gsm8k_gkd_3algo_train_eval_wandb.sh" \
      2>&1 | tee "${eval_log}"
done

echo "[resume] done. logs in ${LOG_ROOT}"
