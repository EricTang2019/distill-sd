#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${WASTE_SD_COMPARE_OUT_ROOT:-${REPO_ROOT}/outputs/off_policy_ckpt_eval_sweep/${TS}}"
LOG_ROOT="${WASTE_SD_COMPARE_LOG_ROOT:-/work5/jingwut/logs/off_policy_ckpt_eval_sweep/${TS}}"

CKPT_ROOT="${WASTE_SD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/waste_sd_compare/20260226_085458_gsm8k_gkd_qwen3_17b/off_policy/checkpoints}"
CKPT_STEPS_CSV="${WASTE_SD_CKPT_STEPS:-500,1000,1500,2000}"

STUDENT_MODEL="${WASTE_SD_STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL="${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen3-1.7B-Base}"
GSM8K_EVAL_FILE="${WASTE_SD_GSM8K_TEST_FILE:-${REPO_ROOT}/data/gsm8k_gkd/test.parquet}"

TEMPERATURE="${WASTE_SD_TEMPERATURE:-1.0}"
GAMMA="${WASTE_SD_GAMMA:-6}"
EVAL_STEPS="${WASTE_SD_EVAL_STEPS:-500}"
EVAL_BATCH_SIZE="${WASTE_SD_EVAL_BATCH_SIZE:-2}"
TRAINER_LOGGER="${WASTE_SD_TRAINER_LOGGER:-console}"
PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-waste_sd_gsm8k_compare}"
FORCE_RAY_STOP="${WASTE_SD_FORCE_RAY_STOP:-1}"
RAY_MEMORY_USAGE_THRESHOLD="${RAY_memory_usage_threshold:-0.99}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${GSM8K_EVAL_FILE}" ]]; then
  echo "[eval] Missing eval file: ${GSM8K_EVAL_FILE}" >&2
  exit 1
fi

ray_force_stop() {
  if [[ "${FORCE_RAY_STOP}" != "1" ]]; then
    return 0
  fi
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

declare -a CKPT_STEPS
IFS=',' read -r -a CKPT_STEPS <<< "${CKPT_STEPS_CSV}"

SUMMARY_TSV="${OUT_ROOT}/summary_all_ckpts.tsv"
echo -e "step\ttotal_samples\tsamples_with_spec_trace\ttotal_blocks\tavg_blocks_per_sample\tavg_blocks_per_traced_sample\ttotal_accepted_tokens\ttotal_response_valid_len\tavg_accepted_tokens_per_sample\tavg_response_valid_len_per_sample\trollout_dir" > "${SUMMARY_TSV}"

for step in "${CKPT_STEPS[@]}"; do
  step="$(echo "${step}" | xargs)"
  ckpt_path="${CKPT_ROOT}/global_step_${step}"
  if [[ ! -d "${ckpt_path}" ]]; then
    echo "[eval] Missing checkpoint: ${ckpt_path}" >&2
    exit 1
  fi

  step_root="${OUT_ROOT}/step_${step}/off_policy"
  rollout_dir="${step_root}/eval_rollout"
  debug_dir="${step_root}/eval_debug"
  log_file="${LOG_ROOT}/eval_step_${step}.log"
  mkdir -p "${rollout_dir}" "${debug_dir}"

  echo "[eval] start step=${step} ckpt=${ckpt_path}"

  ray_force_stop

  RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}" \
  WASTE_SD_DEBUG=0 \
  WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}" \
  WASTE_SD_STUDENT_MODEL="${STUDENT_MODEL}" \
  WASTE_SD_TEACHER_MODEL="${TEACHER_MODEL}" \
  WASTE_SD_TRAIN_FILE="${GSM8K_EVAL_FILE}" \
  WASTE_SD_VAL_FILE="${GSM8K_EVAL_FILE}" \
  WASTE_SD_TOTAL_TRAINING_STEPS=1 \
  WASTE_SD_TEMPERATURE="${TEMPERATURE}" \
  WASTE_SD_GAMMA="${GAMMA}" \
  WASTE_SD_ACTOR_LR=0 \
  WASTE_SD_TRAIN_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_VAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_PPO_MINI_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_SAVE_FREQ=-1 \
  WASTE_SD_TEST_FREQ=-1 \
  WASTE_SD_TRAINER_LOGGER="${TRAINER_LOGGER}" \
  WASTE_SD_PROJECT_NAME="${PROJECT_NAME}" \
  WASTE_SD_EXPERIMENT_NAME="off_policy_eval_${TS}_step_${step}" \
  WASTE_SD_BLOCK_EVAL_ONLY=1 \
  WASTE_SD_BLOCK_EVAL_SPLIT=val \
  WASTE_SD_BLOCK_EVAL_MAX_STEPS="${EVAL_STEPS}" \
  WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR="${rollout_dir}" \
  WASTE_SD_ROLLOUT_DATA_DIR="${rollout_dir}" \
  WASTE_SD_DISTILL_DEBUG_DIR="${debug_dir}" \
  WASTE_SD_RESUME_MODE=resume_path \
  WASTE_SD_RESUME_FROM_PATH="${ckpt_path}" \
  bash "${REPO_ROOT}/recipe/waste_sd/run_waste_sd.sh" 2>&1 | tee "${log_file}"

  step_summary_json="${OUT_ROOT}/step_${step}_summary.json"
  python "${REPO_ROOT}/scripts/waste_sd_compare/summarize_block_count.py" \
    --input "step_${step}=${rollout_dir}" \
    --output-json "${step_summary_json}" > /tmp/waste_sd_step_${step}_summary.tsv

  tail -n 1 /tmp/waste_sd_step_${step}_summary.tsv | awk -v s="${step}" -v d="${rollout_dir}" 'BEGIN{FS=OFS="\t"} {print s,$2,$3,$4,$5,$6,$7,$8,$9,$10,d}' >> "${SUMMARY_TSV}"
  echo "[eval] done step=${step} summary_json=${step_summary_json}"
done

echo "[eval] all done summary_tsv=${SUMMARY_TSV}"
