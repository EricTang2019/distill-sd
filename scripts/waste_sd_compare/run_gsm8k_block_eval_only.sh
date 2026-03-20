#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${WASTE_SD_COMPARE_OUT_ROOT:-${REPO_ROOT}/outputs/waste_sd_block_eval/${TS}}"
LOG_ROOT="${WASTE_SD_COMPARE_LOG_ROOT:-/work5/jingwut/logs/waste_sd_block_eval/${TS}}"

STUDENT_MODEL="${WASTE_SD_STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL="${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen3-1.5B-Base}"
GSM8K_TEST_FILE="${WASTE_SD_GSM8K_TEST_FILE:-${REPO_ROOT}/data/verl_fsdp_smoke/val.json}"
CHECKPOINT_ROOT="${WASTE_SD_CHECKPOINT_ROOT:-}"

if [[ ! -f "${GSM8K_TEST_FILE}" ]]; then
  echo "[eval] Missing eval file: ${GSM8K_TEST_FILE}" >&2
  echo "[eval] Set WASTE_SD_GSM8K_TEST_FILE to a valid path under /work5." >&2
  exit 1
fi

TEMPERATURE="${WASTE_SD_TEMPERATURE:-1.0}"
GAMMA="${WASTE_SD_GAMMA:-3}"
EVAL_STEPS="${WASTE_SD_EVAL_STEPS:-20}"
EVAL_BATCH_SIZE="${WASTE_SD_EVAL_BATCH_SIZE:-1}"
EVAL_LR="${WASTE_SD_EVAL_LR:-0}"
TRAINER_LOGGER="${WASTE_SD_TRAINER_LOGGER:-[console,wandb]}"
ALGO_CSV="${WASTE_SD_ALGORITHMS:-off_policy,on_policy,waste_sd}"
PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-waste_sd_gsm8k_eval}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

declare -a ALGO_LIST
IFS=',' read -r -a ALGO_LIST <<< "${ALGO_CSV}"

declare -a SUMMARY_INPUT_ARGS
FORCE_RAY_STOP="${WASTE_SD_FORCE_RAY_STOP:-0}"

ray_force_stop() {
  if [[ "${FORCE_RAY_STOP}" != "1" ]]; then
    return 0
  fi
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

find_latest_checkpoint() {
  local ckpt_root="$1"
  local latest
  latest="$(ls -dt "${ckpt_root}"/global_step_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  echo "${latest}"
}

for algo in "${ALGO_LIST[@]}"; do
  eval_launcher="${REPO_ROOT}/recipe/waste_sd/run_waste_sd.sh"

  phase_root="${OUT_ROOT}/${algo}/test"
  rollout_dir="${phase_root}/rollout"
  debug_dir="${phase_root}/debug"
  log_file="${LOG_ROOT}/${algo}_test.log"
  mkdir -p "${rollout_dir}" "${debug_dir}"

  algo_env_name="$(echo "${algo}" | tr '[:lower:]' '[:upper:]')"
  ckpt_var_name="WASTE_SD_${algo_env_name}_CKPT_PATH"
  ckpt_override="${!ckpt_var_name:-}"
  checkpoint_path=""
  if [[ -n "${ckpt_override}" ]]; then
    checkpoint_path="${ckpt_override}"
  elif [[ -n "${CHECKPOINT_ROOT}" ]]; then
    ckpt_dir="${CHECKPOINT_ROOT}/${algo}/checkpoints"
    if ! checkpoint_path="$(find_latest_checkpoint "${ckpt_dir}")"; then
      echo "[eval] No checkpoint found for algo=${algo} under ${ckpt_dir}" >&2
      echo "[eval] Set ${ckpt_var_name}=<global_step_dir> to override." >&2
      exit 1
    fi
  else
    echo "[eval] Missing checkpoint for algo=${algo}." >&2
    echo "[eval] Set ${ckpt_var_name}=<global_step_dir> or WASTE_SD_CHECKPOINT_ROOT=<compare_out_root>." >&2
    exit 1
  fi

  echo "[eval] algo=${algo} ckpt=${checkpoint_path} test_file=${GSM8K_TEST_FILE} steps=${EVAL_STEPS} temp=${TEMPERATURE} gamma=${GAMMA}"

  ray_force_stop

  WASTE_SD_DEBUG=1 \
  WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}" \
  WASTE_SD_STUDENT_MODEL="${STUDENT_MODEL}" \
  WASTE_SD_TEACHER_MODEL="${TEACHER_MODEL}" \
  WASTE_SD_TRAIN_FILE="${GSM8K_TEST_FILE}" \
  WASTE_SD_VAL_FILE="${GSM8K_TEST_FILE}" \
  WASTE_SD_TOTAL_TRAINING_STEPS="${EVAL_STEPS}" \
  WASTE_SD_TEMPERATURE="${TEMPERATURE}" \
  WASTE_SD_GAMMA="${GAMMA}" \
  WASTE_SD_ACTOR_LR="${EVAL_LR}" \
  WASTE_SD_TRAIN_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_VAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_PPO_MINI_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WASTE_SD_TRAINER_LOGGER="${TRAINER_LOGGER}" \
  WASTE_SD_TEST_FREQ=-1 \
  WASTE_SD_SAVE_FREQ=-1 \
  WASTE_SD_BLOCK_EVAL_ONLY=1 \
  WASTE_SD_BLOCK_EVAL_SPLIT=val \
  WASTE_SD_BLOCK_EVAL_MAX_STEPS="${EVAL_STEPS}" \
  WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR="${rollout_dir}" \
  WASTE_SD_PROJECT_NAME="${PROJECT_NAME}" \
  WASTE_SD_EXPERIMENT_NAME="${algo}_test_${TS}" \
  WASTE_SD_ROLLOUT_DATA_DIR="${rollout_dir}" \
  WASTE_SD_DISTILL_DEBUG_DIR="${debug_dir}" \
  WASTE_SD_RESUME_MODE=resume_path \
  WASTE_SD_RESUME_FROM_PATH="${checkpoint_path}" \
  bash "${eval_launcher}" 2>&1 | tee "${log_file}"

  SUMMARY_INPUT_ARGS+=("--input" "${algo}=${rollout_dir}")
done

SUMMARY_JSON="${OUT_ROOT}/block_count_summary.json"
SUMMARY_TXT="${OUT_ROOT}/block_count_summary.tsv"
python "${REPO_ROOT}/scripts/waste_sd_compare/summarize_block_count.py" \
  "${SUMMARY_INPUT_ARGS[@]}" \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

echo "[eval] summary_tsv=${SUMMARY_TXT}"
echo "[eval] summary_json=${SUMMARY_JSON}"
