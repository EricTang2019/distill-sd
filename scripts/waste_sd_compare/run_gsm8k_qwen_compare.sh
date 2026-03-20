#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${WASTE_SD_COMPARE_OUT_ROOT:-${REPO_ROOT}/outputs/waste_sd_compare/${TS}}"
LOG_ROOT="${WASTE_SD_COMPARE_LOG_ROOT:-/work5/jingwut/logs/waste_sd_compare/${TS}}"

STUDENT_MODEL="${WASTE_SD_STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
TEACHER_MODEL="${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen3-1.5B-Base}"
GSM8K_TRAIN_FILE="${WASTE_SD_GSM8K_TRAIN_FILE:-${REPO_ROOT}/data/verl_fsdp_smoke/train.json}"
GSM8K_TEST_FILE="${WASTE_SD_GSM8K_TEST_FILE:-${REPO_ROOT}/data/verl_fsdp_smoke/val.json}"

RUN_TRAIN="${WASTE_SD_RUN_TRAIN:-1}"
RUN_EVAL="${WASTE_SD_RUN_EVAL:-1}"

if [[ "${RUN_TRAIN}" != "1" && "${RUN_EVAL}" != "1" ]]; then
  echo "[compare] At least one of WASTE_SD_RUN_TRAIN=1 or WASTE_SD_RUN_EVAL=1 is required." >&2
  exit 1
fi
if [[ "${RUN_TRAIN}" == "1" && ! -f "${GSM8K_TRAIN_FILE}" ]]; then
  echo "[compare] Missing train file: ${GSM8K_TRAIN_FILE}" >&2
  echo "[compare] Set WASTE_SD_GSM8K_TRAIN_FILE to a valid path under /work5." >&2
  exit 1
fi
if [[ "${RUN_EVAL}" == "1" && ! -f "${GSM8K_TEST_FILE}" ]]; then
  echo "[compare] Missing eval file: ${GSM8K_TEST_FILE}" >&2
  echo "[compare] Set WASTE_SD_GSM8K_TEST_FILE to a valid path under /work5." >&2
  exit 1
fi

TEMPERATURE="${WASTE_SD_TEMPERATURE:-1.0}"
GAMMA="${WASTE_SD_GAMMA:-3}"
TRAIN_STEPS="${WASTE_SD_TRAIN_STEPS:-100}"
EVAL_STEPS="${WASTE_SD_EVAL_STEPS:-20}"
EVAL_BATCH_SIZE="${WASTE_SD_EVAL_BATCH_SIZE:-1}"
TRAIN_LR="${WASTE_SD_TRAIN_LR:-1e-6}"
TRAIN_SAVE_FREQ="${WASTE_SD_SAVE_FREQ:-500}"
TRAIN_TEST_FREQ="${WASTE_SD_TEST_FREQ:--1}"
TRAINER_LOGGER="${WASTE_SD_TRAINER_LOGGER:-[console,wandb]}"
PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-waste_sd_gsm8k_compare}"
ALGO_CSV="${WASTE_SD_ALGORITHMS:-off_policy,on_policy,waste_sd}"
N_GPUS_PER_NODE="${WASTE_SD_N_GPUS_PER_NODE:-1}"
DEBUG_FLAG="${WASTE_SD_DEBUG:-0}"

# Eval uses the same trainer path and must satisfy the same minimal-batch divisibility checks.
# Auto-adjust eval batch size to avoid crashing when user sets 1 with multi-GPU.
if [[ "${RUN_EVAL}" == "1" ]]; then
  if (( EVAL_BATCH_SIZE < N_GPUS_PER_NODE )); then
    echo "[compare][warn] WASTE_SD_EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} < n_gpus=${N_GPUS_PER_NODE}; auto-adjust to ${N_GPUS_PER_NODE}"
    EVAL_BATCH_SIZE="${N_GPUS_PER_NODE}"
  elif (( EVAL_BATCH_SIZE % N_GPUS_PER_NODE != 0 )); then
    ADJUSTED_EVAL_BATCH_SIZE=$(( (EVAL_BATCH_SIZE + N_GPUS_PER_NODE - 1) / N_GPUS_PER_NODE * N_GPUS_PER_NODE ))
    echo "[compare][warn] WASTE_SD_EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} not divisible by n_gpus=${N_GPUS_PER_NODE}; auto-adjust to ${ADJUSTED_EVAL_BATCH_SIZE}"
    EVAL_BATCH_SIZE="${ADJUSTED_EVAL_BATCH_SIZE}"
  fi
fi

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

launcher_for_algo() {
  local algo="$1"
  case "${algo}" in
    off_policy)
      echo "${REPO_ROOT}/recipe/waste_sd/run_teacher_off_policy.sh"
      ;;
    on_policy)
      echo "${REPO_ROOT}/recipe/waste_sd/run_teacher_on_policy.sh"
      ;;
    waste_sd)
      echo "${REPO_ROOT}/recipe/waste_sd/run_waste_sd.sh"
      ;;
    *)
      echo "Unsupported algorithm: ${algo}" >&2
      return 1
      ;;
  esac
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

run_train_for_algo() {
  local algo="$1"
  local launcher="$2"
  local algo_root="$3"
  local ckpt_dir="$4"

  local train_rollout_dir="${algo_root}/train_rollout"
  local train_debug_dir="${algo_root}/train_debug"
  local train_log="${LOG_ROOT}/${algo}_train.log"
  mkdir -p "${train_rollout_dir}" "${train_debug_dir}" "${ckpt_dir}"

  ray_force_stop

  echo "[compare][train] algo=${algo} train_steps=${TRAIN_STEPS} temp=${TEMPERATURE} gamma=${GAMMA}"
  WASTE_SD_DEBUG="${DEBUG_FLAG}" \
  WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}" \
  WASTE_SD_STUDENT_MODEL="${STUDENT_MODEL}" \
  WASTE_SD_TEACHER_MODEL="${TEACHER_MODEL}" \
  WASTE_SD_TRAIN_FILE="${GSM8K_TRAIN_FILE}" \
  WASTE_SD_VAL_FILE="${GSM8K_TEST_FILE}" \
  WASTE_SD_TOTAL_TRAINING_STEPS="${TRAIN_STEPS}" \
  WASTE_SD_TEMPERATURE="${TEMPERATURE}" \
  WASTE_SD_GAMMA="${GAMMA}" \
  WASTE_SD_ACTOR_LR="${TRAIN_LR}" \
  WASTE_SD_SAVE_FREQ="${TRAIN_SAVE_FREQ}" \
  WASTE_SD_TEST_FREQ="${TRAIN_TEST_FREQ}" \
  WASTE_SD_TRAINER_LOGGER="${TRAINER_LOGGER}" \
  WASTE_SD_PROJECT_NAME="${PROJECT_NAME}" \
  WASTE_SD_EXPERIMENT_NAME="${algo}_train_${TS}" \
  WASTE_SD_DEFAULT_LOCAL_DIR="${ckpt_dir}" \
  WASTE_SD_BLOCK_EVAL_ONLY=0 \
  WASTE_SD_BLOCK_EVAL_AFTER_TRAIN=0 \
  WASTE_SD_ROLLOUT_DATA_DIR="${train_rollout_dir}" \
  WASTE_SD_DISTILL_DEBUG_DIR="${train_debug_dir}" \
  bash "${launcher}" 2>&1 | tee "${train_log}"
}

run_unified_eval_for_algo() {
  local algo="$1"
  local algo_root="$2"
  local checkpoint_path="$3"

  local eval_launcher="${REPO_ROOT}/recipe/waste_sd/run_waste_sd.sh"
  local eval_rollout_dir="${algo_root}/eval_rollout"
  local eval_debug_dir="${algo_root}/eval_debug"
  local eval_log="${LOG_ROOT}/${algo}_eval.log"
  mkdir -p "${eval_rollout_dir}" "${eval_debug_dir}"

  ray_force_stop

  echo "[compare][eval] algo=${algo} checkpoint=${checkpoint_path} eval_steps=${EVAL_STEPS} temp=${TEMPERATURE} gamma=${GAMMA}"
  WASTE_SD_DEBUG="${DEBUG_FLAG}" \
  WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}" \
  WASTE_SD_STUDENT_MODEL="${STUDENT_MODEL}" \
  WASTE_SD_TEACHER_MODEL="${TEACHER_MODEL}" \
  WASTE_SD_TRAIN_FILE="${GSM8K_TEST_FILE}" \
  WASTE_SD_VAL_FILE="${GSM8K_TEST_FILE}" \
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
  WASTE_SD_EXPERIMENT_NAME="${algo}_eval_${TS}" \
  WASTE_SD_BLOCK_EVAL_ONLY=1 \
  WASTE_SD_BLOCK_EVAL_SPLIT=val \
  WASTE_SD_BLOCK_EVAL_MAX_STEPS="${EVAL_STEPS}" \
  WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR="${eval_rollout_dir}" \
  WASTE_SD_ROLLOUT_DATA_DIR="${eval_rollout_dir}" \
  WASTE_SD_DISTILL_DEBUG_DIR="${eval_debug_dir}" \
  WASTE_SD_RESUME_MODE=resume_path \
  WASTE_SD_RESUME_FROM_PATH="${checkpoint_path}" \
  bash "${eval_launcher}" 2>&1 | tee "${eval_log}"

  SUMMARY_INPUT_ARGS+=("--input" "${algo}=${eval_rollout_dir}")
}

for algo in "${ALGO_LIST[@]}"; do
  launcher="$(launcher_for_algo "${algo}")"
  algo_root="${OUT_ROOT}/${algo}"
  ckpt_dir="${algo_root}/checkpoints"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    run_train_for_algo "${algo}" "${launcher}" "${algo_root}" "${ckpt_dir}"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    latest_ckpt=""
    algo_env_name="$(echo "${algo}" | tr '[:lower:]' '[:upper:]')"
    ckpt_var_name="WASTE_SD_${algo_env_name}_CKPT_PATH"
    ckpt_override="${!ckpt_var_name:-}"
    if [[ -n "${ckpt_override}" ]]; then
      latest_ckpt="${ckpt_override}"
    else
      if ! latest_ckpt="$(find_latest_checkpoint "${ckpt_dir}")"; then
        echo "[compare] No checkpoint found for algo=${algo} under ${ckpt_dir}" >&2
        echo "[compare] Run training first or set ${ckpt_var_name}=<global_step_dir>." >&2
        exit 1
      fi
    fi
    run_unified_eval_for_algo "${algo}" "${algo_root}" "${latest_ckpt}"
  fi
done

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "[compare] Training completed. Eval skipped because WASTE_SD_RUN_EVAL=${RUN_EVAL}."
  exit 0
fi

if (( ${#SUMMARY_INPUT_ARGS[@]} == 0 )); then
  echo "[compare] No eval outputs to summarize." >&2
  exit 1
fi

SUMMARY_JSON="${OUT_ROOT}/block_count_summary.json"
SUMMARY_TXT="${OUT_ROOT}/block_count_summary.tsv"
python "${REPO_ROOT}/scripts/waste_sd_compare/summarize_block_count.py" \
  "${SUMMARY_INPUT_ARGS[@]}" \
  --output-json "${SUMMARY_JSON}" | tee "${SUMMARY_TXT}"

echo "[compare] summary_tsv=${SUMMARY_TXT}"
echo "[compare] summary_json=${SUMMARY_JSON}"
