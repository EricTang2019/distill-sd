#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TIMESTAMP="${WASTE_SD_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MERGE_PYTHON="${WASTE_SD_POST_TRAIN_MERGE_PYTHON:-$(command -v python)}"

if [[ -z "${MERGE_PYTHON}" || ! -x "${MERGE_PYTHON}" ]]; then
  echo "[waste_sd] Post-train merge python not found. Activate env or set WASTE_SD_POST_TRAIN_MERGE_PYTHON." >&2
  exit 1
fi

_has_merged_hf_model() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  compgen -G "${model_dir}/*.safetensors" > /dev/null || compgen -G "${model_dir}/pytorch_model*.bin" > /dev/null
}

_detect_post_train_vllm_env_name() {
  if ! command -v conda > /dev/null 2>&1; then
    printf '%s\n' "verlsd"
    return
  fi

  local env_names
  env_names="$(conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ {print $1}')"
  if grep -qx 'verlsd312' <<< "${env_names}"; then
    printf '%s\n' "verlsd312"
    return
  fi
  if grep -qx 'verlsd' <<< "${env_names}"; then
    printf '%s\n' "verlsd"
    return
  fi
  printf '%s\n' "verlsd"
}

# Keep defaults portable across machines. Users can override any of these env vars.
export WASTE_SD_DATA_ROOT="${WASTE_SD_DATA_ROOT:-${REPO_ROOT}/data/verl_fsdp_smoke}"
export WASTE_SD_TRAIN_FILE="${WASTE_SD_TRAIN_FILE:-${WASTE_SD_DATA_ROOT}/train.json}"
export WASTE_SD_VAL_FILE="${WASTE_SD_VAL_FILE:-${WASTE_SD_TRAIN_FILE}}"
export WASTE_SD_RAY_TMP_DIR="${WASTE_SD_RAY_TMP_DIR:-${TMPDIR:-/tmp}/waste_sd_ray_${USER}_$$}"
export WASTE_SD_DEFAULT_LOCAL_DIR="${WASTE_SD_DEFAULT_LOCAL_DIR:-${REPO_ROOT}/outputs/offline_exact_block_count_${TIMESTAMP}}"
export WASTE_SD_PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-waste_sd_offline_compare}"
export WASTE_SD_EXPERIMENT_NAME="${WASTE_SD_EXPERIMENT_NAME:-offline_exact_block_count_${TIMESTAMP}}"
export WASTE_SD_EXACT_UNWEIGHTED_KL_COEF="${WASTE_SD_EXACT_UNWEIGHTED_KL_COEF:-0.0}"
export WASTE_SD_POST_TRAIN_VLLM_EVAL="${WASTE_SD_POST_TRAIN_VLLM_EVAL:-1}"
export WASTE_SD_POST_TRAIN_VLLM_ENV_NAME="${WASTE_SD_POST_TRAIN_VLLM_ENV_NAME:-$(_detect_post_train_vllm_env_name)}"
export WASTE_SD_POST_TRAIN_VLLM_VISIBLE_DEVICES="${WASTE_SD_POST_TRAIN_VLLM_VISIBLE_DEVICES:-0}"
export WASTE_SD_POST_TRAIN_VLLM_INPUT_DATA="${WASTE_SD_POST_TRAIN_VLLM_INPUT_DATA:-${PROJECT_ROOT}/data/gsm8k_gkd/test.parquet}"
export WASTE_SD_POST_TRAIN_VLLM_MAX_PROMPTS="${WASTE_SD_POST_TRAIN_VLLM_MAX_PROMPTS:-256}"
export WASTE_SD_POST_TRAIN_VLLM_OUTPUT_DIR="${WASTE_SD_POST_TRAIN_VLLM_OUTPUT_DIR:-${WASTE_SD_DEFAULT_LOCAL_DIR}/post_train_vllm_eval}"
export WASTE_SD_POST_TRAIN_VLLM_RUN_LABEL="${WASTE_SD_POST_TRAIN_VLLM_RUN_LABEL:-${WASTE_SD_EXPERIMENT_NAME}_test256}"
export WASTE_SD_POST_TRAIN_VLLM_TARGET_MODEL="${WASTE_SD_POST_TRAIN_VLLM_TARGET_MODEL:-${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen3-1.7B}}"
export WASTE_SD_POST_TRAIN_VLLM_STEPS="${WASTE_SD_POST_TRAIN_VLLM_STEPS:-450,300,150}"
export WASTE_SD_POST_TRAIN_VLLM_PARALLEL_MODE="${WASTE_SD_POST_TRAIN_VLLM_PARALLEL_MODE:-dp}"
export WASTE_SD_POST_TRAIN_VLLM_MAX_MODEL_LEN="${WASTE_SD_POST_TRAIN_VLLM_MAX_MODEL_LEN:-4096}"
export WASTE_SD_POST_TRAIN_VLLM_ENFORCE_EAGER="${WASTE_SD_POST_TRAIN_VLLM_ENFORCE_EAGER:-0}"
export WASTE_SD_POST_TRAIN_VLLM_STRICT_DRAFT_PROBS="${WASTE_SD_POST_TRAIN_VLLM_STRICT_DRAFT_PROBS:-1}"
export WASTE_SD_POST_TRAIN_VLLM_TEMPERATURE="${WASTE_SD_POST_TRAIN_VLLM_TEMPERATURE:-1.0}"
export WASTE_SD_POST_TRAIN_VLLM_MAX_CONCURRENT_REQUESTS="${WASTE_SD_POST_TRAIN_VLLM_MAX_CONCURRENT_REQUESTS:-4}"
export WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_BATCHED_TOKENS="${WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
export WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_SEQS="${WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_SEQS:-64}"
export WASTE_SD_POST_TRAIN_VLLM_MERGE_IF_NEEDED="${WASTE_SD_POST_TRAIN_VLLM_MERGE_IF_NEEDED:-1}"
export WASTE_SD_POST_TRAIN_VLLM_MERGE_BACKEND="${WASTE_SD_POST_TRAIN_VLLM_MERGE_BACKEND:-fsdp}"
export WASTE_SD_POST_TRAIN_VLLM_MERGE_USE_CPU_INITIALIZATION="${WASTE_SD_POST_TRAIN_VLLM_MERGE_USE_CPU_INITIALIZATION:-1}"
export WASTE_SD_POST_TRAIN_VLLM_MERGE_TRUST_REMOTE_CODE="${WASTE_SD_POST_TRAIN_VLLM_MERGE_TRUST_REMOTE_CODE:-0}"

mkdir -p "${WASTE_SD_RAY_TMP_DIR}"
mkdir -p "${WASTE_SD_DEFAULT_LOCAL_DIR}"
mkdir -p "${WASTE_SD_POST_TRAIN_VLLM_OUTPUT_DIR}"

TRAIN_CMD=(
  bash "${REPO_ROOT}/recipe/waste_sd/run_teacher_off_policy.sh"
  distill.data_mode=offline_teacher_rollout \
  distill.loss_type=exact_block_count_wnll \
  distill.weighting_mode=uniform_mean \
  distill.kl_floor_coef=0.0 \
  distill.exact_unweighted_kl_coef="${WASTE_SD_EXACT_UNWEIGHTED_KL_COEF}" \
  data.seed="${WASTE_SD_DATA_SEED:-123}" \
  actor_rollout_ref.actor.data_loader_seed="${WASTE_SD_DATA_LOADER_SEED:-123}" \
  "$@"
)

"${TRAIN_CMD[@]}"

if [[ "${WASTE_SD_POST_TRAIN_VLLM_EVAL}" == "1" ]]; then
  if [[ ! -f "${WASTE_SD_POST_TRAIN_VLLM_INPUT_DATA}" ]]; then
    echo "[waste_sd] Post-train vLLM eval skipped: missing input data ${WASTE_SD_POST_TRAIN_VLLM_INPUT_DATA}" >&2
    exit 0
  fi

  CKPT_DIRS=()
  if [[ -n "${WASTE_SD_POST_TRAIN_VLLM_STEPS}" ]]; then
    IFS=',' read -r -a POST_STEPS <<< "${WASTE_SD_POST_TRAIN_VLLM_STEPS}"
    for step in "${POST_STEPS[@]}"; do
      step="${step// /}"
      [[ -z "${step}" ]] && continue
      ckpt_dir="${WASTE_SD_DEFAULT_LOCAL_DIR}/global_step_${step}"
      if [[ -d "${ckpt_dir}" ]]; then
        CKPT_DIRS+=("${ckpt_dir}")
      else
        echo "[waste_sd] Post-train vLLM eval skipping missing checkpoint ${ckpt_dir}" >&2
      fi
    done
  else
    while IFS= read -r -d '' ckpt_dir; do
      CKPT_DIRS+=("${ckpt_dir}")
    done < <(find "${WASTE_SD_DEFAULT_LOCAL_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print0 | sort -z -V)
  fi

  if (( ${#CKPT_DIRS[@]} == 0 )); then
    echo "[waste_sd] Post-train vLLM eval skipped: no checkpoints found under ${WASTE_SD_DEFAULT_LOCAL_DIR}" >&2
    exit 0
  fi

  unset CUDA_VISIBLE_DEVICES
  for ckpt_dir in "${CKPT_DIRS[@]}"; do
    step_name="$(basename "${ckpt_dir}")"
    actor_ckpt_dir="${ckpt_dir}/actor"
    draft_model_path="${actor_ckpt_dir}/huggingface_merged"
    if ! _has_merged_hf_model "${draft_model_path}"; then
      if [[ "${WASTE_SD_POST_TRAIN_VLLM_MERGE_IF_NEEDED}" != "1" ]]; then
        echo "[waste_sd] Post-train vLLM eval skipping ${step_name}: missing merged draft model at ${draft_model_path}" >&2
        continue
      fi
      if [[ ! -d "${actor_ckpt_dir}" ]]; then
        echo "[waste_sd] Post-train vLLM eval skipping ${step_name}: missing actor checkpoint dir ${actor_ckpt_dir}" >&2
        continue
      fi

      MERGE_CMD=(
        "${MERGE_PYTHON}"
        -m
        verl.model_merger
        merge
        --backend
        "${WASTE_SD_POST_TRAIN_VLLM_MERGE_BACKEND}"
        --local_dir
        "${actor_ckpt_dir}"
        --target_dir
        "${draft_model_path}"
      )
      if [[ "${WASTE_SD_POST_TRAIN_VLLM_MERGE_USE_CPU_INITIALIZATION}" == "1" ]]; then
        MERGE_CMD+=(--use_cpu_initialization)
      fi
      if [[ "${WASTE_SD_POST_TRAIN_VLLM_MERGE_TRUST_REMOTE_CODE}" == "1" ]]; then
        MERGE_CMD+=(--trust-remote-code)
      fi

      echo "[waste_sd] Post-train vLLM eval merging ${step_name} to ${draft_model_path}"
      CUDA_VISIBLE_DEVICES="" "${MERGE_CMD[@]}"
    fi

    if ! _has_merged_hf_model "${draft_model_path}"; then
      echo "[waste_sd] Post-train vLLM eval skipping ${step_name}: merge did not produce a valid HF model at ${draft_model_path}" >&2
      continue
    fi

    WASTE_SD_VLLM_ENV_NAME="${WASTE_SD_POST_TRAIN_VLLM_ENV_NAME}" \
    WASTE_SD_VLLM_TARGET_MODEL="${WASTE_SD_POST_TRAIN_VLLM_TARGET_MODEL}" \
    WASTE_SD_VLLM_DRAFT_MODEL="${draft_model_path}" \
    WASTE_SD_VLLM_INPUT_DATA="${WASTE_SD_POST_TRAIN_VLLM_INPUT_DATA}" \
    WASTE_SD_VLLM_OUTPUT_DIR="${WASTE_SD_POST_TRAIN_VLLM_OUTPUT_DIR}" \
    WASTE_SD_VLLM_RUN_LABEL="${WASTE_SD_POST_TRAIN_VLLM_RUN_LABEL}_${step_name}" \
    WASTE_SD_VLLM_VISIBLE_DEVICES="${WASTE_SD_POST_TRAIN_VLLM_VISIBLE_DEVICES}" \
    WASTE_SD_VLLM_MAX_PROMPTS="${WASTE_SD_POST_TRAIN_VLLM_MAX_PROMPTS}" \
    WASTE_SD_VLLM_PARALLEL_MODE="${WASTE_SD_POST_TRAIN_VLLM_PARALLEL_MODE}" \
    WASTE_SD_VLLM_MAX_MODEL_LEN="${WASTE_SD_POST_TRAIN_VLLM_MAX_MODEL_LEN}" \
    WASTE_SD_VLLM_ENFORCE_EAGER="${WASTE_SD_POST_TRAIN_VLLM_ENFORCE_EAGER}" \
    WASTE_SD_VLLM_STRICT_DRAFT_PROBS="${WASTE_SD_POST_TRAIN_VLLM_STRICT_DRAFT_PROBS}" \
    WASTE_SD_VLLM_TEMPERATURE="${WASTE_SD_POST_TRAIN_VLLM_TEMPERATURE}" \
    WASTE_SD_VLLM_MAX_CONCURRENT_REQUESTS="${WASTE_SD_POST_TRAIN_VLLM_MAX_CONCURRENT_REQUESTS}" \
    WASTE_SD_VLLM_MAX_NUM_BATCHED_TOKENS="${WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_BATCHED_TOKENS}" \
    WASTE_SD_VLLM_MAX_NUM_SEQS="${WASTE_SD_POST_TRAIN_VLLM_MAX_NUM_SEQS}" \
    bash "${REPO_ROOT}/recipe/waste_sd/run_vllm_block_eval_single.sh"
  done
fi
