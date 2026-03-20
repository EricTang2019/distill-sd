#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_USE_UVLOOP=0

# Device visibility strategy:
# - default (1): keep visible-devices untouched and let workers bind by Ray accelerator id.
#   This avoids duplicate-GPU binding on some local+Ray setups.
# - 1: keep visible-devices untouched and let workers bind by Ray accelerator id.
WASTE_SD_RAY_NOSET_VISIBLE_DEVICES="${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES:-1}"
if [[ "${WASTE_SD_RAY_NOSET_VISIBLE_DEVICES}" == "1" ]]; then
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
  unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
  WASTE_SD_RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV="1"
else
  unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
  unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES
  unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES
  # Force-disable NOSET in Ray workers to avoid inheriting stale value from raylet env.
  WASTE_SD_RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV=""
fi

# Use current environment python by default; allow explicit override.
PYTHON_BIN="${WASTE_SD_PYTHON:-$(command -v python)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[waste_sd] Python not found. Activate env or set WASTE_SD_PYTHON." >&2
  exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONNOUSERSITE=1

PY_PREFIX="$("${PYTHON_BIN}" -S - <<'PY'
import sys
print(sys.prefix)
PY
)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Avoid loading host CUDA runtime (e.g., /usr/local/cuda/lib64/libcudart.so.12),
# which can be older/incompatible with torch wheels and breaks `import torch`
# in spawned SGLang scheduler processes.
unset LD_PRELOAD
_site_pkg_dir="$("${PYTHON_BIN}" -S - <<'PY'
import sysconfig
print(sysconfig.get_paths().get("purelib", ""))
PY
)"
_curated_ld_parts=()
if [[ -n "${PY_PREFIX:-}" ]]; then
  _curated_ld_parts+=("${PY_PREFIX}/lib")
fi
for _p in \
  "${_site_pkg_dir}/nvidia/cuda_runtime/lib" \
  "${_site_pkg_dir}/nvidia/cuda_nvrtc/lib" \
  "${_site_pkg_dir}/nvidia/cublas/lib" \
  "${_site_pkg_dir}/nvidia/cudnn/lib"; do
  if [[ -d "${_p}" ]]; then
    _curated_ld_parts+=("${_p}")
  fi
done
if (( ${#_curated_ld_parts[@]} > 0 )); then
  LD_LIBRARY_PATH="$(IFS=:; echo "${_curated_ld_parts[*]}")"
else
  # Fallback: only remove known problematic host CUDA dirs.
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH="$(
      printf '%s' "${LD_LIBRARY_PATH}" \
        | tr ':' '\n' \
        | awk '$0 !~ "^/usr/local/cuda(-[0-9.]+)?/lib64$" && $0 !~ "^/usr/local/cuda$" && !seen[$0]++' \
        | paste -sd: -
    )"
  else
    LD_LIBRARY_PATH=""
  fi
fi
export LD_LIBRARY_PATH

# Keep tmp/cache paths consistent across parent and Ray workers.
if [[ -n "${TMPDIR:-}" ]]; then export TMPDIR; fi
if [[ -n "${TEMP:-}" ]]; then export TEMP; fi
if [[ -n "${TMP:-}" ]]; then export TMP; fi
if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then export TORCHINDUCTOR_CACHE_DIR; fi

# Keep this short: AF_UNIX socket path must be <= 107 bytes on Linux.
export WASTE_SD_RAY_TMP_DIR="${WASTE_SD_RAY_TMP_DIR:-/work5/jingwut/rwsd_${USER}_$$}"
# Ray appends "/session_.../sockets/plasma_store" to _temp_dir.
_RAY_SOCKET_SUFFIX_LEN=70
if (( ${#WASTE_SD_RAY_TMP_DIR} + _RAY_SOCKET_SUFFIX_LEN > 107 )); then
  _fallback_tmp_dir="/work5/jingwut/rwsd_${USER}_$$"
  echo "[waste_sd] WASTE_SD_RAY_TMP_DIR is too long for AF_UNIX sockets: '${WASTE_SD_RAY_TMP_DIR}'" >&2
  echo "[waste_sd] Falling back to '${_fallback_tmp_dir}'" >&2
  export WASTE_SD_RAY_TMP_DIR="${_fallback_tmp_dir}"
fi
mkdir -p "${WASTE_SD_RAY_TMP_DIR}"

# On-policy teacher baseline does not require Waste-SD speculative runtime patches.
export VERL_SGLANG_WASTE_SD_PATCH=0
export VERL_SGLANG_STANDALONE_SYNC_MODE=both

# Multi-node and sync-tuning knobs (defaults keep current single-node behavior).
WASTE_SD_NNODES="${WASTE_SD_NNODES:-1}"
WASTE_SD_N_GPUS_PER_NODE="${WASTE_SD_N_GPUS_PER_NODE:-1}"
WASTE_SD_RAY_ADDRESS="${WASTE_SD_RAY_ADDRESS:-local}"
WASTE_SD_UPDATE_WEIGHTS_BUCKET_MB="${WASTE_SD_UPDATE_WEIGHTS_BUCKET_MB:-2048}"
WASTE_SD_TRAIN_BATCH_SIZE="${WASTE_SD_TRAIN_BATCH_SIZE:-8}"
WASTE_SD_VAL_BATCH_SIZE="${WASTE_SD_VAL_BATCH_SIZE:-8}"
WASTE_SD_MAX_PROMPT_LENGTH="${WASTE_SD_MAX_PROMPT_LENGTH:-128}"
WASTE_SD_MAX_RESPONSE_LENGTH="${WASTE_SD_MAX_RESPONSE_LENGTH:-64}"
WASTE_SD_FILTER_OVERLONG_PROMPTS="${WASTE_SD_FILTER_OVERLONG_PROMPTS:-false}"
WASTE_SD_ROLLOUT_PROMPT_LENGTH="${WASTE_SD_ROLLOUT_PROMPT_LENGTH:-${WASTE_SD_MAX_PROMPT_LENGTH}}"
WASTE_SD_ROLLOUT_RESPONSE_LENGTH="${WASTE_SD_ROLLOUT_RESPONSE_LENGTH:-${WASTE_SD_MAX_RESPONSE_LENGTH}}"
WASTE_SD_PPO_MINI_BATCH_SIZE="${WASTE_SD_PPO_MINI_BATCH_SIZE:-4}"
WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU="${WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
WASTE_SD_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${WASTE_SD_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
WASTE_SD_ROLLOUT_AGENT_NUM_WORKERS="${WASTE_SD_ROLLOUT_AGENT_NUM_WORKERS:-1}"
WASTE_SD_ROLLOUT_GPU_MEMORY_UTILIZATION="${WASTE_SD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
WASTE_SD_ROLLOUT_MAX_NUM_SEQS="${WASTE_SD_ROLLOUT_MAX_NUM_SEQS:-64}"
WASTE_SD_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${WASTE_SD_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-512}"
WASTE_SD_ROLLOUT_ENFORCE_EAGER="${WASTE_SD_ROLLOUT_ENFORCE_EAGER:-false}"
WASTE_SD_TRAINER_LOGGER="${WASTE_SD_TRAINER_LOGGER:-console}"
WASTE_SD_TEST_FREQ="${WASTE_SD_TEST_FREQ:--1}"
WASTE_SD_SAVE_FREQ="${WASTE_SD_SAVE_FREQ:--1}"
WASTE_SD_PROJECT_NAME="${WASTE_SD_PROJECT_NAME:-}"
WASTE_SD_EXPERIMENT_NAME="${WASTE_SD_EXPERIMENT_NAME:-}"
WASTE_SD_DEFAULT_LOCAL_DIR="${WASTE_SD_DEFAULT_LOCAL_DIR:-}"
WASTE_SD_RESUME_FROM_PATH="${WASTE_SD_RESUME_FROM_PATH:-}"
WASTE_SD_RESUME_MODE="${WASTE_SD_RESUME_MODE:-}"
WASTE_SD_VAL_BEFORE_TRAIN="${WASTE_SD_VAL_BEFORE_TRAIN:-}"
WASTE_SD_LOG_VAL_GENERATIONS="${WASTE_SD_LOG_VAL_GENERATIONS:-}"
WASTE_SD_VALIDATION_DATA_DIR="${WASTE_SD_VALIDATION_DATA_DIR:-}"
WASTE_SD_STUDENT_MODEL="${WASTE_SD_STUDENT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
WASTE_SD_TEACHER_MODEL="${WASTE_SD_TEACHER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
WASTE_SD_DATA_ROOT="${WASTE_SD_DATA_ROOT:-${REPO_ROOT}/data/verl_fsdp_smoke}"
WASTE_SD_TRAIN_FILE="${WASTE_SD_TRAIN_FILE:-${WASTE_SD_DATA_ROOT}/train.json}"
WASTE_SD_VAL_FILE="${WASTE_SD_VAL_FILE:-${WASTE_SD_DATA_ROOT}/val.json}"
WASTE_SD_TOTAL_TRAINING_STEPS="${WASTE_SD_TOTAL_TRAINING_STEPS:-3}"
WASTE_SD_TOTAL_EPOCHS="${WASTE_SD_TOTAL_EPOCHS:-}"
WASTE_SD_TEMPERATURE="${WASTE_SD_TEMPERATURE:-1.0}"
WASTE_SD_GAMMA="${WASTE_SD_GAMMA:-1}"
WASTE_SD_ACTOR_LR="${WASTE_SD_ACTOR_LR:-1e-6}"
WASTE_SD_BLOCK_EVAL_ONLY="${WASTE_SD_BLOCK_EVAL_ONLY:-0}"
WASTE_SD_BLOCK_EVAL_SPLIT="${WASTE_SD_BLOCK_EVAL_SPLIT:-val}"
WASTE_SD_BLOCK_EVAL_MAX_STEPS="${WASTE_SD_BLOCK_EVAL_MAX_STEPS:-}"
WASTE_SD_BLOCK_EVAL_AFTER_TRAIN="${WASTE_SD_BLOCK_EVAL_AFTER_TRAIN:-0}"
WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR="${WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR:-}"

if [[ ! -f "${WASTE_SD_TRAIN_FILE}" ]]; then
  echo "[waste_sd] Missing train file: ${WASTE_SD_TRAIN_FILE}" >&2
  echo "[waste_sd] Set WASTE_SD_TRAIN_FILE or WASTE_SD_DATA_ROOT to a valid path under /work5." >&2
  exit 1
fi
if [[ ! -f "${WASTE_SD_VAL_FILE}" ]]; then
  echo "[waste_sd] Missing val file: ${WASTE_SD_VAL_FILE}" >&2
  echo "[waste_sd] Set WASTE_SD_VAL_FILE or WASTE_SD_DATA_ROOT to a valid path under /work5." >&2
  exit 1
fi

# Save rollout data only in debug runs.
ROLLOUT_SAVE_OVERRIDES=()
DISTILL_DEBUG_OVERRIDES=()
if [[ "${WASTE_SD_DEBUG:-0}" == "1" ]]; then
  export VERL_DEBUG_DEVICE_BINDING="${VERL_DEBUG_DEVICE_BINDING:-1}"
  export WASTE_SD_ROLLOUT_DATA_DIR="${WASTE_SD_ROLLOUT_DATA_DIR:-${REPO_ROOT}/outputs/waste_sd_rollout/$(date +%Y%m%d_%H%M%S)}"
  export WASTE_SD_DISTILL_DEBUG_DIR="${WASTE_SD_DISTILL_DEBUG_DIR:-${REPO_ROOT}/outputs/waste_sd_debug/$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${WASTE_SD_ROLLOUT_DATA_DIR}"
  mkdir -p "${WASTE_SD_DISTILL_DEBUG_DIR}"
  ROLLOUT_SAVE_OVERRIDES+=("+trainer.rollout_data_dir=${WASTE_SD_ROLLOUT_DATA_DIR}")
  DISTILL_DEBUG_OVERRIDES+=("distill.debug.enable=true")
  DISTILL_DEBUG_OVERRIDES+=("distill.debug.dump_dir=${WASTE_SD_DISTILL_DEBUG_DIR}")
else
  DISTILL_DEBUG_OVERRIDES+=("distill.debug.enable=false")
fi

TRAINER_EXTRA_OVERRIDES=()
if [[ -n "${WASTE_SD_PROJECT_NAME}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.project_name=${WASTE_SD_PROJECT_NAME}")
fi
if [[ -n "${WASTE_SD_EXPERIMENT_NAME}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.experiment_name=${WASTE_SD_EXPERIMENT_NAME}")
fi
if [[ -n "${WASTE_SD_TOTAL_EPOCHS}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.total_epochs=${WASTE_SD_TOTAL_EPOCHS}")
fi
if [[ -n "${WASTE_SD_DEFAULT_LOCAL_DIR}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.default_local_dir=${WASTE_SD_DEFAULT_LOCAL_DIR}")
fi
if [[ -n "${WASTE_SD_RESUME_FROM_PATH}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.resume_from_path=${WASTE_SD_RESUME_FROM_PATH}")
fi
if [[ -n "${WASTE_SD_RESUME_MODE}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.resume_mode=${WASTE_SD_RESUME_MODE}")
fi
if [[ -n "${WASTE_SD_VAL_BEFORE_TRAIN}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.val_before_train=${WASTE_SD_VAL_BEFORE_TRAIN}")
fi
if [[ -n "${WASTE_SD_LOG_VAL_GENERATIONS}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.log_val_generations=${WASTE_SD_LOG_VAL_GENERATIONS}")
fi
if [[ -n "${WASTE_SD_VALIDATION_DATA_DIR}" ]]; then
  TRAINER_EXTRA_OVERRIDES+=("trainer.validation_data_dir=${WASTE_SD_VALIDATION_DATA_DIR}")
fi

BLOCK_EVAL_OVERRIDES=(
  "+trainer.block_eval_only=${WASTE_SD_BLOCK_EVAL_ONLY}"
  "+trainer.block_eval_split=${WASTE_SD_BLOCK_EVAL_SPLIT}"
  "+trainer.block_eval_after_train=${WASTE_SD_BLOCK_EVAL_AFTER_TRAIN}"
)
if [[ -n "${WASTE_SD_BLOCK_EVAL_MAX_STEPS}" ]]; then
  BLOCK_EVAL_OVERRIDES+=("+trainer.block_eval_max_steps=${WASTE_SD_BLOCK_EVAL_MAX_STEPS}")
fi
if [[ -n "${WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR}" ]]; then
  BLOCK_EVAL_OVERRIDES+=("+trainer.block_eval_rollout_data_dir=${WASTE_SD_BLOCK_EVAL_ROLLOUT_DATA_DIR}")
fi

RUNTIME_ENV_OVERRIDES=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_PRELOAD="
  "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${WASTE_SD_RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV}"
)
for _wb_env in WANDB_API_KEY WANDB_MODE WANDB_BASE_URL WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_ID WANDB_RESUME WANDB_DIR; do
  _wb_val="${!_wb_env:-}"
  if [[ -n "${_wb_val}" ]]; then
    _wb_val_escaped="${_wb_val//\\/\\\\}"
    _wb_val_escaped="${_wb_val_escaped//\"/\\\"}"
    RUNTIME_ENV_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.${_wb_env}=\"${_wb_val_escaped}\"")
  fi
done
if [[ -n "${TMPDIR:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=${TMPDIR}")
fi
if [[ -n "${TEMP:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.TEMP=${TEMP}")
fi
if [[ -n "${TMP:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.TMP=${TMP}")
fi
if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
  )
fi
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  )
fi
if [[ -n "${VERL_DEBUG_DEVICE_BINDING:-}" ]]; then
  RUNTIME_ENV_OVERRIDES+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_DEBUG_DEVICE_BINDING=${VERL_DEBUG_DEVICE_BINDING}"
  )
fi
"${PYTHON_BIN}" -m recipe.waste_sd.main_teacher_on_policy \
  actor_rollout_ref.model.path="${WASTE_SD_STUDENT_MODEL}" \
  +actor_rollout_ref.ref.model.path="${WASTE_SD_TEACHER_MODEL}" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.tokenizer_worker_num=1 \
  actor_rollout_ref.rollout.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.temperature="${WASTE_SD_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${WASTE_SD_TEMPERATURE}" \
  distill.loss_type=fkl \
  distill.q_source=local_ref \
  distill.strict=true \
  distill.gamma="${WASTE_SD_GAMMA}" \
  distill.staleness_max_version_gap=0 \
  distill.rollout_target=actor \
  data.train_files="${WASTE_SD_TRAIN_FILE}" \
  data.val_files="${WASTE_SD_VAL_FILE}" \
  trainer.total_training_steps="${WASTE_SD_TOTAL_TRAINING_STEPS}" \
  trainer.n_gpus_per_node="${WASTE_SD_N_GPUS_PER_NODE}" \
  trainer.nnodes="${WASTE_SD_NNODES}" \
  trainer.save_freq="${WASTE_SD_SAVE_FREQ}" \
  trainer.test_freq="${WASTE_SD_TEST_FREQ}" \
  trainer.logger="${WASTE_SD_TRAINER_LOGGER}" \
  "${TRAINER_EXTRA_OVERRIDES[@]}" \
  "${BLOCK_EVAL_OVERRIDES[@]}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WASTE_SD_UPDATE_WEIGHTS_BUCKET_MB}" \
  "${ROLLOUT_SAVE_OVERRIDES[@]}" \
  "${DISTILL_DEBUG_OVERRIDES[@]}" \
  "+ray_kwargs.ray_init.address=${WASTE_SD_RAY_ADDRESS}" \
  "+ray_kwargs.ray_init._temp_dir=${WASTE_SD_RAY_TMP_DIR}" \
  "${RUNTIME_ENV_OVERRIDES[@]}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  data.train_batch_size="${WASTE_SD_TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${WASTE_SD_VAL_BATCH_SIZE}" \
  data.dataloader_num_workers=0 \
  data.max_prompt_length="${WASTE_SD_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${WASTE_SD_MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts="${WASTE_SD_FILTER_OVERLONG_PROMPTS}" \
  actor_rollout_ref.rollout.agent.num_workers="${WASTE_SD_ROLLOUT_AGENT_NUM_WORKERS}" \
  actor_rollout_ref.actor.strategy=fsdp \
  critic.strategy=fsdp \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${WASTE_SD_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${WASTE_SD_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${WASTE_SD_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  critic.ppo_micro_batch_size_per_gpu="${WASTE_SD_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.prompt_length="${WASTE_SD_ROLLOUT_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${WASTE_SD_ROLLOUT_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${WASTE_SD_ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager="${WASTE_SD_ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_seqs="${WASTE_SD_ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${WASTE_SD_ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.actor.optim.lr="${WASTE_SD_ACTOR_LR}" \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.mtp.enable=False \
  actor_rollout_ref.model.mtp.enable_rollout=False
