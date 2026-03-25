#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_NAME="${WASTE_SD_VLLM_ENV_NAME:-verlsd}"

TARGET_MODEL_PATH="${WASTE_SD_VLLM_TARGET_MODEL:-Qwen/Qwen3-1.7B}"
DRAFT_MODEL_PATH="${WASTE_SD_VLLM_DRAFT_MODEL:?Set WASTE_SD_VLLM_DRAFT_MODEL to the draft HF path or model id.}"
INPUT_DATA="${WASTE_SD_VLLM_INPUT_DATA:-${PROJECT_ROOT}/data/gsm8k_gkd/test.parquet}"
OUTPUT_DIR="${WASTE_SD_VLLM_OUTPUT_DIR:-${ROOT}/outputs}"
RUN_LABEL="${WASTE_SD_VLLM_RUN_LABEL:-vllm_block_eval}"
TIMESTAMP="${WASTE_SD_VLLM_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TMP_DIR="${WASTE_SD_VLLM_TMP_DIR:-${TMPDIR:-/tmp}/waste_sd_vllm_${USER}}"

VISIBLE_DEVICES="${WASTE_SD_VLLM_VISIBLE_DEVICES:-6,7}"
PARALLEL_MODE="${WASTE_SD_VLLM_PARALLEL_MODE:-dp}"
TP_SIZE="${WASTE_SD_VLLM_TP_SIZE:-}"

MAX_PROMPTS="${WASTE_SD_VLLM_MAX_PROMPTS:--1}"
SAMPLES_PER_PROMPT="${WASTE_SD_VLLM_SAMPLES_PER_PROMPT:-1}"
MAX_NEW_TOKENS="${WASTE_SD_VLLM_MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LEN="${WASTE_SD_VLLM_MAX_MODEL_LEN:-4096}"
TEMPERATURE="${WASTE_SD_VLLM_TEMPERATURE:-1.0}"
DRAFT_TEMPERATURE="${WASTE_SD_VLLM_DRAFT_TEMPERATURE:-}"
TOP_P="${WASTE_SD_VLLM_TOP_P:-1.0}"
TOP_K="${WASTE_SD_VLLM_TOP_K:-0}"

# gamma=6 in waste_sd corresponds to a draft budget of 6. In vLLM DELTA tracing,
# a full accepted block can therefore have length 7 because it includes the bonus token.
NUM_SPECULATIVE_TOKENS="${WASTE_SD_VLLM_NUM_SPECULATIVE_TOKENS:-6}"
GPU_MEMORY_UTILIZATION="${WASTE_SD_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_BATCHED_TOKENS="${WASTE_SD_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${WASTE_SD_VLLM_MAX_NUM_SEQS:-128}"
LOG_EVERY="${WASTE_SD_VLLM_LOG_EVERY:-8}"

ENABLE_THINKING="${WASTE_SD_VLLM_ENABLE_THINKING:-1}"
NO_ROLLOUT_INSTRUCTION="${WASTE_SD_VLLM_NO_ROLLOUT_INSTRUCTION:-1}"
STRICT_DRAFT_PROBS="${WASTE_SD_VLLM_STRICT_DRAFT_PROBS:-1}"
if [[ -n "${WASTE_SD_VLLM_ENFORCE_EAGER+x}" ]]; then
  ENFORCE_EAGER="${WASTE_SD_VLLM_ENFORCE_EAGER}"
else
  ENFORCE_EAGER="0"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TMP_DIR}"

OUT="${OUTPUT_DIR}/${RUN_LABEL}_g${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}.jsonl"
LOG="${OUTPUT_DIR}/${RUN_LABEL}_g${NUM_SPECULATIVE_TOKENS}_${TIMESTAMP}.log"

EXTRA_ARGS=()
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-thinking)
fi
if [[ "${NO_ROLLOUT_INSTRUCTION}" == "1" ]]; then
  EXTRA_ARGS+=(--no-rollout-instruction)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${STRICT_DRAFT_PROBS}" == "1" ]]; then
  EXTRA_ARGS+=(--strict-draft-probs)
fi
if [[ -n "${DRAFT_TEMPERATURE}" ]]; then
  EXTRA_ARGS+=(--draft-temperature "${DRAFT_TEMPERATURE}")
fi
if [[ "${PARALLEL_MODE}" == "tp" && -n "${TP_SIZE}" ]]; then
  EXTRA_ARGS+=(--tp-size "${TP_SIZE}")
fi

echo "Output: ${OUT}"
echo "Log: ${LOG}"

RUN_CMD=(
  conda run
  --no-capture-output
  -n "${ENV_NAME}"
  python
  "${ROOT}/recipe/waste_sd/eval_block_counts_vllm.py"
  --target-model-path "${TARGET_MODEL_PATH}"
  --draft-model-path "${DRAFT_MODEL_PATH}"
  --input-data "${INPUT_DATA}"
  --output-jsonl "${OUT}"
  --tmp-dir "${TMP_DIR}"
  --visible-devices "${VISIBLE_DEVICES}"
  --parallel-mode "${PARALLEL_MODE}"
  --speculative-method draft_model
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
  --max-prompts "${MAX_PROMPTS}"
  --samples-per-prompt "${SAMPLES_PER_PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --log-every "${LOG_EVERY}"
)
RUN_CMD+=("${EXTRA_ARGS[@]}")

CUDA_DEVICE_ORDER=PCI_BUS_ID "${RUN_CMD[@]}" 2>&1 | tee "${LOG}"

conda run --no-capture-output -n "${ENV_NAME}" python - "${OUT}" "${MAX_NEW_TOKENS}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_new_tokens = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    print(path)
    print("num_rows", 0)
    raise SystemExit(0)
verify = [row["spec_verify_ct"] for row in rows]
completion = [row["completion_tokens"] for row in rows]
accepted = [row["spec_accepted_tokens"] for row in rows]
ratio = [v / c for v, c in zip(verify, completion) if c > 0]
capped = sum(1 for value in completion if value >= max_new_tokens)

print(path)
print("num_rows", len(rows))
print("mean_spec_verify_ct", sum(verify) / len(verify))
print("mean_completion_tokens", sum(completion) / len(completion))
print("mean_spec_accepted_tokens", sum(accepted) / len(accepted))
print("mean_verify_per_token", sum(ratio) / len(ratio))
print("median_spec_verify_ct", statistics.median(verify))
print("median_completion_tokens", statistics.median(completion))
print("capped_max_new_tokens", capped)
PY
