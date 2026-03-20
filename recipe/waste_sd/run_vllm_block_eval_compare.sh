#!/usr/bin/env bash
set -euo pipefail

ROOT="/work5/jingwut/On-Policy-Distillation/verl"
SCRIPT="${ROOT}/recipe/waste_sd/run_vllm_block_eval_single.sh"
TIMESTAMP="${WASTE_SD_VLLM_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

UNIFORM_DRAFT_MODEL="${WASTE_SD_VLLM_UNIFORM_DRAFT_MODEL:-/work5/jingwut/On-Policy-Distillation/verl/outputs/offline_uniform_q31p7_to_q30p6b_bs64_lr1e4_450steps_gpu01_20260310_193112/global_step_450/actor/huggingface_merged}"
REMBUDGET_DRAFT_MODEL="${WASTE_SD_VLLM_REMBUDGET_DRAFT_MODEL:-/work5/jingwut/On-Policy-Distillation/verl/outputs/offline_rembudget_q31p7_to_q30p6b_bs64_lr1e4_450steps_gpu23_20260310_193118/global_step_450/actor/huggingface_merged}"

run_one() {
  local label="$1"
  local draft_model="$2"
  echo
  echo "=== ${label} ==="
  WASTE_SD_VLLM_TIMESTAMP="${TIMESTAMP}" \
  WASTE_SD_VLLM_RUN_LABEL="${label}" \
  WASTE_SD_VLLM_DRAFT_MODEL="${draft_model}" \
  bash "${SCRIPT}"
}

run_one "vllm_block_eval_uniform_step450" "${UNIFORM_DRAFT_MODEL}"
run_one "vllm_block_eval_rembudget_step450" "${REMBUDGET_DRAFT_MODEL}"
