#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASTE_SD_ROLLOUT_TARGET="${WASTE_SD_ROLLOUT_TARGET:-actor}"

case "${WASTE_SD_ROLLOUT_TARGET}" in
  actor)
    exec bash "${SCRIPT_DIR}/run_teacher_on_policy.sh" "$@"
    ;;
  local_ref)
    exec bash "${SCRIPT_DIR}/run_teacher_off_policy.sh" "$@"
    ;;
  *)
    echo "[waste_sd] Unsupported WASTE_SD_ROLLOUT_TARGET=${WASTE_SD_ROLLOUT_TARGET}" >&2
    echo "[waste_sd] Expected one of: actor, local_ref" >&2
    exit 1
    ;;
esac
