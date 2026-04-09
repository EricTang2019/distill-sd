#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVS_DIR="${REPO_ROOT}/envs"
VALIDATE_SCRIPT="${SCRIPT_DIR}/validate_env.py"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/setup/bootstrap_envs.sh [all|distillsd|verlsd] [--recreate]

Examples:
  bash scripts/setup/bootstrap_envs.sh all
  bash scripts/setup/bootstrap_envs.sh distillsd
  bash scripts/setup/bootstrap_envs.sh verlsd --recreate

This exact bootstrap currently targets linux-aarch64 only.
It recreates the two officially supported environments for this repo:
  - distillsd: training / SGLang
  - verlsd: strict evaluation / vLLM
USAGE
}

TARGET="${1:-all}"
if [[ "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

RECREATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap only supports Linux." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
  echo "This exact bootstrap currently supports linux-aarch64 only. Current machine: ${ARCH}" >&2
  exit 1
fi

find_solver() {
  local candidate
  for candidate in micromamba mamba conda; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

SOLVER="$(find_solver || true)"
if [[ -z "${SOLVER}" ]]; then
  echo "Could not find micromamba, mamba, or conda on PATH." >&2
  exit 1
fi

targets_for() {
  case "$1" in
    all)
      echo distillsd
      echo verlsd
      ;;
    distillsd|distillsd-deltaai)
      echo distillsd
      ;;
    verlsd|verlsd-deltaai)
      echo verlsd
      ;;
    *)
      echo "Unknown target: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

env_exists() {
  local env_name="$1"
  "${SOLVER}" run -n "${env_name}" python -c 'import sys; sys.exit(0)' >/dev/null 2>&1
}

create_env() {
  local env_name="$1"
  local conda_file="$2"
  echo "[${env_name}] creating conda environment from ${conda_file}"
  "${SOLVER}" create -y -n "${env_name}" --file "${conda_file}"
}

remove_env() {
  local env_name="$1"
  echo "[${env_name}] removing existing environment"
  "${SOLVER}" env remove -y -n "${env_name}"
}

run_in_env() {
  local env_name="$1"
  shift
  "${SOLVER}" run -n "${env_name}" "$@"
}

sync_pip_snapshot() {
  local env_name="$1"
  local pip_file="$2"
  echo "[${env_name}] syncing pip snapshot from ${pip_file}"
  run_in_env "${env_name}" python -m pip install --no-deps -r "${pip_file}"
}

verify_pip_snapshot() {
  local env_name="$1"
  local pip_file="$2"
  local actual_freeze
  actual_freeze="$(mktemp)"
  trap 'rm -f "${actual_freeze}"' RETURN
  run_in_env "${env_name}" python -m pip freeze --exclude-editable > "${actual_freeze}"
  if ! diff -u "${pip_file}" "${actual_freeze}" >/dev/null; then
    echo "[${env_name}] installed pip snapshot does not exactly match ${pip_file}" >&2
    diff -u "${pip_file}" "${actual_freeze}" || true
    exit 1
  fi
}

install_repo() {
  local env_name="$1"
  echo "[${env_name}] installing repo in editable mode"
  run_in_env "${env_name}" python -m pip install --no-deps -e "${REPO_ROOT}"
}

validate_env() {
  local env_name="$1"
  local spec_file="$2"
  echo "[${env_name}] validating exact package snapshot"
  run_in_env "${env_name}" python "${VALIDATE_SCRIPT}" "${spec_file}"
}

bootstrap_one() {
  local env_name="$1"
  local conda_file="${ENVS_DIR}/${env_name}.linux-aarch64.conda-explicit.txt"
  local pip_file="${ENVS_DIR}/${env_name}.linux-aarch64.pip-freeze.txt"
  local spec_file="${ENVS_DIR}/${env_name}.versions.yaml"

  if [[ ! -f "${conda_file}" || ! -f "${pip_file}" || ! -f "${spec_file}" ]]; then
    echo "Missing environment snapshot files for ${env_name}." >&2
    exit 1
  fi

  if env_exists "${env_name}"; then
    if [[ "${RECREATE}" -eq 1 ]]; then
      remove_env "${env_name}"
      create_env "${env_name}" "${conda_file}"
    else
      echo "[${env_name}] existing environment found; reusing it"
    fi
  else
    create_env "${env_name}" "${conda_file}"
  fi

  sync_pip_snapshot "${env_name}" "${pip_file}"
  install_repo "${env_name}"
  verify_pip_snapshot "${env_name}" "${pip_file}"
  validate_env "${env_name}" "${spec_file}"
  echo "[${env_name}] ready"
}

while IFS= read -r env_name; do
  [[ -n "${env_name}" ]] || continue
  bootstrap_one "${env_name}"
done < <(targets_for "${TARGET}")

echo
echo "Exact environments are ready. Suggested next commands:"
echo "  ${SOLVER} run -n distillsd python -m recipe.waste_sd.test_waste_sd_components"
echo "  ${SOLVER} run -n verlsd python -m recipe.waste_sd.eval_block_counts_vllm --help"
