#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <distillsd|verlsd> <repo_root> -- <command...>" >&2
  exit 2
fi

ENV_NAME="$1"
REPO_ROOT="$2"
shift 2

if [[ "$1" != "--" ]]; then
  echo "Expected '--' before the command to execute." >&2
  exit 2
fi
shift

case "${ENV_NAME}" in
  distillsd)
    SPEC="/opt/verl-deltaai/env.distillsd.versions.yaml"
    ;;
  verlsd)
    SPEC="/opt/verl-deltaai/env.verlsd.versions.yaml"
    ;;
  *)
    echo "Unknown environment name: ${ENV_NAME}" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -n "${DELTAAI_PYTHONUSERBASE:-}" ]]; then
  export PYTHONUSERBASE="${DELTAAI_PYTHONUSERBASE}"
  USER_SITE="$(python - <<'PY'
import site
print(site.getusersitepackages())
PY
)"
  export PATH="${PYTHONUSERBASE}/bin:${PATH}"
  export PYTHONPATH="${USER_SITE}:${PYTHONPATH}"
fi

if [[ "${SLURM_NNODES:-1}" != "1" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
fi

python /opt/verl-deltaai/scripts/validate_env.py "${SPEC}" --ignore-python-patch

cd "${REPO_ROOT}/verl"
exec "$@"
