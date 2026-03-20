#!/bin/bash
#SBATCH --job-name=verl-deltaai-distillsd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --partition=ghx4
#SBATCH --constraint="projects"
#SBATCH --account=<your-account>

set -euo pipefail

module reset
module list

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/$USER/apptainer-cache}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

mkdir -p "${APPTAINER_CACHEDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

REPO_ROOT="${REPO_ROOT:-/projects/<your-project>/On-Policy-Distillation}"
SIF_PATH="${SIF_PATH:-/projects/<your-project>/containers/verl-deltaai-distillsd.sif}"
DELTAAI_PYTHONUSERBASE="${DELTAAI_PYTHONUSERBASE:-/scratch/$USER/verl-deltaai/distillsd-userbase}"

mkdir -p "${DELTAAI_PYTHONUSERBASE}"

srun apptainer exec --nv \
  --bind /projects,/scratch \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF_PATH}" \
  env DELTAAI_PYTHONUSERBASE="${DELTAAI_PYTHONUSERBASE}" \
  /opt/verl-deltaai/scripts/run_in_deltaai_env.sh distillsd "${REPO_ROOT}" -- \
  python /opt/verl-deltaai/scripts/smoke_imports.py distillsd
