# DeltaAI Container Plan

This directory contains a DeltaAI/GH200-oriented container setup for the current
`verl` workspace.

Goals:

- keep two environments aligned with the local x86 conda environments:
  - `distillsd` for `sglang` training
  - `verlsd` for `vllm` strict evaluation
- avoid changing the main `verl` codebase
- make runtime debugging fast via exact version validation and lightweight smoke tests
- target DeltaAI's `linux/arm64` + `Apptainer` workflow

## Version targets

The exact Python-package targets are stored in:

- `env.distillsd.versions.yaml`
- `env.verlsd.versions.yaml`
- `env.distillsd.pip-freeze.txt`
- `env.verlsd.pip-freeze.txt`
- `env.distillsd.conda-explicit.txt`
- `env.verlsd.conda-explicit.txt`
- `env.distillsd.constraints.txt`
- `env.verlsd.constraints.txt`

The YAML files are the source of truth for behavior-critical validation.
The `pip-freeze` and `conda-explicit` files are the full local snapshots kept for
debugging and future rebuilds.
The `constraints.txt` files are sanitized, arm64-oriented versions of the local
freezes used to keep transitive dependency resolution from drifting during image builds.

## Files

- `Dockerfile.base.arm64`
  - shared ARM base image
  - copies validation scripts into the image
- `Dockerfile.distillsd.arm64`
  - `sglang` training environment
- `Dockerfile.verlsd.arm64`
  - `vllm` strict-eval environment
- `scripts/validate_env.py`
  - fails fast if imported versions do not match the target snapshot
- `scripts/generate_constraints.py`
  - turns raw local `pip freeze` output into an arm64-safe `constraints.txt`
- `scripts/run_in_deltaai_env.sh`
  - runs local repo code by prepending the mounted repo to `PYTHONPATH`
  - optionally honors `DELTAAI_PYTHONUSERBASE` for fast user-level overlays
- `scripts/smoke_imports.py`
  - lightweight import smoke tests for both environments
- `build_deltaai_images.sh`
  - local `docker buildx` helper
- `sbatch.smoke.distillsd.sh`
  - smallest useful DeltaAI smoke test for the `sglang` training image
- `sbatch.smoke.verlsd.sh`
  - smallest useful DeltaAI smoke test for the `vllm` eval image

## Design notes

1. The images are split on purpose.

   Local usage already splits into:
   - `distillsd`: `sglang 0.5.6`
   - `verlsd`: `vllm 0.15.1`

   Keeping them separate reduces ARM/GH200 build risk and makes it easier to
   debug failures.

2. "Exact" means exact Python package versions for the behavior-critical stack.

   Because DeltaAI is `arm64`, the system libraries below Python cannot be
   byte-identical to the current local x86 environments. The exactness target is:

   - Python package versions
   - CUDA extension package versions
   - runtime import compatibility

   The `validate_env.py` step enforces this for the core stack, while the full
   freeze files remain available to inspect or replay differences.

3. The local repo is not baked into the image.

   At runtime, use:

   ```bash
   /opt/verl-deltaai/scripts/run_in_deltaai_env.sh <distillsd|verlsd> /path/to/repo -- <command...>
   ```

   This keeps debugging fast and avoids rebuilding the heavy image for every code change.

4. User-level overlays are supported.

   DeltaAI's Apptainer documentation explicitly allows `pip install --user`
   against the container's Python. If you set `DELTAAI_PYTHONUSERBASE`,
   `run_in_deltaai_env.sh` will prepend that overlay to `PYTHONPATH` and `PATH`.
   This is useful for quick debugging without rebuilding the base image.

## Build examples

Build locally with Docker Buildx on an ARM-capable builder or a remote builder:

```bash
cd /path/to/On-Policy-Distillation/verl
bash docker/deltaai/build_deltaai_images.sh
```

If you want custom tags:

```bash
BASE_TAG=myrepo/verl-deltaai-base:25.05-arm64 \
DISTILLSD_TAG=myrepo/verl-deltaai-distillsd:arm64 \
VERLSD_TAG=myrepo/verl-deltaai-verlsd:arm64 \
bash docker/deltaai/build_deltaai_images.sh
```

If you only want the minimal-risk `base -> verlsd` path:

```bash
BUILD_TARGETS=base,verlsd \
bash docker/deltaai/build_deltaai_images.sh
```

If you want to push directly from a remote buildx builder instead of loading the
image into the local daemon:

```bash
BUILD_OUTPUT_FLAG=--push \
BASE_TAG=myrepo/verl-deltaai-base:25.05-arm64 \
DISTILLSD_TAG=myrepo/verl-deltaai-distillsd:arm64 \
VERLSD_TAG=myrepo/verl-deltaai-verlsd:arm64 \
bash docker/deltaai/build_deltaai_images.sh
```

## DeltaAI / Apptainer flow

After pushing the image to a registry:

```bash
apptainer pull verl-deltaai-distillsd.sif docker://myrepo/verl-deltaai-distillsd:arm64
apptainer pull verl-deltaai-verlsd.sif docker://myrepo/verl-deltaai-verlsd:arm64
```

Recommended cache placement on DeltaAI:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer-cache
export HF_HOME=/scratch/$USER/hf
export TRANSFORMERS_CACHE=/scratch/$USER/hf/transformers
export TMPDIR=/scratch/$USER/tmp
export DELTAAI_PYTHONUSERBASE=/scratch/$USER/verl-deltaai/userbase
```

For multi-node jobs on DeltaAI, the official job guide uses:

```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
```

The helper launcher in `scripts/run_in_deltaai_env.sh` will apply these defaults automatically when `SLURM_NNODES > 1`.

## Smoke-test examples

`distillsd`:

```bash
apptainer exec --nv --bind /projects,/scratch,/path/to/On-Policy-Distillation \
  verl-deltaai-distillsd.sif \
  /opt/verl-deltaai/scripts/run_in_deltaai_env.sh distillsd /path/to/On-Policy-Distillation \
  -- python /opt/verl-deltaai/scripts/smoke_imports.py distillsd
```

`verlsd`:

```bash
apptainer exec --nv --bind /projects,/scratch,/path/to/On-Policy-Distillation \
  verl-deltaai-verlsd.sif \
  /opt/verl-deltaai/scripts/run_in_deltaai_env.sh verlsd /path/to/On-Policy-Distillation \
  -- python /opt/verl-deltaai/scripts/smoke_imports.py verlsd
```

If either smoke test fails, fix the image before launching expensive jobs.

Batch-script templates are also included:

- `sbatch.smoke.distillsd.sh`
- `sbatch.smoke.verlsd.sh`

The sample `sbatch` scripts default to `--partition=ghx4` and
`--constraint="projects"` because the mounted repo and container paths in the
templates live under `/projects`, matching DeltaAI's file-system dependency guidance.

## Current limitations

- The Dockerfiles pin the behavior-critical packages to the current local environment snapshots.
- The full local `pip freeze` and `conda list --explicit` snapshots are stored beside the Dockerfiles for auditing.
- Whether each exact pin is available as an `arm64` wheel depends on upstream availability.
- If an exact wheel is not available on `arm64`, the build will fail early rather than silently drifting to a different version.
