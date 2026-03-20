#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELTA_DOCKER_DIR="${ROOT}/docker/deltaai"

BASE_TAG="${BASE_TAG:-verl-deltaai-base:25.05-arm64}"
DISTILLSD_CORE_TAG="${DISTILLSD_CORE_TAG:-verl-deltaai-distillsd-core:local-arm64}"
DISTILLSD_TAG="${DISTILLSD_TAG:-verl-deltaai-distillsd:local-arm64}"
VERLSD_CORE_TAG="${VERLSD_CORE_TAG:-verl-deltaai-verlsd-core:local-arm64}"
VERLSD_TAG="${VERLSD_TAG:-verl-deltaai-verlsd:local-arm64}"
PLATFORM="${PLATFORM:-linux/arm64}"
BUILD_OUTPUT_FLAG="${BUILD_OUTPUT_FLAG:---load}"
BUILD_PROGRESS="${BUILD_PROGRESS:-plain}"
BUILD_TARGETS="${BUILD_TARGETS:-base,distillsd,verlsd}"

build_requested() {
  local name="$1"
  [[ ",${BUILD_TARGETS}," == *",${name},"* ]]
}

uses_local_base_handoff() {
  [[ "${BUILD_OUTPUT_FLAG}" == "--load" || "${BUILD_OUTPUT_FLAG}" == --output=type=docker* ]]
}

build_child_image() {
  local dockerfile="$1"
  local image_tag="$2"
  local parent_image="$3"

  if uses_local_base_handoff; then
    env -u BUILDX_BUILDER DOCKER_BUILDKIT=1 docker build \
      --progress "${BUILD_PROGRESS}" \
      --platform "${PLATFORM}" \
      --pull=false \
      --build-arg BASE_IMAGE="${parent_image}" \
      -f "${dockerfile}" \
      -t "${image_tag}" \
      "${ROOT}"
  else
    docker buildx build \
      "${BUILD_OUTPUT_FLAG}" \
      --progress "${BUILD_PROGRESS}" \
      --platform "${PLATFORM}" \
      --build-arg BASE_IMAGE="${parent_image}" \
      -f "${dockerfile}" \
      -t "${image_tag}" \
      "${ROOT}"
  fi
}

if build_requested base; then
  docker buildx build \
    "${BUILD_OUTPUT_FLAG}" \
    --progress "${BUILD_PROGRESS}" \
    --platform "${PLATFORM}" \
    -f "${DELTA_DOCKER_DIR}/Dockerfile.base.arm64" \
    -t "${BASE_TAG}" \
    "${ROOT}"
fi

if build_requested distillsd-core || build_requested distillsd; then
  build_child_image "${DELTA_DOCKER_DIR}/Dockerfile.distillsd.core.arm64" "${DISTILLSD_CORE_TAG}" "${BASE_TAG}"
fi

if build_requested distillsd; then
  build_child_image "${DELTA_DOCKER_DIR}/Dockerfile.distillsd.arm64" "${DISTILLSD_TAG}" "${DISTILLSD_CORE_TAG}"
fi

if build_requested verlsd-core || build_requested verlsd; then
  build_child_image "${DELTA_DOCKER_DIR}/Dockerfile.verlsd.core.arm64" "${VERLSD_CORE_TAG}" "${BASE_TAG}"
fi

if build_requested verlsd; then
  build_child_image "${DELTA_DOCKER_DIR}/Dockerfile.verlsd.arm64" "${VERLSD_TAG}" "${VERLSD_CORE_TAG}"
fi

echo "Build targets: ${BUILD_TARGETS}"
if build_requested base; then
  echo "Built: ${BASE_TAG}"
fi
if build_requested distillsd; then
  echo "Built core: ${DISTILLSD_CORE_TAG}"
  echo "Built: ${DISTILLSD_TAG}"
fi
if build_requested distillsd-core && ! build_requested distillsd; then
  echo "Built core: ${DISTILLSD_CORE_TAG}"
fi
if build_requested verlsd; then
  echo "Built core: ${VERLSD_CORE_TAG}"
  echo "Built: ${VERLSD_TAG}"
fi
if build_requested verlsd-core && ! build_requested verlsd; then
  echo "Built core: ${VERLSD_CORE_TAG}"
fi
