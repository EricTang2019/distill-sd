# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SYNC_MODE_ENV = "VERL_SGLANG_STANDALONE_SYNC_MODE"
_LEGACY_TARGET_ONLY_ENV = "VERL_SGLANG_STANDALONE_TARGET_ONLY_SYNC"
_PATCH_MARKER = "_verl_standalone_sync_mode_patched"


def _legacy_target_only_enabled() -> bool:
    value = os.getenv(_LEGACY_TARGET_ONLY_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_sync_mode() -> str:
    # New explicit mode takes precedence.
    mode = os.getenv(_SYNC_MODE_ENV, "").strip().lower()
    if mode:
        return mode

    # Backward compatibility with previous single-flag behavior.
    if _legacy_target_only_enabled():
        return "target_only"

    return "both"


def apply_standalone_target_only_patch() -> None:
    """Patch StandaloneWorker weight sync mode for speculative decoding.

    Supported modes via environment variable `VERL_SGLANG_STANDALONE_SYNC_MODE`:
      - `both`: update draft and target (default upstream behavior)
      - `target_only`: update target only
      - `draft_only`: update draft only

    For backward compatibility, `VERL_SGLANG_STANDALONE_TARGET_ONLY_SYNC=1`
    maps to `target_only` when `VERL_SGLANG_STANDALONE_SYNC_MODE` is unset.
    """
    sync_mode = _resolve_sync_mode()
    if sync_mode == "both":
        return

    if sync_mode not in {"target_only", "draft_only"}:
        logger.warning(
            "Skip standalone sync-mode patch: invalid %s=%s (expected both/target_only/draft_only).",
            _SYNC_MODE_ENV,
            sync_mode,
        )
        return

    try:
        from sglang.srt.speculative.standalone_worker import StandaloneWorker
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    except Exception as e:
        logger.warning("Skip standalone target-only patch: failed to import sglang modules: %s", e)
        return

    if getattr(StandaloneWorker, _PATCH_MARKER, False):
        return

    def _update_weights_from_tensor(self: Any, recv_req: Any):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = True, ""

        if sync_mode in {"draft_only"}:
            success, message = self.model_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
            return success, message

        if sync_mode in {"target_only"}:
            success, message = self.target_worker.model_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
            return success, message

        return success, message

    StandaloneWorker.update_weights_from_tensor = _update_weights_from_tensor
    setattr(StandaloneWorker, _PATCH_MARKER, True)
    logger.warning(
        "Applied standalone weight-sync patch (mode=%s, env %s=%s).",
        sync_mode,
        _SYNC_MODE_ENV,
        sync_mode,
    )
