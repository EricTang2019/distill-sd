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

from typing import Iterable, Optional
import warnings

import numpy as np
import torch


def _normalize_accept_lens(spec_accept_lens: Optional[Iterable[int]]) -> list[int]:
    if spec_accept_lens is None:
        return []
    return [int(x) for x in spec_accept_lens]


def build_strict_weights(
    spec_accept_lens: Optional[Iterable[int]],
    response_valid_len: int,
    gamma: int,
    *,
    strict: bool = True,
    full_block_participate: Optional[bool] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build token-level waste-aware weights from SD accept lengths.

    Weight definition (strictly aligned with notes.tex):
      - Let `s` be the within-block accepted-count before emitting this token (i.e., `S_n`).
      - Then `w=max(gamma-s, 0)`.
      - This naturally handles all cases, including full blocks, without special branching.

    Args:
        spec_accept_lens: Accepted lengths per speculative block.
        response_valid_len: Number of valid response tokens (from `response_mask.sum()`).
        gamma: Draft budget length.
        strict: Whether to fail on alignment mismatch.
        full_block_participate: Deprecated compatibility flag (ignored if provided).
        device: Device of output tensor.
        dtype: Output tensor dtype.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if response_valid_len < 0:
        raise ValueError(f"response_valid_len must be non-negative, got {response_valid_len}")

    # Backward compatibility only. Full blocks are already covered by the same formula.
    if full_block_participate is not None:
        warnings.warn(
            "full_block_participate is deprecated and ignored. "
            "Full blocks are always handled by the strict weight formula.",
            stacklevel=2,
        )

    accept_lens = _normalize_accept_lens(spec_accept_lens)

    for block_len in accept_lens:
        if block_len < 0:
            raise ValueError(f"accept length must be non-negative, got {block_len}")

    # Vectorized weight computation: offset[i] = S_n before token i, weight = max(gamma - S_n, 0).
    # Each block of length L contributes offsets [0, 1, ..., L-1]; much faster than a Python loop
    # over all tokens for long responses.
    if not accept_lens:
        offsets = np.empty(0, dtype=np.int32)
    else:
        offsets = np.concatenate([np.arange(bl, dtype=np.int32) for bl in accept_lens])
    weights_np = np.maximum(gamma - offsets, 0).astype(np.float32)

    alignment_ok = len(weights_np) == response_valid_len
    if strict and not alignment_ok:
        raise ValueError(
            "Strict alignment failed: "
            f"sum(spec_accept_lens) mapped tokens={len(weights_np)} vs response_valid_len={response_valid_len}"
        )

    if len(weights_np) < response_valid_len:
        weights_np = np.concatenate([weights_np, np.zeros(response_valid_len - len(weights_np), dtype=np.float32)])
    elif len(weights_np) > response_valid_len:
        weights_np = weights_np[:response_valid_len]

    weight_tensor = torch.tensor(weights_np, dtype=dtype, device=device)
    nonzero = (weight_tensor > 0).float()

    stats = {
        "distill/weight_nonzero_ratio": nonzero.mean().item() if response_valid_len > 0 else 0.0,
        "distill/weight_mean": weight_tensor.mean().item() if response_valid_len > 0 else 0.0,
        "distill/weight_max": weight_tensor.max().item() if response_valid_len > 0 else 0.0,
        "distill/weight_sum": weight_tensor.sum().item(),
        "distill/strict_alignment_ok": float(alignment_ok),
    }
    return weight_tensor, stats
