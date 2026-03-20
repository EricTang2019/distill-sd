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

from collections.abc import Callable
from typing import Any

import torch

_DISTILL_LOSS_REGISTRY: dict[str, Callable] = {}


def register_distill_loss(name: str):
    def decorator(fn: Callable):
        key = name.lower()
        if key in _DISTILL_LOSS_REGISTRY:
            raise ValueError(f"Duplicate distill loss registration: {name}")
        _DISTILL_LOSS_REGISTRY[key] = fn
        return fn

    return decorator


def get_distill_loss_fn(name: str) -> Callable:
    key = name.lower()
    if key not in _DISTILL_LOSS_REGISTRY:
        raise KeyError(f"Unknown distill loss type {name}. Available: {sorted(_DISTILL_LOSS_REGISTRY.keys())}")
    return _DISTILL_LOSS_REGISTRY[key]


def _extract_teacher_dense(teacher: Any, device: torch.device) -> tuple[torch.Tensor, bool]:
    """Extract dense teacher tensor.

    Returns:
        tensor: teacher tensor on `device`
        is_log_probs: whether `tensor` is already normalized log-probabilities
    """
    if torch.is_tensor(teacher):
        return teacher.to(device=device), False

    if not isinstance(teacher, dict):
        raise TypeError(f"Dense teacher must be Tensor or dict, got {type(teacher)}")

    if "log_probs" in teacher:
        return torch.as_tensor(teacher["log_probs"], dtype=torch.float32, device=device), True
    if "logits" in teacher:
        return torch.as_tensor(teacher["logits"], dtype=torch.float32, device=device), False
    raise KeyError("Dense teacher dict must contain either 'logits' or 'log_probs'")


def _prepare_dense_inputs(
    student_logits: torch.Tensor,
    teacher: Any,
    token_weights: torch.Tensor,
    response_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Normalize [B, T, V] inputs for dense distillation losses."""
    if student_logits.ndim == 2:
        student_logits = student_logits.unsqueeze(0)
    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must be rank-2/3 [..., tokens, vocab], got {student_logits.shape}")

    teacher_tensor, teacher_is_log_probs = _extract_teacher_dense(teacher, device=student_logits.device)
    if teacher_tensor.ndim == 2:
        teacher_tensor = teacher_tensor.unsqueeze(0)
    if teacher_tensor.ndim != 3:
        raise ValueError(f"teacher tensor must be rank-2/3 [..., tokens, vocab], got {teacher_tensor.shape}")

    if token_weights.ndim == 1:
        token_weights = token_weights.unsqueeze(0)
    if token_weights.ndim != 2:
        raise ValueError(f"token_weights must be rank-1/2 [..., tokens], got {token_weights.shape}")
    weights = token_weights.to(device=student_logits.device, dtype=torch.float32)

    mask = None
    if response_mask is not None:
        if response_mask.ndim == 1:
            response_mask = response_mask.unsqueeze(0)
        if response_mask.ndim != 2:
            raise ValueError(f"response_mask must be rank-1/2 [..., tokens], got {response_mask.shape}")
        mask = response_mask.to(device=student_logits.device, dtype=torch.bool)

    student_bsz, student_tokens, student_vocab = student_logits.shape
    teacher_bsz, teacher_tokens, teacher_vocab = teacher_tensor.shape
    weights_bsz, weights_tokens = weights.shape

    expected_bt = (student_bsz, student_tokens)
    if (teacher_bsz, teacher_tokens, teacher_vocab) != (student_bsz, student_tokens, student_vocab):
        raise ValueError(
            "teacher shape mismatch: expected %s, got %s"
            % ((student_bsz, student_tokens, student_vocab), (teacher_bsz, teacher_tokens, teacher_vocab))
        )
    if (weights_bsz, weights_tokens) != expected_bt:
        raise ValueError(
            f"token_weights shape mismatch: expected {expected_bt}, got {(weights_bsz, weights_tokens)}"
        )
    if mask is not None and tuple(mask.shape) != expected_bt:
        raise ValueError(f"response_mask shape mismatch: expected {expected_bt}, got {tuple(mask.shape)}")

    if mask is None:
        mask = torch.ones(expected_bt, dtype=torch.bool, device=student_logits.device)
    return student_logits.float(), teacher_tensor.float(), weights, mask, teacher_is_log_probs


def compute_dense_distill_loss_batched(
    student_logits: torch.Tensor,
    teacher_dense: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute weighted FKL/RKL/TVD with dense teacher outputs [B, T, V]."""
    student_logits, teacher_tensor, weights, mask, teacher_is_log_probs = _prepare_dense_inputs(
        student_logits,
        teacher_dense,
        token_weights,
        response_mask,
    )

    if student_logits.numel() == 0:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, {
            f"distill/{loss_type}_token_div_mean": 0.0,
            f"distill/{loss_type}_token_div_max": 0.0,
            "distill/token_count": 0.0,
        }

    student_logp = torch.log_softmax(student_logits, dim=-1)
    teacher_logp = teacher_tensor if teacher_is_log_probs else torch.log_softmax(teacher_tensor, dim=-1)

    if loss_type == "fkl":
        teacher_prob = torch.exp(teacher_logp)
        # Guard against 0 * (-inf) = NaN when teacher_prob underflows to 0 in float32.
        token_div = torch.sum(
            torch.where(teacher_prob > 0, teacher_prob * (teacher_logp - student_logp), teacher_logp.new_zeros(())),
            dim=-1,
        )
    elif loss_type == "rkl":
        student_prob = torch.exp(student_logp)
        token_div = torch.sum(student_prob * (student_logp - teacher_logp), dim=-1)
    elif loss_type == "tvd":
        student_prob = torch.exp(student_logp)
        teacher_prob = torch.exp(teacher_logp)
        token_div = 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
    else:
        raise ValueError(f"Unsupported loss_type {loss_type}")

    flat_mask = mask.reshape(-1)
    token_div_flat = token_div.reshape(-1)[flat_mask]
    weight_flat = weights.reshape(-1)[flat_mask]

    if token_div_flat.numel() == 0:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, {
            f"distill/{loss_type}_token_div_mean": 0.0,
            f"distill/{loss_type}_token_div_max": 0.0,
            "distill/token_count": 0.0,
        }

    weighted_sum = torch.sum(token_div_flat * weight_flat)
    weight_sum = torch.sum(weight_flat)
    loss = weighted_sum / weight_sum if float(weight_sum.item()) > 0.0 else weighted_sum * 0.0

    metrics = {
        f"distill/{loss_type}_token_div_mean": token_div_flat.mean().item(),
        f"distill/{loss_type}_token_div_max": token_div_flat.max().item(),
        "distill/token_count": float(token_div_flat.numel()),
    }
    return loss, weighted_sum, weight_sum, metrics


def compute_teacher_greedy_nll_batched(
    student_logits: torch.Tensor,
    teacher_dense: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute masked weighted NLL against the teacher greedy token at each sampled prefix."""
    student_logits, teacher_tensor, weights, mask, _ = _prepare_dense_inputs(
        student_logits,
        teacher_dense,
        token_weights,
        response_mask,
    )

    if student_logits.numel() == 0:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, {
            "distill/teacher_greedy_nll_token_div_mean": 0.0,
            "distill/teacher_greedy_nll_token_div_max": 0.0,
            "distill/teacher_greedy_match_rate": 0.0,
            "distill/token_count": 0.0,
        }

    teacher_greedy_ids = teacher_tensor.argmax(dim=-1)
    student_logp = torch.log_softmax(student_logits, dim=-1)
    token_nll = -torch.gather(student_logp, dim=-1, index=teacher_greedy_ids.unsqueeze(-1)).squeeze(-1)
    student_greedy_ids = student_logits.argmax(dim=-1)
    greedy_match = student_greedy_ids.eq(teacher_greedy_ids)

    flat_mask = mask.reshape(-1)
    token_nll_flat = token_nll.reshape(-1)[flat_mask]
    weight_flat = weights.reshape(-1)[flat_mask]
    greedy_match_flat = greedy_match.reshape(-1)[flat_mask]

    if token_nll_flat.numel() == 0:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        return zero, zero, zero, {
            "distill/teacher_greedy_nll_token_div_mean": 0.0,
            "distill/teacher_greedy_nll_token_div_max": 0.0,
            "distill/teacher_greedy_match_rate": 0.0,
            "distill/token_count": 0.0,
        }

    weighted_sum = torch.sum(token_nll_flat * weight_flat)
    weight_sum = torch.sum(weight_flat)
    loss = weighted_sum / weight_sum if float(weight_sum.item()) > 0.0 else weighted_sum * 0.0

    metrics = {
        "distill/teacher_greedy_nll_token_div_mean": token_nll_flat.mean().item(),
        "distill/teacher_greedy_nll_token_div_max": token_nll_flat.max().item(),
        "distill/teacher_greedy_match_rate": greedy_match_flat.to(dtype=torch.float32).mean().item(),
        "distill/token_count": float(token_nll_flat.numel()),
    }
    return loss, weighted_sum, weight_sum, metrics


@register_distill_loss("fkl")
def compute_fkl(
    student_logits: torch.Tensor,
    teacher_distribution: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
):
    return compute_dense_distill_loss_batched(
        student_logits,
        teacher_distribution,
        token_weights,
        response_mask=response_mask,
        loss_type="fkl",
    )


@register_distill_loss("rkl")
def compute_rkl(
    student_logits: torch.Tensor,
    teacher_distribution: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
):
    return compute_dense_distill_loss_batched(
        student_logits,
        teacher_distribution,
        token_weights,
        response_mask=response_mask,
        loss_type="rkl",
    )


@register_distill_loss("tvd")
def compute_tvd(
    student_logits: torch.Tensor,
    teacher_distribution: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
):
    return compute_dense_distill_loss_batched(
        student_logits,
        teacher_distribution,
        token_weights,
        response_mask=response_mask,
        loss_type="tvd",
    )


@register_distill_loss("teacher_greedy_nll")
def compute_teacher_greedy_nll(
    student_logits: torch.Tensor,
    teacher_distribution: Any,
    token_weights: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
):
    return compute_teacher_greedy_nll_batched(
        student_logits,
        teacher_distribution,
        token_weights,
        response_mask=response_mask,
    )
