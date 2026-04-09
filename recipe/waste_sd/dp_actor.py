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
import math
import os
from typing import Any

import torch

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.distill_debug import DistillDebugRecorder
from recipe.waste_sd.distill_losses import get_distill_loss_fn
from recipe.waste_sd.exact_block_count_loss import compute_exact_block_count_wnll_from_logits
from recipe.waste_sd.waste_weighting import build_strict_weights
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SUPPORTED_WEIGHTING_MODES = {"waste", "uniform_mean", "remaining_budget_forward"}
_SUPPORTED_COEF_SCHEDULES = {"constant", "linear"}
_SUPPORTED_EXACT_AUX_LOSS_TYPES = {"fkl", "tvd"}
_SUPPORTED_TVD_REMBUDGET_BIAS_SCORE_TYPES = {"rembudget", "rembudget_tvd"}
_SUPPORTED_REMBUDGET_TVD_OBJECTIVE_MODES = {"weighted_mean", "theorem_unnormalized"}


def _resolve_scheduled_coef(
    *,
    current_step: int,
    base_coef: float,
    schedule_field_name: str,
    schedule: str = "constant",
    start_coef: float | None = None,
    end_coef: float | None = None,
    start_step: int = 0,
    end_step: int = 0,
) -> float:
    if schedule == "constant":
        return float(base_coef)
    if schedule != "linear":
        raise ValueError(
            f"{schedule_field_name} must be one of "
            f"{sorted(_SUPPORTED_COEF_SCHEDULES)!r}, got {schedule!r}."
        )

    start = float(base_coef if start_coef is None else start_coef)
    end = float(base_coef if end_coef is None else end_coef)
    if current_step <= start_step:
        return start
    if current_step >= end_step or end_step <= start_step:
        return end

    progress = float(current_step - start_step) / float(end_step - start_step)
    return start + progress * (end - start)


def resolve_exact_unweighted_aux_coef(
    *,
    current_step: int,
    base_coef: float,
    schedule: str = "constant",
    start_coef: float | None = None,
    end_coef: float | None = None,
    start_step: int = 0,
    end_step: int = 0,
) -> float:
    return _resolve_scheduled_coef(
        current_step=current_step,
        base_coef=base_coef,
        schedule_field_name="distill.exact_unweighted_aux_coef_schedule",
        schedule=schedule,
        start_coef=start_coef,
        end_coef=end_coef,
        start_step=start_step,
        end_step=end_step,
    )

def resolve_rembudget_unweighted_kl_coef(
    *,
    current_step: int,
    base_coef: float,
    schedule: str = "constant",
    start_coef: float | None = None,
    end_coef: float | None = None,
    start_step: int = 0,
    end_step: int = 0,
) -> float:
    return _resolve_scheduled_coef(
        current_step=current_step,
        base_coef=base_coef,
        schedule_field_name="distill.rembudget_unweighted_kl_coef_schedule",
        schedule=schedule,
        start_coef=start_coef,
        end_coef=end_coef,
        start_step=start_step,
        end_step=end_step,
    )


def resolve_rembudget_tvd_unweighted_fkl_coef(
    *,
    current_step: int,
    base_coef: float,
    schedule: str = "constant",
    start_coef: float | None = None,
    end_coef: float | None = None,
    start_step: int = 0,
    end_step: int = 0,
    ) -> float:
    return _resolve_scheduled_coef(
        current_step=current_step,
        base_coef=base_coef,
        schedule_field_name="distill.rembudget_tvd_unweighted_fkl_coef_schedule",
        schedule=schedule,
        start_coef=start_coef,
        end_coef=end_coef,
        start_step=start_step,
        end_step=end_step,
    )


def rembudget_tvd_bias_active(
    *,
    current_step: int,
    coef: float,
    start_step: int,
) -> bool:
    return coef > 0.0 and current_step >= start_step


def build_mild_rembudget_bias_weights(
    rembudget_weights: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    coef: float,
    power: float,
    eps: float,
    clip: float,
) -> torch.Tensor:
    rembudget_weights = rembudget_weights.to(dtype=torch.float32)
    mask_f = response_mask.to(dtype=torch.float32)
    if rembudget_weights.shape != response_mask.shape:
        raise ValueError(
            "rembudget_weights and response_mask must have the same shape, "
            f"got {tuple(rembudget_weights.shape)} and {tuple(response_mask.shape)}."
        )
    softened = torch.pow(rembudget_weights + float(eps), float(power)) * mask_f
    return build_positive_boost_weights_from_score(
        softened,
        response_mask,
        coef=coef,
        clip=clip,
    )


def build_positive_boost_weights_from_score(
    score: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    coef: float,
    clip: float,
) -> torch.Tensor:
    score = score.to(dtype=torch.float32)
    mask_f = response_mask.to(dtype=torch.float32)
    if score.shape != response_mask.shape:
        raise ValueError(
            "score and response_mask must have the same shape, "
            f"got {tuple(score.shape)} and {tuple(response_mask.shape)}."
        )
    if coef <= 0.0:
        return torch.ones_like(score, dtype=torch.float32) * mask_f
    masked_score = score * mask_f
    denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_score = masked_score.sum(dim=-1, keepdim=True) / denom
    normalized = masked_score / mean_score.clamp_min(1e-8)
    centered = torch.relu(normalized - 1.0)
    if clip > 0.0:
        centered = torch.clamp(centered, max=float(clip))
    return (1.0 + float(coef) * centered) * mask_f


def compute_theorem_rb_tvd_blocks_per_sample(
    *,
    remaining_budget_weight: torch.Tensor,
    token_tvd: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: int,
) -> torch.Tensor:
    valid_f = response_mask.to(dtype=torch.float32)
    seq_len = valid_f.sum(dim=-1)
    return seq_len / float(gamma + 1) + (float(gamma) / float(gamma + 1)) * torch.sum(
        remaining_budget_weight.to(dtype=torch.float32) * token_tvd.to(dtype=torch.float32) * valid_f,
        dim=-1,
    )


def compute_theorem_rb_tvd_blocks_from_logits(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: int,
    kl_floor_coef: float = 0.0,
    dp_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_token_logprobs = torch.gather(teacher_logp, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
    student_token_logprobs = torch.gather(student_logp, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
    forward_weight_result = compute_forward_remaining_budget_weights(
        teacher_logprobs=teacher_token_logprobs,
        student_logprobs=student_token_logprobs,
        gamma=gamma,
        response_mask=response_mask,
        kl_floor_coef=kl_floor_coef,
        dp_dtype=dp_dtype,
    )
    token_tvd = 0.5 * torch.sum(torch.abs(torch.exp(student_logp) - torch.exp(teacher_logp)), dim=-1)
    return compute_theorem_rb_tvd_blocks_per_sample(
        remaining_budget_weight=forward_weight_result.remaining_budget_weight,
        token_tvd=token_tvd,
        response_mask=response_mask,
        gamma=gamma,
    )


class _TokenTVDFromLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        student_prob = torch.exp(student_log_probs)
        teacher_prob = torch.exp(teacher_log_probs)
        diff = student_prob - teacher_prob
        ctx.save_for_backward(
            student_log_probs,
            teacher_log_probs,
            diff.sign().to(torch.int8),
        )
        return 0.5 * diff.abs().sum(dim=-1)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_log_probs, teacher_log_probs, sign = ctx.saved_tensors
        sign_f = sign.to(dtype=student_log_probs.dtype)
        grad_scale = 0.5 * grad_output.unsqueeze(-1)
        grad_student = grad_scale * sign_f * torch.exp(student_log_probs)
        grad_teacher = -grad_scale * sign_f * torch.exp(teacher_log_probs)
        return grad_student, grad_teacher


def _resolve_schedule_bounds(base_coef: float, start_coef: float | None, end_coef: float | None) -> tuple[float, float]:
    start = float(base_coef if start_coef is None else start_coef)
    end = float(base_coef if end_coef is None else end_coef)
    return start, end


def soft_distill_needs_dense_log_probs(
    *,
    loss_type: str,
    rembudget_unweighted_kl_coef: float,
    rembudget_tvd_unweighted_fkl_coef: float,
) -> bool:
    # Same-loss rembudget mixing should not force the TVD path onto the more
    # expensive full-vocab log_softmax graph. The dense log-prob path is only
    # needed when the base loss is FKL or when TVD explicitly mixes in
    # unweighted FKL.
    return loss_type == "fkl" or rembudget_tvd_unweighted_fkl_coef > 0.0


def normalize_exact_unweighted_aux_config(distill_config: Any) -> dict[str, Any]:
    generic_keys = (
        "exact_unweighted_aux_loss_type",
        "exact_unweighted_aux_coef",
        "exact_unweighted_aux_coef_schedule",
        "exact_unweighted_aux_coef_start",
        "exact_unweighted_aux_coef_end",
        "exact_unweighted_aux_coef_start_step",
        "exact_unweighted_aux_coef_end_step",
    )
    legacy_kl_keys = (
        "exact_unweighted_kl_coef",
        "exact_unweighted_kl_coef_schedule",
        "exact_unweighted_kl_coef_start",
        "exact_unweighted_kl_coef_end",
        "exact_unweighted_kl_coef_start_step",
        "exact_unweighted_kl_coef_end_step",
    )
    legacy_tvd_keys = (
        "exact_unweighted_tvd_coef",
        "exact_unweighted_tvd_coef_schedule",
        "exact_unweighted_tvd_coef_start",
        "exact_unweighted_tvd_coef_end",
        "exact_unweighted_tvd_coef_start_step",
        "exact_unweighted_tvd_coef_end_step",
    )

    has_generic = any(key in distill_config for key in generic_keys)
    has_legacy_kl = any(key in distill_config for key in legacy_kl_keys)
    has_legacy_tvd = any(key in distill_config for key in legacy_tvd_keys)

    generic_loss_type = str(distill_config.get("exact_unweighted_aux_loss_type", "fkl")).lower()
    generic_coef = float(distill_config.get("exact_unweighted_aux_coef", 0.0))
    generic_schedule = str(distill_config.get("exact_unweighted_aux_coef_schedule", "constant")).lower()
    generic_start = float(distill_config.get("exact_unweighted_aux_coef_start", generic_coef))
    generic_end = float(distill_config.get("exact_unweighted_aux_coef_end", generic_coef))
    generic_start_step = int(distill_config.get("exact_unweighted_aux_coef_start_step", 0))
    generic_end_step = int(distill_config.get("exact_unweighted_aux_coef_end_step", generic_start_step))
    generic_is_default = (
        generic_loss_type == "fkl"
        and generic_coef == 0.0
        and generic_schedule == "constant"
        and generic_start == 0.0
        and generic_end == 0.0
        and generic_start_step == 0
        and generic_end_step == 0
    )
    has_generic_override = has_generic and not generic_is_default

    if has_generic_override and (has_legacy_kl or has_legacy_tvd):
        raise ValueError(
            "Do not mix distill.exact_unweighted_aux_* with legacy "
            "distill.exact_unweighted_kl_* or distill.exact_unweighted_tvd_* fields."
        )
    if has_legacy_kl and has_legacy_tvd:
        raise ValueError(
            "distill.exact_unweighted_kl_* and distill.exact_unweighted_tvd_* are mutually exclusive. "
            "Use exactly one legacy exact mixing path, or switch to distill.exact_unweighted_aux_*."
        )

    if has_legacy_tvd:
        aux_loss_type = "tvd"
        coef = float(distill_config.get("exact_unweighted_tvd_coef", 0.0))
        schedule = str(distill_config.get("exact_unweighted_tvd_coef_schedule", "constant")).lower()
        start_coef = float(distill_config.get("exact_unweighted_tvd_coef_start", coef))
        end_coef = float(distill_config.get("exact_unweighted_tvd_coef_end", coef))
        start_step = int(distill_config.get("exact_unweighted_tvd_coef_start_step", 0))
        end_step = int(distill_config.get("exact_unweighted_tvd_coef_end_step", start_step))
    elif has_legacy_kl:
        aux_loss_type = "fkl"
        coef = float(distill_config.get("exact_unweighted_kl_coef", 0.0))
        schedule = str(distill_config.get("exact_unweighted_kl_coef_schedule", "constant")).lower()
        start_coef = float(distill_config.get("exact_unweighted_kl_coef_start", coef))
        end_coef = float(distill_config.get("exact_unweighted_kl_coef_end", coef))
        start_step = int(distill_config.get("exact_unweighted_kl_coef_start_step", 0))
        end_step = int(distill_config.get("exact_unweighted_kl_coef_end_step", start_step))
    else:
        aux_loss_type = generic_loss_type
        coef = generic_coef
        schedule = generic_schedule
        start_coef = generic_start
        end_coef = generic_end
        start_step = generic_start_step
        end_step = generic_end_step

    return {
        "loss_type": aux_loss_type,
        "coef": coef,
        "schedule": schedule,
        "start": start_coef,
        "end": end_coef,
        "start_step": start_step,
        "end_step": end_step,
    }


def _validate_scheduled_coef_config(
    *,
    field_name: str,
    applies: bool,
    base_coef: float,
    schedule: str,
    start_coef: float | None,
    end_coef: float | None,
    start_step: int,
    end_step: int,
) -> tuple[float, float]:
    if schedule not in _SUPPORTED_COEF_SCHEDULES:
        raise ValueError(
            f"{field_name}_schedule must be one of "
            f"{sorted(_SUPPORTED_COEF_SCHEDULES)!r}, got {schedule!r}."
        )
    if schedule != "constant" and not applies:
        raise ValueError(
            f"{field_name}_schedule only applies to the matching distillation mode. "
            f"Got schedule={schedule!r}."
        )

    start, end = _resolve_schedule_bounds(base_coef, start_coef, end_coef)
    if not (0.0 <= start <= 1.0):
        raise ValueError(f"{field_name}_start must be in [0, 1], got {start}.")
    if not (0.0 <= end <= 1.0):
        raise ValueError(f"{field_name}_end must be in [0, 1], got {end}.")
    if start_step < 0:
        raise ValueError(f"{field_name}_start_step must be >= 0, got {start_step}.")
    if end_step < start_step:
        raise ValueError(
            f"{field_name}_end_step must be >= {field_name}_start_step. "
            f"Got start_step={start_step}, end_step={end_step}."
        )
    return start, end


def validate_distill_objective_config(
    loss_type: str,
    weighting_mode: str,
    kl_floor_coef: float,
    rembudget_tvd_objective_mode: str = "weighted_mean",
    rembudget_tvd_backward_length_normalize: bool = False,
    rembudget_unweighted_kl_coef: float = 0.0,
    rembudget_unweighted_kl_coef_schedule: str = "constant",
    rembudget_unweighted_kl_coef_start: float | None = None,
    rembudget_unweighted_kl_coef_end: float | None = None,
    rembudget_unweighted_kl_coef_start_step: int = 0,
    rembudget_unweighted_kl_coef_end_step: int = 0,
    rembudget_tvd_unweighted_fkl_coef: float = 0.0,
    rembudget_tvd_unweighted_fkl_coef_schedule: str = "constant",
    rembudget_tvd_unweighted_fkl_coef_start: float | None = None,
    rembudget_tvd_unweighted_fkl_coef_end: float | None = None,
    rembudget_tvd_unweighted_fkl_coef_start_step: int = 0,
    rembudget_tvd_unweighted_fkl_coef_end_step: int = 0,
    exact_unweighted_aux_loss_type: str = "fkl",
    exact_unweighted_aux_coef: float = 0.0,
    exact_unweighted_aux_coef_schedule: str = "constant",
    exact_unweighted_aux_coef_start: float | None = None,
    exact_unweighted_aux_coef_end: float | None = None,
    exact_unweighted_aux_coef_start_step: int = 0,
    exact_unweighted_aux_coef_end_step: int = 0,
    tvd_rembudget_bias_score_type: str = "rembudget",
    tvd_rembudget_bias_coef: float = 0.0,
    tvd_rembudget_bias_power: float = 1.0,
    tvd_rembudget_bias_eps: float = 1e-3,
    tvd_rembudget_bias_clip: float = 0.5,
    tvd_rembudget_bias_start_step: int = 0,
) -> None:
    rembudget_tvd_objective_mode = rembudget_tvd_objective_mode.lower()
    if weighting_mode not in _SUPPORTED_WEIGHTING_MODES:
        raise ValueError(
            "distill.weighting_mode must be one of "
            f"{sorted(_SUPPORTED_WEIGHTING_MODES)!r}, got {weighting_mode!r}."
        )
    if rembudget_tvd_objective_mode not in _SUPPORTED_REMBUDGET_TVD_OBJECTIVE_MODES:
        raise ValueError(
            "distill.rembudget_tvd_objective_mode must be one of "
            f"{sorted(_SUPPORTED_REMBUDGET_TVD_OBJECTIVE_MODES)!r}, "
            f"got {rembudget_tvd_objective_mode!r}."
        )
    if loss_type == "teacher_greedy_nll" and weighting_mode != "uniform_mean":
        raise ValueError(
            "distill.loss_type='teacher_greedy_nll' currently requires "
            "distill.weighting_mode='uniform_mean' so the objective remains plain "
            "teacher-greedy token MLE on sampled prefixes. "
            f"Got weighting_mode={weighting_mode!r}."
        )
    if loss_type == "exact_block_count_wnll" and weighting_mode != "uniform_mean":
        raise ValueError(
            "distill.loss_type='exact_block_count_wnll' currently requires "
            "distill.weighting_mode='uniform_mean'. The exact theorem-backed objective "
            "is a detached weighted sampled-token NLL with alpha-weighted token-mean aggregation, "
            "so waste/remaining-budget weighting modes do not apply on top. "
            f"Got weighting_mode={weighting_mode!r}."
        )
    if weighting_mode == "remaining_budget_forward" and loss_type not in {"fkl", "tvd"}:
        raise ValueError(
            "distill.weighting_mode='remaining_budget_forward' currently requires "
            "distill.loss_type in {'fkl', 'tvd'}. "
            f"Got loss_type={loss_type!r}."
        )
    if not (0.0 <= kl_floor_coef <= 1.0):
        raise ValueError(f"distill.kl_floor_coef must be in [0, 1], got {kl_floor_coef}.")
    if not (0.0 <= rembudget_unweighted_kl_coef <= 1.0):
        raise ValueError(
            "distill.rembudget_unweighted_kl_coef must be in [0, 1], "
            f"got {rembudget_unweighted_kl_coef}."
        )
    if rembudget_unweighted_kl_coef > 0.0 and not (
        weighting_mode == "remaining_budget_forward" and loss_type in {"fkl", "tvd"}
    ):
        raise ValueError(
            "distill.rembudget_unweighted_kl_coef only applies to "
            "distill.weighting_mode='remaining_budget_forward' with "
            "distill.loss_type in {'fkl', 'tvd'}. "
            f"Got weighting_mode={weighting_mode!r}, loss_type={loss_type!r}."
        )
    rembudget_same_loss_applies = weighting_mode == "remaining_budget_forward" and loss_type in {"fkl", "tvd"}
    rembudget_same_loss_start, rembudget_same_loss_end = _validate_scheduled_coef_config(
        field_name="distill.rembudget_unweighted_kl_coef",
        applies=rembudget_same_loss_applies,
        base_coef=rembudget_unweighted_kl_coef,
        schedule=rembudget_unweighted_kl_coef_schedule,
        start_coef=rembudget_unweighted_kl_coef_start,
        end_coef=rembudget_unweighted_kl_coef_end,
        start_step=rembudget_unweighted_kl_coef_start_step,
        end_step=rembudget_unweighted_kl_coef_end_step,
    )
    if not (0.0 <= rembudget_tvd_unweighted_fkl_coef <= 1.0):
        raise ValueError(
            "distill.rembudget_tvd_unweighted_fkl_coef must be in [0, 1], "
            f"got {rembudget_tvd_unweighted_fkl_coef}."
        )
    rembudget_tvd_fkl_applies = weighting_mode == "remaining_budget_forward" and loss_type == "tvd"
    if rembudget_tvd_unweighted_fkl_coef > 0.0 and not rembudget_tvd_fkl_applies:
        raise ValueError(
            "distill.rembudget_tvd_unweighted_fkl_coef only applies to "
            "distill.weighting_mode='remaining_budget_forward' with "
            "distill.loss_type='tvd'. "
            f"Got weighting_mode={weighting_mode!r}, loss_type={loss_type!r}."
        )
    rembudget_tvd_fkl_start, rembudget_tvd_fkl_end = _validate_scheduled_coef_config(
        field_name="distill.rembudget_tvd_unweighted_fkl_coef",
        applies=rembudget_tvd_fkl_applies,
        base_coef=rembudget_tvd_unweighted_fkl_coef,
        schedule=rembudget_tvd_unweighted_fkl_coef_schedule,
        start_coef=rembudget_tvd_unweighted_fkl_coef_start,
        end_coef=rembudget_tvd_unweighted_fkl_coef_end,
        start_step=rembudget_tvd_unweighted_fkl_coef_start_step,
        end_step=rembudget_tvd_unweighted_fkl_coef_end_step,
    )
    rembudget_same_loss_active = max(rembudget_unweighted_kl_coef, rembudget_same_loss_start, rembudget_same_loss_end) > 0.0
    rembudget_tvd_fkl_active = max(
        rembudget_tvd_unweighted_fkl_coef,
        rembudget_tvd_fkl_start,
        rembudget_tvd_fkl_end,
    ) > 0.0
    if rembudget_same_loss_active and rembudget_tvd_fkl_active:
        raise ValueError(
            "distill.rembudget_unweighted_kl_coef and "
            "distill.rembudget_tvd_unweighted_fkl_coef are mutually exclusive. "
            "Choose one rembudget mixing path per run."
        )
    if rembudget_tvd_objective_mode == "theorem_unnormalized":
        if not (weighting_mode == "remaining_budget_forward" and loss_type == "tvd"):
            raise ValueError(
                "distill.rembudget_tvd_objective_mode='theorem_unnormalized' only applies to "
                "distill.weighting_mode='remaining_budget_forward' with distill.loss_type='tvd'. "
                f"Got weighting_mode={weighting_mode!r}, loss_type={loss_type!r}."
            )
        if kl_floor_coef != 0.0:
            raise ValueError(
                "distill.rembudget_tvd_objective_mode='theorem_unnormalized' requires "
                "distill.kl_floor_coef=0.0 so the raw theorem weight w_n^rem is used. "
                f"Got kl_floor_coef={kl_floor_coef}."
            )
        if rembudget_same_loss_active or rembudget_tvd_fkl_active:
            raise ValueError(
                "distill.rembudget_tvd_objective_mode='theorem_unnormalized' does not support "
                "rembudget mixing coefficients. Set distill.rembudget_unweighted_kl_coef=0 and "
                "distill.rembudget_tvd_unweighted_fkl_coef=0."
            )
    elif rembudget_tvd_backward_length_normalize:
        raise ValueError(
            "distill.rembudget_tvd_backward_length_normalize only applies to "
            "distill.rembudget_tvd_objective_mode='theorem_unnormalized'."
        )
    exact_unweighted_aux_loss_type = exact_unweighted_aux_loss_type.lower()
    if exact_unweighted_aux_loss_type not in _SUPPORTED_EXACT_AUX_LOSS_TYPES:
        raise ValueError(
            "distill.exact_unweighted_aux_loss_type must be one of "
            f"{sorted(_SUPPORTED_EXACT_AUX_LOSS_TYPES)!r}, got {exact_unweighted_aux_loss_type!r}."
        )
    if not (0.0 <= exact_unweighted_aux_coef <= 1.0):
        raise ValueError(
            "distill.exact_unweighted_aux_coef must be in [0, 1], "
            f"got {exact_unweighted_aux_coef}."
        )
    if exact_unweighted_aux_coef > 0.0 and loss_type != "exact_block_count_wnll":
        raise ValueError(
            "distill.exact_unweighted_aux_coef only applies to "
            "distill.loss_type='exact_block_count_wnll'. "
            f"Got loss_type={loss_type!r}."
        )
    _validate_scheduled_coef_config(
        field_name="distill.exact_unweighted_aux_coef",
        applies=loss_type == "exact_block_count_wnll",
        base_coef=exact_unweighted_aux_coef,
        schedule=exact_unweighted_aux_coef_schedule,
        start_coef=exact_unweighted_aux_coef_start,
        end_coef=exact_unweighted_aux_coef_end,
        start_step=exact_unweighted_aux_coef_start_step,
        end_step=exact_unweighted_aux_coef_end_step,
    )
    tvd_rembudget_bias_score_type = tvd_rembudget_bias_score_type.lower()
    if tvd_rembudget_bias_score_type not in _SUPPORTED_TVD_REMBUDGET_BIAS_SCORE_TYPES:
        raise ValueError(
            "distill.tvd_rembudget_bias_score_type must be one of "
            f"{sorted(_SUPPORTED_TVD_REMBUDGET_BIAS_SCORE_TYPES)!r}, got {tvd_rembudget_bias_score_type!r}."
        )
    if tvd_rembudget_bias_coef < 0.0:
        raise ValueError(
            "distill.tvd_rembudget_bias_coef must be >= 0, "
            f"got {tvd_rembudget_bias_coef}."
        )
    if tvd_rembudget_bias_coef > 0.0 and not (loss_type == "tvd" and weighting_mode == "uniform_mean"):
        raise ValueError(
            "distill.tvd_rembudget_bias_coef only applies to "
            "distill.loss_type='tvd' with distill.weighting_mode='uniform_mean'. "
            f"Got loss_type={loss_type!r}, weighting_mode={weighting_mode!r}."
        )
    if tvd_rembudget_bias_power <= 0.0:
        raise ValueError(
            "distill.tvd_rembudget_bias_power must be > 0, "
            f"got {tvd_rembudget_bias_power}."
        )
    if tvd_rembudget_bias_eps < 0.0:
        raise ValueError(
            "distill.tvd_rembudget_bias_eps must be >= 0, "
            f"got {tvd_rembudget_bias_eps}."
        )
    if tvd_rembudget_bias_clip <= 0.0:
        raise ValueError(
            "distill.tvd_rembudget_bias_clip must be > 0, "
            f"got {tvd_rembudget_bias_clip}."
        )
    if tvd_rembudget_bias_start_step < 0:
        raise ValueError(
            "distill.tvd_rembudget_bias_start_step must be >= 0, "
            f"got {tvd_rembudget_bias_start_step}."
        )


class DataParallelWasteSDDistillActor(DataParallelPPOActor):
    """FSDP actor used by strict waste-aware SD distillation."""

    def __init__(
        self,
        config,
        actor_module: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        distill_config: Any,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.distill_config = distill_config
        self.loss_type = str(distill_config.get("loss_type", "fkl")).lower()
        self.q_source = str(distill_config.get("q_source", "local_ref")).lower()
        self.weighting_mode = str(distill_config.get("weighting_mode", "waste")).lower()
        self.gamma = int(distill_config.get("gamma", 1))
        self.kl_floor_coef = float(distill_config.get("kl_floor_coef", 0.0))
        self.rembudget_tvd_objective_mode = str(
            distill_config.get("rembudget_tvd_objective_mode", "weighted_mean")
        ).lower()
        self.rembudget_tvd_backward_length_normalize = bool(
            distill_config.get("rembudget_tvd_backward_length_normalize", False)
        )
        self.rembudget_unweighted_kl_coef = float(distill_config.get("rembudget_unweighted_kl_coef", 0.0))
        self.rembudget_unweighted_kl_coef_schedule = str(
            distill_config.get("rembudget_unweighted_kl_coef_schedule", "constant")
        ).lower()
        self.rembudget_unweighted_kl_coef_start = float(
            distill_config.get("rembudget_unweighted_kl_coef_start", self.rembudget_unweighted_kl_coef)
        )
        self.rembudget_unweighted_kl_coef_end = float(
            distill_config.get("rembudget_unweighted_kl_coef_end", self.rembudget_unweighted_kl_coef)
        )
        self.rembudget_unweighted_kl_coef_start_step = int(
            distill_config.get("rembudget_unweighted_kl_coef_start_step", 0)
        )
        self.rembudget_unweighted_kl_coef_end_step = int(
            distill_config.get(
                "rembudget_unweighted_kl_coef_end_step",
                self.rembudget_unweighted_kl_coef_start_step,
            )
        )
        self.rembudget_tvd_unweighted_fkl_coef = float(
            distill_config.get("rembudget_tvd_unweighted_fkl_coef", 0.0)
        )
        self.rembudget_tvd_unweighted_fkl_coef_schedule = str(
            distill_config.get("rembudget_tvd_unweighted_fkl_coef_schedule", "constant")
        ).lower()
        self.rembudget_tvd_unweighted_fkl_coef_start = float(
            distill_config.get(
                "rembudget_tvd_unweighted_fkl_coef_start",
                self.rembudget_tvd_unweighted_fkl_coef,
            )
        )
        self.rembudget_tvd_unweighted_fkl_coef_end = float(
            distill_config.get(
                "rembudget_tvd_unweighted_fkl_coef_end",
                self.rembudget_tvd_unweighted_fkl_coef,
            )
        )
        self.rembudget_tvd_unweighted_fkl_coef_start_step = int(
            distill_config.get("rembudget_tvd_unweighted_fkl_coef_start_step", 0)
        )
        self.rembudget_tvd_unweighted_fkl_coef_end_step = int(
            distill_config.get(
                "rembudget_tvd_unweighted_fkl_coef_end_step",
                self.rembudget_tvd_unweighted_fkl_coef_start_step,
            )
        )
        exact_aux_config = normalize_exact_unweighted_aux_config(distill_config)
        self.exact_unweighted_aux_loss_type = str(exact_aux_config["loss_type"]).lower()
        self.exact_unweighted_aux_coef = float(exact_aux_config["coef"])
        self.exact_unweighted_aux_coef_schedule = str(exact_aux_config["schedule"]).lower()
        self.exact_unweighted_aux_coef_start = float(exact_aux_config["start"])
        self.exact_unweighted_aux_coef_end = float(exact_aux_config["end"])
        self.exact_unweighted_aux_coef_start_step = int(exact_aux_config["start_step"])
        self.exact_unweighted_aux_coef_end_step = int(exact_aux_config["end_step"])
        self.tvd_rembudget_bias_score_type = str(
            distill_config.get("tvd_rembudget_bias_score_type", "rembudget")
        ).lower()
        self.tvd_rembudget_bias_coef = float(distill_config.get("tvd_rembudget_bias_coef", 0.0))
        self.tvd_rembudget_bias_power = float(distill_config.get("tvd_rembudget_bias_power", 1.0))
        self.tvd_rembudget_bias_eps = float(distill_config.get("tvd_rembudget_bias_eps", 1e-3))
        self.tvd_rembudget_bias_clip = float(distill_config.get("tvd_rembudget_bias_clip", 0.5))
        self.tvd_rembudget_bias_start_step = int(distill_config.get("tvd_rembudget_bias_start_step", 0))
        self.log_current_batch_theorem_rb_tvd = bool(
            distill_config.get("log_current_batch_theorem_rb_tvd", False)
        )
        self.log_post_update_batch_theorem_rb_tvd = bool(
            distill_config.get("log_post_update_batch_theorem_rb_tvd", False)
        )
        self.strict = bool(distill_config.get("strict", True))
        self.forward_path = str(distill_config.get("forward_path", "auto")).lower()
        legacy_full_block_flag = distill_config.get("full_block_participate", None)
        if legacy_full_block_flag is not None:
            logger.warning(
                "distill.full_block_participate is deprecated and ignored. "
                "Full blocks are always handled by the strict weight formula."
            )
        if self.q_source != "local_ref":
            raise ValueError(
                "waste_sd actor now requires distill.q_source=local_ref. "
                f"Got {self.q_source}."
            )
        validate_distill_objective_config(
            loss_type=self.loss_type,
            weighting_mode=self.weighting_mode,
            kl_floor_coef=self.kl_floor_coef,
            rembudget_tvd_objective_mode=self.rembudget_tvd_objective_mode,
            rembudget_tvd_backward_length_normalize=self.rembudget_tvd_backward_length_normalize,
            rembudget_unweighted_kl_coef=self.rembudget_unweighted_kl_coef,
            rembudget_unweighted_kl_coef_schedule=self.rembudget_unweighted_kl_coef_schedule,
            rembudget_unweighted_kl_coef_start=self.rembudget_unweighted_kl_coef_start,
            rembudget_unweighted_kl_coef_end=self.rembudget_unweighted_kl_coef_end,
            rembudget_unweighted_kl_coef_start_step=self.rembudget_unweighted_kl_coef_start_step,
            rembudget_unweighted_kl_coef_end_step=self.rembudget_unweighted_kl_coef_end_step,
            rembudget_tvd_unweighted_fkl_coef=self.rembudget_tvd_unweighted_fkl_coef,
            rembudget_tvd_unweighted_fkl_coef_schedule=self.rembudget_tvd_unweighted_fkl_coef_schedule,
            rembudget_tvd_unweighted_fkl_coef_start=self.rembudget_tvd_unweighted_fkl_coef_start,
            rembudget_tvd_unweighted_fkl_coef_end=self.rembudget_tvd_unweighted_fkl_coef_end,
            rembudget_tvd_unweighted_fkl_coef_start_step=self.rembudget_tvd_unweighted_fkl_coef_start_step,
            rembudget_tvd_unweighted_fkl_coef_end_step=self.rembudget_tvd_unweighted_fkl_coef_end_step,
            exact_unweighted_aux_loss_type=self.exact_unweighted_aux_loss_type,
            exact_unweighted_aux_coef=self.exact_unweighted_aux_coef,
            exact_unweighted_aux_coef_schedule=self.exact_unweighted_aux_coef_schedule,
            exact_unweighted_aux_coef_start=self.exact_unweighted_aux_coef_start,
            exact_unweighted_aux_coef_end=self.exact_unweighted_aux_coef_end,
            exact_unweighted_aux_coef_start_step=self.exact_unweighted_aux_coef_start_step,
            exact_unweighted_aux_coef_end_step=self.exact_unweighted_aux_coef_end_step,
            tvd_rembudget_bias_score_type=self.tvd_rembudget_bias_score_type,
            tvd_rembudget_bias_coef=self.tvd_rembudget_bias_coef,
            tvd_rembudget_bias_power=self.tvd_rembudget_bias_power,
            tvd_rembudget_bias_eps=self.tvd_rembudget_bias_eps,
            tvd_rembudget_bias_clip=self.tvd_rembudget_bias_clip,
            tvd_rembudget_bias_start_step=self.tvd_rembudget_bias_start_step,
        )
        self.teacher_module: torch.nn.Module | None = None
        self.loss_fn = None if self.loss_type == "exact_block_count_wnll" else get_distill_loss_fn(self.loss_type)
        self.unweighted_fkl_loss_fn = get_distill_loss_fn("fkl")
        self.unweighted_tvd_loss_fn = get_distill_loss_fn("tvd")
        self.debug_recorder = DistillDebugRecorder(distill_config.get("debug", {}))
        self._local_update_step = 0
        self._warned_forward_path_fallback = False
        self._sync_debug = os.getenv("WASTE_SD_DISTILL_SYNC_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _uses_theorem_unnormalized_rembudget_tvd(self) -> bool:
        return (
            self.weighting_mode == "remaining_budget_forward"
            and self.loss_type == "tvd"
            and self.rembudget_tvd_objective_mode == "theorem_unnormalized"
        )

    def _compute_post_update_batch_theorem_rb_tvd_metrics(
        self,
        *,
        micro_batches: list[DataProto],
        temperature: float,
        pad_token_id: int,
    ) -> dict[str, float]:
        was_training = self.actor_module.training
        self.actor_module.eval()
        try:
            block_sum = 0.0
            sample_count = 0.0
            with torch.no_grad():
                for micro_batch in micro_batches:
                    eval_batch = micro_batch.to(get_device_id())
                    model_inputs = {**eval_batch.batch, **eval_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"].bool()
                    local_sample_count = int(response_mask.any(dim=-1).sum().item())
                    if local_sample_count <= 0:
                        continue
                    student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
                    teacher_logits = self._forward_response_logits(
                        model_inputs,
                        temperature=temperature,
                        module=self.teacher_module,
                    )
                    theorem_blocks = compute_theorem_rb_tvd_blocks_from_logits(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        responses=model_inputs["responses"],
                        response_mask=response_mask,
                        gamma=self.gamma,
                        kl_floor_coef=self.kl_floor_coef,
                        dp_dtype=torch.float64,
                    )
                    block_sum += float(theorem_blocks.sum().item())
                    sample_count += float(theorem_blocks.numel())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                reduce_t = torch.tensor(
                    [block_sum, sample_count],
                    dtype=torch.float32,
                    device=torch.device(self.device_name, get_device_id()),
                )
                torch.distributed.all_reduce(reduce_t, op=torch.distributed.ReduceOp.SUM)
                block_sum = float(reduce_t[0].item())
                sample_count = float(reduce_t[1].item())
            mean_blocks = block_sum / sample_count if sample_count > 0.0 else 0.0
            return {
                "distill/post_update_batch_theorem_rb_tvd_mean_blocks": mean_blocks,
                "distill/post_update_batch_theorem_rb_tvd_sum_blocks": block_sum,
            }
        finally:
            if was_training:
                self.actor_module.train()

    def _forward_response_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        *,
        module: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if self.forward_path == "remove_padding":
            # Keep forward_path flag for future experiments but use padded forward in v1 for robustness.
            if not self._warned_forward_path_fallback:
                logger.warning("forward_path=remove_padding is not fully supported in waste_sd v1, fallback to padded.")
                self._warned_forward_path_fallback = True
        elif self.forward_path not in {"auto", "padded_only", "remove_padding"}:
            raise ValueError(f"Unsupported distill.forward_path={self.forward_path}")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        response_length = micro_batch["responses"].size(-1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            outputs = (module or self.actor_module)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = outputs.logits[:, -response_length - 1 : -1, :]
            logits = logits.float()
            logits.div_(temperature)
        return logits

    def _get_non_tensor_value(self, data: DataProto, key: str, index: int):
        if key not in data.non_tensor_batch:
            return None
        values = data.non_tensor_batch[key]
        if index >= len(values):
            return None
        return values[index]

    @staticmethod
    def _reduce_list_metric(metric_key: str, values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        if "max" in metric_key:
            return float(max(values))
        if "min" in metric_key:
            return float(min(values))
        return float(sum(float(x) for x in values) / len(values))

    def _reduce_sample_metrics(self, sample_metrics: dict[str, Any]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in sample_metrics.items():
            if isinstance(value, list):
                reduced[key] = self._reduce_list_metric(key, value)
            else:
                reduced[key] = float(value)
        return reduced

    def _init_micro_metrics(self, *, local_has_contrib: bool, contributing_samples: int) -> dict[str, float]:
        micro_metrics: dict[str, float] = {
            "distill/zero_contrib_micro_batches": 1.0 if not local_has_contrib else 0.0,
            "distill/rank_zero_contrib_while_global_nonzero": 0.0,
            "distill/weighted_divergence_mean": 0.0,
            f"distill/weighted_{self.loss_type}_mean": 0.0,
            "distill/contributing_samples": float(contributing_samples),
            "distill/micro_weight_sum": 0.0,
            "distill/weight_nonzero_ratio": 0.0,
            "distill/weight_mean": 0.0,
            "distill/weight_max": 0.0,
            "distill/weight_sum": 0.0,
            "distill/strict_alignment_ok": 1.0,
            f"distill/{self.loss_type}_token_div_mean": 0.0,
            f"distill/{self.loss_type}_token_div_max": 0.0,
            "distill/student_nll_all_response_tokens_mean": 0.0,
            "distill/token_count": 0.0,
        }
        if self.log_current_batch_theorem_rb_tvd and self.loss_type == "tvd":
            micro_metrics["distill/current_batch_theorem_rb_tvd_mean_blocks"] = 0.0
            micro_metrics["distill/current_batch_theorem_rb_tvd_sum_blocks"] = 0.0
        if self._uses_theorem_unnormalized_rembudget_tvd():
            micro_metrics["distill/theorem_backward_avg_tokens"] = 0.0
        if self.loss_type == "fkl":
            micro_metrics["distill/weighted_kl_mean"] = 0.0
        if self.weighting_mode != "uniform_mean" and not self._uses_theorem_unnormalized_rembudget_tvd():
            micro_metrics["actor/distill_loss"] = 0.0
        return micro_metrics

    @staticmethod
    def _gather_response_token_logprobs(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return torch.gather(log_probs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _gather_response_token_logprobs_via_logsumexp(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        selected = torch.gather(logits_f, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits_f, dim=-1)

    @staticmethod
    def _gather_response_token_logprobs_from_log_probs(log_probs: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        return torch.gather(log_probs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _token_tvd_from_log_probs(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
        return _TokenTVDFromLogProbs.apply(student_log_probs, teacher_log_probs)

    @staticmethod
    def _token_nll_stats_from_token_logprobs(
        token_logp: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[float, float]:
        token_nll = -token_logp[response_mask]
        if token_nll.numel() == 0:
            return 0.0, 0.0
        return token_nll.mean().item(), token_nll.max().item()

    @staticmethod
    def _masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        return torch.sum(values * mask_f, dim=-1) / denom

    @staticmethod
    def _weighted_token_mean(weighted_sum: torch.Tensor, weight_sum: torch.Tensor) -> torch.Tensor:
        if float(weight_sum.item()) > 0.0:
            return weighted_sum / weight_sum
        return weighted_sum * 0.0

    def _student_token_nll_stats(
        self,
        student_logits: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[float, float]:
        token_logp = self._gather_response_token_logprobs(student_logits, responses)
        return self._token_nll_stats_from_token_logprobs(token_logp, response_mask)

    def update_policy_distill(self, data: DataProto) -> dict[str, Any]:
        current_step = int(data.meta_info.get("global_steps", self._local_update_step))
        self._local_update_step += 1
        exact_unweighted_aux_coef = resolve_exact_unweighted_aux_coef(
            current_step=current_step,
            base_coef=self.exact_unweighted_aux_coef,
            schedule=self.exact_unweighted_aux_coef_schedule,
            start_coef=self.exact_unweighted_aux_coef_start,
            end_coef=self.exact_unweighted_aux_coef_end,
            start_step=self.exact_unweighted_aux_coef_start_step,
            end_step=self.exact_unweighted_aux_coef_end_step,
        )
        exact_aux_loss_type = self.exact_unweighted_aux_loss_type
        exact_aux_loss_fn = self.unweighted_fkl_loss_fn if exact_aux_loss_type == "fkl" else self.unweighted_tvd_loss_fn
        exact_aux_metric_name = f"distill/unweighted_{exact_aux_loss_type}_mean"
        exact_aux_mixed_metric_name = f"distill/mixed_exact_unweighted_{exact_aux_loss_type}_mean"
        rembudget_unweighted_kl_coef = resolve_rembudget_unweighted_kl_coef(
            current_step=current_step,
            base_coef=self.rembudget_unweighted_kl_coef,
            schedule=self.rembudget_unweighted_kl_coef_schedule,
            start_coef=self.rembudget_unweighted_kl_coef_start,
            end_coef=self.rembudget_unweighted_kl_coef_end,
            start_step=self.rembudget_unweighted_kl_coef_start_step,
            end_step=self.rembudget_unweighted_kl_coef_end_step,
        )
        rembudget_tvd_unweighted_fkl_coef = resolve_rembudget_tvd_unweighted_fkl_coef(
            current_step=current_step,
            base_coef=self.rembudget_tvd_unweighted_fkl_coef,
            schedule=self.rembudget_tvd_unweighted_fkl_coef_schedule,
            start_coef=self.rembudget_tvd_unweighted_fkl_coef_start,
            end_coef=self.rembudget_tvd_unweighted_fkl_coef_end,
            start_step=self.rembudget_tvd_unweighted_fkl_coef_start_step,
            end_step=self.rembudget_tvd_unweighted_fkl_coef_end_step,
        )
        tvd_rembudget_bias_enabled = rembudget_tvd_bias_active(
            current_step=current_step,
            coef=self.tvd_rembudget_bias_coef,
            start_step=self.tvd_rembudget_bias_start_step,
        )
        log_current_batch_theorem_rb_tvd = self.log_current_batch_theorem_rb_tvd and self.loss_type == "tvd"
        log_post_update_batch_theorem_rb_tvd = self.log_post_update_batch_theorem_rb_tvd and self.loss_type == "tvd"
        self.actor_module.train()
        if self.teacher_module is None:
            raise RuntimeError(
                "waste_sd requires a local teacher model in actor worker. "
                "Ensure ref model is initialized and attached as actor.teacher_module."
            )
        self.teacher_module.eval()

        if self.config.get("use_dynamic_bsz", False):
            raise NotImplementedError("waste_sd v1 does not support use_dynamic_bsz=True for distill update.")

        temperature = float(data.meta_info["temperature"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"Invalid distillation temperature={temperature}. "
                "Expected a positive finite value."
            )
        pad_token_id = int(data.meta_info.get("pad_token_id", 0))

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = [
            key
            for key in ["spec_accept_lens", "uid", "multi_modal_inputs"]
            if key in data.non_tensor_batch
        ]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batch_size = int(self.config.ppo_mini_batch_size)
        micro_batch_size = int(self.config.ppo_micro_batch_size_per_gpu)
        mini_batches = data.split(mini_batch_size)

        metrics: dict[str, Any] = {"actor/distill_loss": []}
        did_update_step = False
        uniform_step_weighted_sum = 0.0
        uniform_step_token_count = 0.0
        exact_step_loss_sum = 0.0
        exact_step_weight_sum = 0.0
        exact_aux_step_weighted_sum = 0.0
        exact_aux_step_weight_sum = 0.0
        theorem_step_block_sum = 0.0
        theorem_step_sample_count = 0.0
        theorem_step_token_count = 0.0
        theorem_unnormalized_rembudget_tvd = self._uses_theorem_unnormalized_rembudget_tvd()
        theorem_world_size = (
            torch.distributed.get_world_size()
            if theorem_unnormalized_rembudget_tvd
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 1
        )

        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(micro_batch_size)
            grad_accum_steps = max(len(micro_batches), 1)
            uniform_total_weight_sum = 0.0
            exact_total_weight_sum = 0.0
            theorem_total_sample_count = 0.0
            theorem_total_token_count = 0.0
            trainable_params = [param for param in self.actor_module.parameters() if param.requires_grad]
            exact_grad_buffers: list[torch.Tensor | None] = [None] * len(trainable_params)
            local_has_contrib_flags: list[bool] = []
            micro_records: list[dict[str, Any]] = []
            sync_device = None
            if self.weighting_mode == "uniform_mean" and self.loss_type != "exact_block_count_wnll":
                # Exact token-level mean across the whole mini-batch:
                #   L = (sum_i weighted_sum_i) / (sum_i weight_sum_i)
                # For uniform_mean, the denominator stays the valid-token count even when
                # token weights are used as a detached multiplicative TVD bias.
                for micro_batch_ref in micro_batches:
                    uniform_total_weight_sum += float(micro_batch_ref.batch["response_mask"].sum().item())
            if theorem_unnormalized_rembudget_tvd:
                # The strict theorem objective is an expectation over samples:
                #   (1 / B) * sum_i [ T_i/(gamma+1) + gamma/(gamma+1) * sum_n w_{i,n} TVD_{i,n} ]
                # so we normalize by valid sample count, not by token count.
                for micro_batch_ref in micro_batches:
                    theorem_total_sample_count += float(
                        micro_batch_ref.batch["response_mask"].any(dim=-1).sum().item()
                    )
                    theorem_total_token_count += float(micro_batch_ref.batch["response_mask"].sum().item())
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    theorem_total_count_t = torch.tensor(
                        [theorem_total_sample_count, theorem_total_token_count],
                        dtype=torch.float32,
                        device=torch.device(self.device_name, get_device_id()),
                    )
                    torch.distributed.all_reduce(theorem_total_count_t, op=torch.distributed.ReduceOp.SUM)
                    theorem_total_sample_count = float(theorem_total_count_t[0].item())
                    theorem_total_token_count = float(theorem_total_count_t[1].item())

            self.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}

                response_mask = model_inputs["response_mask"].bool()
                student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
                with torch.no_grad():
                    teacher_logits = self._forward_response_logits(
                        model_inputs,
                        temperature=temperature,
                        module=self.teacher_module,
                    )

                micro_weighted_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
                micro_weight_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
                contributing_samples = 0
                sample_level_metrics = {}
                student_logp_for_dense: torch.Tensor | None = None
                teacher_logp_for_dense: torch.Tensor | None = None
                student_token_logprobs: torch.Tensor | None = None
                forward_weight_result = None
                shared_token_tvd: torch.Tensor | None = None
                theorem_metric_token_tvd: torch.Tensor | None = None
                theorem_metric_blocks: torch.Tensor | None = None

                bsz, response_len, _ = student_logits.shape
                exact_loss_result = None
                if self.loss_type == "exact_block_count_wnll":
                    exact_loss_result = compute_exact_block_count_wnll_from_logits(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        responses=model_inputs["responses"],
                        gamma=self.gamma,
                        response_mask=response_mask,
                        dp_dtype=torch.float64,
                    )
                    contributing_samples = int(exact_loss_result.contributing_samples)
                    if contributing_samples > 0:
                        valid_token_counts = response_mask.sum(dim=-1)
                        valid_sample_mask = valid_token_counts > 0
                        valid_counts_f = valid_token_counts[valid_sample_mask].to(dtype=torch.float32)
                        valid_mask = response_mask[valid_sample_mask]
                        valid_mask_float = valid_mask.to(dtype=torch.float32)
                        valid_alpha = exact_loss_result.alpha[valid_sample_mask]
                        valid_alpha_sum = torch.sum(valid_alpha * valid_mask_float, dim=-1)
                        valid_alpha_mean = valid_alpha_sum / valid_counts_f
                        valid_alpha_nonzero_ratio = (
                            torch.sum((valid_alpha > 0).to(dtype=torch.float32) * valid_mask_float, dim=-1)
                            / valid_counts_f
                        )
                        valid_alpha_max = torch.max(valid_alpha.masked_fill(~valid_mask, 0.0), dim=-1).values
                        valid_omega_mean = self._masked_sample_mean(
                            exact_loss_result.dp_result.omega[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_reject_mean = self._masked_sample_mean(
                            exact_loss_result.dp_result.reject_prob[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_sample_loss = exact_loss_result.sample_loss_sum[valid_sample_mask].to(dtype=torch.float32)
                        valid_token_nll = exact_loss_result.token_nll[valid_sample_mask].to(dtype=torch.float32)
                        valid_token_nll_mean = self._masked_sample_mean(valid_token_nll, valid_mask)
                        valid_token_nll_max = torch.max(
                            valid_token_nll.masked_fill(~valid_mask, 0.0),
                            dim=-1,
                        ).values
                        sample_level_metrics = {
                            "distill/weight_nonzero_ratio": valid_alpha_nonzero_ratio.tolist(),
                            "distill/weight_mean": valid_alpha_mean.tolist(),
                            "distill/weight_max": valid_alpha_max.tolist(),
                            "distill/weight_sum": valid_alpha_sum.tolist(),
                            "distill/strict_alignment_ok": [1.0] * contributing_samples,
                            "distill/block_count_omega_mean": valid_omega_mean.tolist(),
                            "distill/reject_prob_mean": valid_reject_mean.tolist(),
                            "distill/student_nll_on_teacher_tokens_mean": valid_token_nll_mean.tolist(),
                            "distill/student_nll_on_teacher_tokens_max": valid_token_nll_max.tolist(),
                            "distill/sample_loss_sum": valid_sample_loss.tolist(),
                        }
                        if self.debug_recorder.enabled:
                            valid_indices = valid_sample_mask.nonzero(as_tuple=False).flatten().tolist()
                            for i in valid_indices:
                                sample_mask = response_mask[i]
                                sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                                sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                                sample_responses = model_inputs["responses"][i][sample_mask]
                                sample_alpha = exact_loss_result.alpha[i][sample_mask]
                                self.debug_recorder.maybe_record_sample(
                                    step=current_step,
                                    uid=sample_uid,
                                    sample_index=i,
                                    loss_type=self.loss_type,
                                    gamma=self.gamma,
                                    strict=self.strict,
                                    temperature=temperature,
                                    spec_accept_lens=sample_accept_lens,
                                    response_ids=sample_responses,
                                    token_weights=sample_alpha,
                                    student_logits=student_logits[i][sample_mask],
                                    teacher_logits=teacher_logits[i][sample_mask],
                                )
                elif self.weighting_mode == "uniform_mean":
                    # Off-policy baseline fast path: uniform weights on valid response tokens.
                    token_weights = response_mask.to(dtype=torch.float32)
                    valid_token_counts = response_mask.sum(dim=-1)
                    valid_sample_mask = valid_token_counts > 0
                    contributing_samples = int(valid_sample_mask.sum().item())
                    if contributing_samples > 0:
                        count = int(contributing_samples)
                        if self.loss_type == "tvd" and tvd_rembudget_bias_enabled:
                            with torch.no_grad():
                                teacher_token_logprobs = self._gather_response_token_logprobs_via_logsumexp(
                                    teacher_logits,
                                    model_inputs["responses"],
                                )
                                student_token_logprobs = self._gather_response_token_logprobs_via_logsumexp(
                                    student_logits.detach(),
                                    model_inputs["responses"],
                                )
                                forward_weight_result = compute_forward_remaining_budget_weights(
                                    teacher_logprobs=teacher_token_logprobs,
                                    student_logprobs=student_token_logprobs,
                                    gamma=self.gamma,
                                    response_mask=response_mask,
                                    kl_floor_coef=self.kl_floor_coef,
                                    dp_dtype=torch.float64,
                                )
                                raw_rembudget_weights = forward_weight_result.mixed_weight.to(
                                    device=student_logits.device,
                                    dtype=torch.float32,
                                )
                            score_mode = self.tvd_rembudget_bias_score_type
                            score_tensor = raw_rembudget_weights
                            if score_mode == "rembudget_tvd":
                                detached_student_prob = torch.softmax(student_logits.detach().float(), dim=-1)
                                detached_teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                detached_token_tvd = 0.5 * torch.sum(
                                    torch.abs(detached_student_prob - detached_teacher_prob),
                                    dim=-1,
                                )
                                theorem_metric_token_tvd = detached_token_tvd
                                score_tensor = raw_rembudget_weights * detached_token_tvd
                                token_weights = build_positive_boost_weights_from_score(
                                    score_tensor,
                                    response_mask,
                                    coef=self.tvd_rembudget_bias_coef,
                                    clip=self.tvd_rembudget_bias_clip,
                                )
                            else:
                                token_weights = build_mild_rembudget_bias_weights(
                                    raw_rembudget_weights,
                                    response_mask,
                                    coef=self.tvd_rembudget_bias_coef,
                                    power=self.tvd_rembudget_bias_power,
                                    eps=self.tvd_rembudget_bias_eps,
                                    clip=self.tvd_rembudget_bias_clip,
                                )
                            valid_mask = response_mask[valid_sample_mask]
                            valid_mask_float = valid_mask.to(dtype=torch.float32)
                            valid_bias = token_weights[valid_sample_mask]
                            valid_raw = raw_rembudget_weights[valid_sample_mask]
                            valid_score = score_tensor[valid_sample_mask]
                            valid_counts_f = valid_token_counts[valid_sample_mask].to(dtype=torch.float32)
                            bias_sum = torch.sum(valid_bias * valid_mask_float, dim=-1)
                            bias_mean = bias_sum / valid_counts_f
                            bias_max = torch.max(valid_bias.masked_fill(~valid_mask, 0.0), dim=-1).values
                            raw_mean = self._masked_sample_mean(valid_raw, valid_mask)
                            raw_max = torch.max(valid_raw.masked_fill(~valid_mask, 0.0), dim=-1).values
                            score_mean = self._masked_sample_mean(valid_score, valid_mask)
                            score_max = torch.max(valid_score.masked_fill(~valid_mask, 0.0), dim=-1).values
                            sample_level_metrics = {
                                "distill/weight_nonzero_ratio": [1.0] * count,
                                "distill/weight_mean": bias_mean.tolist(),
                                "distill/weight_max": bias_max.tolist(),
                                "distill/weight_sum": bias_sum.tolist(),
                                "distill/strict_alignment_ok": [1.0] * count,
                                "distill/rembudget_raw_weight_mean": raw_mean.tolist(),
                                "distill/rembudget_raw_weight_max": raw_max.tolist(),
                                "distill/rembudget_bias_score_mean": score_mean.tolist(),
                                "distill/rembudget_bias_score_max": score_max.tolist(),
                            }
                        elif self.loss_type == "tvd" and log_current_batch_theorem_rb_tvd:
                            with torch.no_grad():
                                teacher_token_logprobs = self._gather_response_token_logprobs_via_logsumexp(
                                    teacher_logits,
                                    model_inputs["responses"],
                                )
                                student_token_logprobs = self._gather_response_token_logprobs_via_logsumexp(
                                    student_logits.detach(),
                                    model_inputs["responses"],
                                )
                                forward_weight_result = compute_forward_remaining_budget_weights(
                                    teacher_logprobs=teacher_token_logprobs,
                                    student_logprobs=student_token_logprobs,
                                    gamma=self.gamma,
                                    response_mask=response_mask,
                                    kl_floor_coef=self.kl_floor_coef,
                                    dp_dtype=torch.float64,
                                )
                                detached_student_prob = torch.softmax(student_logits.detach().float(), dim=-1)
                                detached_teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                theorem_metric_token_tvd = 0.5 * torch.sum(
                                    torch.abs(detached_student_prob - detached_teacher_prob),
                                    dim=-1,
                                )
                        else:
                            sample_level_metrics = {
                                "distill/weight_nonzero_ratio": [1.0] * count,
                                "distill/weight_mean": [1.0] * count,
                                "distill/weight_max": [1.0] * count,
                                "distill/weight_sum": valid_token_counts[valid_sample_mask].to(dtype=torch.float32).tolist(),
                                "distill/strict_alignment_ok": [1.0] * count,
                            }
                        if self.debug_recorder.enabled:
                            valid_indices = valid_sample_mask.nonzero(as_tuple=False).flatten().tolist()
                            for i in valid_indices:
                                sample_mask = response_mask[i]
                                valid_tokens = int(valid_token_counts[i].item())
                                sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                                sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                                sample_responses = None
                                if "responses" in model_inputs:
                                    sample_responses = model_inputs["responses"][i][sample_mask]
                                sample_weights = token_weights[i][sample_mask]
                                self.debug_recorder.maybe_record_sample(
                                    step=current_step,
                                    uid=sample_uid,
                                    sample_index=i,
                                    loss_type=self.loss_type,
                                    gamma=self.gamma,
                                    strict=self.strict,
                                    temperature=temperature,
                                    spec_accept_lens=sample_accept_lens,
                                    response_ids=sample_responses,
                                    token_weights=sample_weights,
                                    student_logits=student_logits[i][sample_mask],
                                    teacher_logits=teacher_logits[i][sample_mask],
                                )
                elif self.weighting_mode == "waste":
                    token_weights = torch.zeros((bsz, response_len), dtype=torch.float32, device=student_logits.device)
                    for i in range(bsz):
                        sample_mask = response_mask[i]
                        valid_tokens = int(sample_mask.sum().item())
                        if valid_tokens <= 0:
                            continue

                        sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                        sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                        if self.strict and (sample_accept_lens is None or len(sample_accept_lens) == 0):
                            raise ValueError(
                                "Strict alignment requires non-empty spec_accept_lens for every valid sample. "
                                f"Got spec_accept_lens={sample_accept_lens!r}, valid_tokens={valid_tokens}, uid={sample_uid!r}. "
                                "This usually means SGLang patch metadata did not propagate."
                            )
                        sample_weights, weight_metrics = build_strict_weights(
                            spec_accept_lens=sample_accept_lens,
                            response_valid_len=valid_tokens,
                            gamma=self.gamma,
                            strict=self.strict,
                            device=student_logits.device,
                        )
                        append_to_dict(sample_level_metrics, weight_metrics)
                        token_weights[i, sample_mask] = sample_weights

                        sample_responses = None
                        if "responses" in model_inputs:
                            sample_responses = model_inputs["responses"][i][sample_mask]
                        self.debug_recorder.maybe_record_sample(
                            step=current_step,
                            uid=sample_uid,
                            sample_index=i,
                            loss_type=self.loss_type,
                            gamma=self.gamma,
                            strict=self.strict,
                            temperature=temperature,
                            spec_accept_lens=sample_accept_lens,
                            response_ids=sample_responses,
                            token_weights=sample_weights,
                            student_logits=student_logits[i][sample_mask],
                            teacher_logits=teacher_logits[i][sample_mask],
                        )
                        contributing_samples += 1
                else:
                    need_shared_dense_tvd = self.loss_type == "tvd"
                    need_dense_log_probs = (
                        soft_distill_needs_dense_log_probs(
                            loss_type=self.loss_type,
                            rembudget_unweighted_kl_coef=rembudget_unweighted_kl_coef,
                            rembudget_tvd_unweighted_fkl_coef=rembudget_tvd_unweighted_fkl_coef,
                        )
                        or need_shared_dense_tvd
                    )
                    with torch.no_grad():
                        if need_dense_log_probs:
                            teacher_logp_for_dense = torch.log_softmax(teacher_logits.float(), dim=-1)
                            teacher_token_logprobs = self._gather_response_token_logprobs_from_log_probs(
                                teacher_logp_for_dense,
                                model_inputs["responses"],
                            )
                        else:
                            teacher_token_logprobs = self._gather_response_token_logprobs(
                                teacher_logits,
                                model_inputs["responses"],
                            )
                    if need_dense_log_probs:
                        student_logp_for_dense = torch.log_softmax(student_logits.float(), dim=-1)
                        student_token_logprobs = self._gather_response_token_logprobs_from_log_probs(
                            student_logp_for_dense if theorem_unnormalized_rembudget_tvd else student_logp_for_dense.detach(),
                            model_inputs["responses"],
                        )
                    else:
                        with torch.no_grad():
                            student_token_logprobs = self._gather_response_token_logprobs(
                                student_logits.detach(),
                                model_inputs["responses"],
                            )
                    if need_shared_dense_tvd:
                        if student_logp_for_dense is None or teacher_logp_for_dense is None:
                            raise RuntimeError("Expected shared dense log-probs for remaining-budget TVD path.")
                        shared_token_tvd = self._token_tvd_from_log_probs(student_logp_for_dense, teacher_logp_for_dense)
                    if theorem_unnormalized_rembudget_tvd:
                        forward_weight_result = compute_forward_remaining_budget_weights(
                            teacher_logprobs=teacher_token_logprobs,
                            student_logprobs=student_token_logprobs,
                            gamma=self.gamma,
                            response_mask=response_mask,
                            kl_floor_coef=self.kl_floor_coef,
                            dp_dtype=torch.float64,
                        )
                        token_weights = forward_weight_result.remaining_budget_weight.to(
                            device=student_logits.device,
                            dtype=torch.float32,
                        )
                    else:
                        with torch.no_grad():
                            forward_weight_result = compute_forward_remaining_budget_weights(
                                teacher_logprobs=teacher_token_logprobs,
                                student_logprobs=student_token_logprobs,
                                gamma=self.gamma,
                                response_mask=response_mask,
                                kl_floor_coef=self.kl_floor_coef,
                                dp_dtype=torch.float64,
                            )
                            token_weights = forward_weight_result.mixed_weight.to(
                                device=student_logits.device,
                                dtype=torch.float32,
                            )

                    valid_token_counts = response_mask.sum(dim=-1)
                    valid_sample_mask = valid_token_counts > 0
                    contributing_samples = int(valid_sample_mask.sum().item())
                    if contributing_samples > 0:
                        valid_counts_f = valid_token_counts[valid_sample_mask].to(dtype=torch.float32)
                        valid_weights = token_weights[valid_sample_mask]
                        valid_mask_float = response_mask[valid_sample_mask].to(dtype=torch.float32)
                        valid_weight_sum = torch.sum(valid_weights * valid_mask_float, dim=-1)
                        valid_weight_mean = valid_weight_sum / valid_counts_f
                        valid_weight_nonzero_ratio = (
                            torch.sum((valid_weights > 0).to(dtype=torch.float32) * valid_mask_float, dim=-1)
                            / valid_counts_f
                        )
                        valid_weight_max = torch.max(valid_weights.masked_fill(~response_mask[valid_sample_mask], 0.0), dim=-1).values
                        valid_mask = response_mask[valid_sample_mask]
                        valid_rem_mean = self._masked_sample_mean(
                            forward_weight_result.remaining_budget_weight[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_pos_mean = self._masked_sample_mean(
                            forward_weight_result.expected_position[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_reject_mean = self._masked_sample_mean(
                            forward_weight_result.reject_prob[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        sample_level_metrics = {
                            "distill/weight_nonzero_ratio": valid_weight_nonzero_ratio.tolist(),
                            "distill/weight_mean": valid_weight_mean.tolist(),
                            "distill/weight_max": valid_weight_max.tolist(),
                            "distill/weight_sum": valid_weight_sum.tolist(),
                            "distill/strict_alignment_ok": [1.0] * contributing_samples,
                            "distill/remaining_budget_weight_mean": valid_rem_mean.tolist(),
                            "distill/expected_position_mean": valid_pos_mean.tolist(),
                            "distill/reject_prob_mean": valid_reject_mean.tolist(),
                        }

                local_valid_tokens = int(response_mask.sum().item())
                local_has_contrib = contributing_samples > 0
                local_has_contrib_flags.append(local_has_contrib)
                sync_device = student_logits.device
                dist_rank = -1
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    dist_rank = torch.distributed.get_rank()

                micro_metrics = self._init_micro_metrics(
                    local_has_contrib=local_has_contrib,
                    contributing_samples=contributing_samples,
                )

                micro_objective = torch.zeros((), device=student_logits.device, dtype=torch.float32)

                if local_has_contrib:
                    if self.loss_type == "exact_block_count_wnll":
                        assert exact_loss_result is not None
                        micro_weighted_sum = exact_loss_result.total_loss_sum
                        micro_weight_sum = exact_loss_result.alpha_weight_sum
                        micro_weighted_mean = exact_loss_result.loss
                        exact_aux_coef = exact_unweighted_aux_coef
                        exact_coef = 1.0 - exact_aux_coef
                        valid_token_nll_flat = exact_loss_result.token_nll[response_mask].to(dtype=torch.float32)
                        micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                        micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = micro_weighted_mean.detach().item()
                        micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                        micro_metrics[f"distill/{self.loss_type}_token_div_mean"] = valid_token_nll_flat.mean().item()
                        micro_metrics[f"distill/{self.loss_type}_token_div_max"] = valid_token_nll_flat.max().item()
                        micro_metrics["distill/student_nll_all_response_tokens_mean"] = valid_token_nll_flat.mean().item()
                        micro_metrics["distill/token_count"] = float(response_mask.sum().item())
                        micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                        if exact_aux_coef > 0.0:
                            uniform_token_weights = response_mask.to(dtype=torch.float32)
                            aux_loss, aux_weighted_sum, aux_weight_sum, _ = exact_aux_loss_fn(
                                student_logits,
                                {"logits": teacher_logits},
                                uniform_token_weights,
                                response_mask=response_mask,
                            )
                            micro_metrics[exact_aux_metric_name] = aux_loss.detach().item()
                            micro_metrics[exact_aux_mixed_metric_name] = (
                                exact_coef * micro_weighted_mean.detach().item()
                                + exact_aux_coef * aux_loss.detach().item()
                            )
                            exact_aux_step_weighted_sum += float(aux_weighted_sum.detach().item())
                            exact_aux_step_weight_sum += float(aux_weight_sum.detach().item())
                            # Mixed exact+aux under FSDP is treated as a convex combination of
                            # per-micro normalized objectives. The exact-only path below keeps the
                            # stricter global alpha-normalized aggregation, but mixing that with an
                            # additional dense-loss backward via autograd.grad + backward on the same
                            # graph proved fragile under FSDP sharding.
                            micro_objective = exact_coef * micro_weighted_mean + exact_aux_coef * aux_loss
                        else:
                            micro_objective = micro_weighted_sum
                        exact_total_weight_sum += float(micro_weight_sum.detach().item())
                        exact_step_loss_sum += float(micro_weighted_sum.detach().item())
                        exact_step_weight_sum += float(micro_weight_sum.detach().item())
                    else:
                        if student_token_logprobs is not None:
                            all_token_nll_mean, all_token_nll_max = self._token_nll_stats_from_token_logprobs(
                                student_token_logprobs,
                                response_mask,
                            )
                        else:
                            all_token_nll_mean, all_token_nll_max = self._student_token_nll_stats(
                                student_logits=student_logits,
                                responses=model_inputs["responses"],
                                response_mask=response_mask,
                            )
                        if (
                            self.weighting_mode == "remaining_budget_forward"
                            and self.loss_type in {"fkl", "tvd"}
                            and (
                                rembudget_unweighted_kl_coef > 0.0
                                or rembudget_tvd_unweighted_fkl_coef > 0.0
                            )
                        ):
                            # Reuse a single dense token-divergence graph, then reduce it two
                            # ways: remaining-budget weighted and plain unweighted. Calling the
                            # same dense loss twice on the same FSDP-backed logits graph can
                            # trigger fragile backward/view behavior, so keep this branch
                            # single-pass.
                            token_fkl = None
                            if self.loss_type == "fkl" or rembudget_tvd_unweighted_fkl_coef > 0.0:
                                student_logp = student_logp_for_dense
                                teacher_logp = teacher_logp_for_dense
                                if student_logp is None:
                                    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
                                if teacher_logp is None:
                                    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
                                teacher_prob = torch.exp(teacher_logp)
                                token_fkl = torch.sum(
                                    torch.where(
                                        teacher_prob > 0,
                                        teacher_prob * (teacher_logp - student_logp),
                                        teacher_logp.new_zeros(()),
                                    ),
                                    dim=-1,
                                )
                                if self.loss_type == "fkl":
                                    token_div = token_fkl
                                else:
                                    token_div = (
                                        shared_token_tvd
                                        if shared_token_tvd is not None
                                        else self._token_tvd_from_log_probs(student_logp, teacher_logp)
                                    )
                            else:
                                student_logp = student_logp_for_dense
                                teacher_logp = teacher_logp_for_dense
                                if shared_token_tvd is not None:
                                    token_div = shared_token_tvd
                                elif student_logp is not None and teacher_logp is not None:
                                    token_div = self._token_tvd_from_log_probs(student_logp, teacher_logp)
                                else:
                                    student_prob = torch.softmax(student_logits.float(), dim=-1)
                                    teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                    token_div = 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
                            flat_mask = response_mask.reshape(-1)
                            token_div_flat = token_div.reshape(-1)[flat_mask]
                            weight_flat = token_weights.reshape(-1)[flat_mask].to(dtype=torch.float32)
                            if log_current_batch_theorem_rb_tvd and self.loss_type == "tvd":
                                theorem_metric_token_tvd = token_div.detach()
                            if token_div_flat.numel() == 0:
                                micro_weighted_sum = student_logits.new_zeros((), dtype=torch.float32)
                                micro_weight_sum = student_logits.new_zeros((), dtype=torch.float32)
                                micro_weighted_mean = micro_weighted_sum
                                unweighted_loss = micro_weighted_sum
                                unweighted_fkl_loss = micro_weighted_sum
                                loss_metrics = {
                                    f"distill/{self.loss_type}_token_div_mean": 0.0,
                                    f"distill/{self.loss_type}_token_div_max": 0.0,
                                    "distill/token_count": 0.0,
                                }
                            else:
                                micro_weighted_sum = torch.sum(token_div_flat * weight_flat)
                                micro_weight_sum = torch.sum(weight_flat)
                                micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                                unweighted_loss = token_div_flat.mean()
                                if token_fkl is None:
                                    unweighted_fkl_loss = micro_weighted_sum * 0.0
                                else:
                                    unweighted_fkl_loss = token_fkl.reshape(-1)[flat_mask].mean()
                                loss_metrics = {
                                    f"distill/{self.loss_type}_token_div_mean": token_div_flat.mean().item(),
                                    f"distill/{self.loss_type}_token_div_max": token_div_flat.max().item(),
                                    "distill/token_count": float(token_div_flat.numel()),
                                }
                            append_to_dict(sample_level_metrics, loss_metrics)
                            micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                            if self.loss_type == "fkl":
                                micro_metrics["distill/weighted_kl_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                            if rembudget_tvd_unweighted_fkl_coef > 0.0:
                                mix_coef = rembudget_tvd_unweighted_fkl_coef
                                micro_metrics["distill/rembudget_tvd_unweighted_fkl_mix_coef"] = mix_coef
                                micro_metrics["distill/unweighted_fkl_mean"] = unweighted_fkl_loss.detach().item()
                                micro_metrics["distill/mixed_rembudget_tvd_unweighted_fkl_mean"] = (
                                    (1.0 - mix_coef) * micro_weighted_mean.detach().item()
                                    + mix_coef * unweighted_fkl_loss.detach().item()
                                )
                                micro_objective = (
                                    (1.0 - mix_coef) * micro_weighted_mean
                                    + mix_coef * unweighted_fkl_loss
                                )
                            else:
                                mix_coef = rembudget_unweighted_kl_coef
                                micro_metrics["distill/rembudget_unweighted_mix_coef"] = mix_coef
                                micro_metrics[f"distill/unweighted_{self.loss_type}_mean"] = unweighted_loss.detach().item()
                                micro_metrics[f"distill/mixed_rembudget_unweighted_{self.loss_type}_mean"] = (
                                    (1.0 - mix_coef) * micro_weighted_mean.detach().item()
                                    + mix_coef * unweighted_loss.detach().item()
                                )
                                micro_objective = (1.0 - mix_coef) * micro_weighted_mean + mix_coef * unweighted_loss
                            micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                            micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                            micro_metrics["actor/distill_loss"] = micro_objective.detach().item()
                        else:
                            if theorem_unnormalized_rembudget_tvd:
                                if shared_token_tvd is not None:
                                    token_div = shared_token_tvd
                                else:
                                    student_prob = torch.softmax(student_logits.float(), dim=-1)
                                    teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                    token_div = 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
                                flat_mask = response_mask.reshape(-1)
                                token_div_flat = token_div.reshape(-1)[flat_mask]
                                weight_flat = token_weights.reshape(-1)[flat_mask].to(dtype=torch.float32)
                                if token_div_flat.numel() == 0:
                                    micro_weighted_sum = student_logits.new_zeros((), dtype=torch.float32)
                                    micro_weight_sum = student_logits.new_zeros((), dtype=torch.float32)
                                    micro_weighted_mean = micro_weighted_sum
                                    theorem_blocks = student_logits.new_zeros((0,), dtype=torch.float32)
                                    loss_metrics = {
                                        f"distill/{self.loss_type}_token_div_mean": 0.0,
                                        f"distill/{self.loss_type}_token_div_max": 0.0,
                                        "distill/token_count": 0.0,
                                    }
                                else:
                                    micro_weighted_sum = torch.sum(token_div_flat * weight_flat)
                                    micro_weight_sum = torch.sum(weight_flat)
                                    micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                                    theorem_blocks = compute_theorem_rb_tvd_blocks_per_sample(
                                        remaining_budget_weight=forward_weight_result.remaining_budget_weight,
                                        token_tvd=token_div,
                                        response_mask=response_mask,
                                        gamma=self.gamma,
                                    )
                                    theorem_metric_blocks = theorem_blocks.detach()
                                    loss_metrics = {
                                        f"distill/{self.loss_type}_token_div_mean": token_div_flat.mean().item(),
                                        f"distill/{self.loss_type}_token_div_max": token_div_flat.max().item(),
                                        "distill/token_count": float(token_div_flat.numel()),
                                    }
                                append_to_dict(sample_level_metrics, loss_metrics)
                                micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                                micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = (
                                    micro_weighted_mean.detach().item()
                                )
                                micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                                micro_metrics["distill/theorem_unnormalized_rb_tvd_sum_blocks"] = (
                                    theorem_blocks.sum().detach().item() if theorem_blocks.numel() > 0 else 0.0
                                )
                                if theorem_total_sample_count > 0.0:
                                    micro_metrics["distill/theorem_backward_avg_tokens"] = (
                                        theorem_total_token_count / theorem_total_sample_count
                                    )
                                micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                                micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                                micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                                theorem_denom = theorem_total_sample_count
                                if self.rembudget_tvd_backward_length_normalize:
                                    theorem_denom = theorem_total_token_count
                                if theorem_denom > 0.0:
                                    micro_objective = theorem_blocks.sum() / float(theorem_denom)
                                    if theorem_world_size > 1:
                                        # FSDP/DDP averages gradients across ranks; rescale so the
                                        # resulting gradient matches the exact global theorem objective.
                                        micro_objective = micro_objective * float(theorem_world_size)
                                else:
                                    micro_objective = micro_weighted_sum * 0.0
                                theorem_step_block_sum += float(theorem_blocks.sum().detach().item())
                                theorem_step_sample_count += float(theorem_blocks.numel())
                                theorem_step_token_count += float(response_mask.sum().item())
                            elif self.loss_type == "tvd" and shared_token_tvd is not None:
                                token_div = shared_token_tvd
                                flat_mask = response_mask.reshape(-1)
                                token_div_flat = token_div.reshape(-1)[flat_mask]
                                weight_flat = token_weights.reshape(-1)[flat_mask].to(dtype=torch.float32)
                                if token_div_flat.numel() == 0:
                                    micro_weighted_sum = student_logits.new_zeros((), dtype=torch.float32)
                                    micro_weight_sum = student_logits.new_zeros((), dtype=torch.float32)
                                    loss_metrics = {
                                        "distill/tvd_token_div_mean": 0.0,
                                        "distill/tvd_token_div_max": 0.0,
                                        "distill/token_count": 0.0,
                                    }
                                else:
                                    micro_weighted_sum = torch.sum(token_div_flat * weight_flat)
                                    micro_weight_sum = torch.sum(weight_flat)
                                    loss_metrics = {
                                        "distill/tvd_token_div_mean": token_div_flat.mean().item(),
                                        "distill/tvd_token_div_max": token_div_flat.max().item(),
                                        "distill/token_count": float(token_div_flat.numel()),
                                    }
                                append_to_dict(sample_level_metrics, loss_metrics)

                                micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                                micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                                micro_metrics["distill/weighted_tvd_mean"] = micro_weighted_mean.detach().item()
                                micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                                micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                                micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                                micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                                if log_current_batch_theorem_rb_tvd and forward_weight_result is not None:
                                    theorem_metric_token_tvd = token_div.detach()
                                micro_objective = micro_weighted_mean
                                micro_metrics["actor/distill_loss"] = micro_objective.detach().item()
                            else:
                                _, micro_weighted_sum, micro_weight_sum, loss_metrics = self.loss_fn(
                                    student_logits,
                                    {"logits": teacher_logits},
                                    token_weights,
                                    response_mask=response_mask,
                                )
                                append_to_dict(sample_level_metrics, loss_metrics)

                                micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                                micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                                micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = (
                                    micro_weighted_mean.detach().item()
                                )
                                micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                                if self.loss_type == "fkl":
                                    micro_metrics["distill/weighted_kl_mean"] = micro_weighted_mean.detach().item()
                                micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                                micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                                micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                                if (
                                    log_current_batch_theorem_rb_tvd
                                    and self.loss_type == "tvd"
                                    and forward_weight_result is not None
                                    and theorem_metric_token_tvd is None
                                ):
                                    with torch.no_grad():
                                        detached_student_prob = torch.softmax(student_logits.detach().float(), dim=-1)
                                        detached_teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                        theorem_metric_token_tvd = 0.5 * torch.sum(
                                            torch.abs(detached_student_prob - detached_teacher_prob),
                                            dim=-1,
                                        )

                                if self.weighting_mode == "uniform_mean":
                                    if self.loss_type == "tvd" and tvd_rembudget_bias_enabled:
                                        micro_metrics["distill/tvd_rembudget_bias_coef"] = self.tvd_rembudget_bias_coef
                                        micro_metrics["distill/tvd_rembudget_bias_power"] = self.tvd_rembudget_bias_power
                                        micro_metrics["distill/tvd_rembudget_bias_clip"] = self.tvd_rembudget_bias_clip
                                        micro_metrics["distill/tvd_rembudget_bias_enabled"] = 1.0
                                        micro_metrics["distill/tvd_rembudget_bias_uses_tvd_score"] = float(
                                            self.tvd_rembudget_bias_score_type == "rembudget_tvd"
                                        )
                                        token_count = float(loss_metrics["distill/token_count"])
                                        if token_count > 0.0:
                                            micro_metrics["distill/mild_rembudget_tvd_mean"] = (
                                                micro_weighted_sum.detach().item() / token_count
                                            )
                                    # Backprop per-micro weighted-sum contribution normalized by
                                    # mini-batch total valid-token count to preserve exact token-level mean.
                                    if uniform_total_weight_sum > 0.0:
                                        micro_objective = micro_weighted_sum / float(uniform_total_weight_sum)
                                    else:
                                        micro_objective = micro_weighted_sum * 0.0
                                    uniform_step_weighted_sum += float(micro_weighted_sum.detach().item())
                                    uniform_step_token_count += float(loss_metrics["distill/token_count"])
                                else:
                                    # Optimize weighted token-mean objective for soft distillation:
                                    #   (sum_n w_n * D_n) / (sum_n w_n)
                                    # This keeps the weighting semantics at the token level instead of
                                    # implicitly upweighting long responses via per-sample sums.
                                    micro_objective = micro_weighted_mean
                                    micro_metrics["actor/distill_loss"] = micro_objective.detach().item()

                        if (
                            log_current_batch_theorem_rb_tvd
                            and self.loss_type == "tvd"
                            and forward_weight_result is not None
                        ):
                            if theorem_metric_blocks is not None:
                                theorem_blocks = theorem_metric_blocks
                            elif theorem_metric_token_tvd is not None:
                                theorem_blocks = compute_theorem_rb_tvd_blocks_per_sample(
                                    remaining_budget_weight=forward_weight_result.remaining_budget_weight,
                                    token_tvd=theorem_metric_token_tvd,
                                    response_mask=response_mask,
                                    gamma=self.gamma,
                                )
                            else:
                                theorem_blocks = None
                            if theorem_blocks is not None:
                                micro_metrics["distill/current_batch_theorem_rb_tvd_mean_blocks"] = (
                                    theorem_blocks.mean().item()
                                )
                                micro_metrics["distill/current_batch_theorem_rb_tvd_sum_blocks"] = (
                                    theorem_blocks.sum().item()
                                )

                if local_has_contrib:
                    if self.loss_type == "exact_block_count_wnll" and exact_unweighted_aux_coef > 0.0:
                        scaled_loss = micro_objective / grad_accum_steps
                    elif self.weighting_mode == "uniform_mean" or theorem_unnormalized_rembudget_tvd:
                        scaled_loss = micro_objective
                    else:
                        scaled_loss = micro_objective / grad_accum_steps
                else:
                    # Keep FSDP collectives aligned even when this rank has no valid tokens.
                    scaled_loss = student_logits.sum() * 0.0

                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_records.append(
                    {
                        "metrics": micro_metrics,
                        "local_has_contrib": local_has_contrib,
                        "local_valid_tokens": local_valid_tokens,
                        "contributing_samples": contributing_samples,
                        "micro_batch_size": bsz,
                        "dist_rank": dist_rank,
                    }
                )

            global_has_contrib_flags = list(local_has_contrib_flags)
            if (
                len(local_has_contrib_flags) > 0
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                global_flags_t = torch.tensor(
                    [1 if flag else 0 for flag in local_has_contrib_flags],
                    dtype=torch.int32,
                    device=sync_device,
                )
                torch.distributed.all_reduce(global_flags_t, op=torch.distributed.ReduceOp.SUM)
                global_has_contrib_flags = [bool(v > 0) for v in global_flags_t.tolist()]

            global_has_contrib = any(global_has_contrib_flags)
            for record, micro_global_has_contrib in zip(micro_records, global_has_contrib_flags):
                micro_metrics = record["metrics"]
                local_has_contrib = bool(record["local_has_contrib"])
                if (not local_has_contrib) and micro_global_has_contrib:
                    micro_metrics["distill/rank_zero_contrib_while_global_nonzero"] = 1.0
                    if self._sync_debug:
                        logger.warning(
                            "[distill-sync-debug] step=%s rank=%s local_valid_tokens=%s "
                            "contributing_samples=%s micro_batch_size=%s",
                            current_step,
                            int(record["dist_rank"]),
                            int(record["local_valid_tokens"]),
                            int(record["contributing_samples"]),
                            int(record["micro_batch_size"]),
                        )
                append_to_dict(metrics, micro_metrics)

            if global_has_contrib:
                if self.loss_type == "exact_block_count_wnll":
                    if exact_unweighted_aux_coef <= 0.0 and exact_total_weight_sum > 0.0:
                        grad_scale = 1.0 / float(exact_total_weight_sum)
                        for param in self.actor_module.parameters():
                            if param.grad is not None:
                                param.grad.mul_(grad_scale)
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
                if log_post_update_batch_theorem_rb_tvd:
                    append_to_dict(
                        metrics,
                        self._compute_post_update_batch_theorem_rb_tvd_metrics(
                            micro_batches=micro_batches,
                            temperature=temperature,
                            pad_token_id=pad_token_id,
                        ),
                    )
                did_update_step = True
            else:
                self.actor_optimizer.zero_grad()

        if self.weighting_mode == "uniform_mean":
            if self.loss_type == "exact_block_count_wnll":
                exact_mean = exact_step_loss_sum / exact_step_weight_sum if exact_step_weight_sum > 0.0 else 0.0
                if exact_unweighted_aux_coef > 0.0:
                    aux_mean = exact_aux_step_weighted_sum / exact_aux_step_weight_sum if exact_aux_step_weight_sum > 0.0 else 0.0
                    step_loss = (1.0 - exact_unweighted_aux_coef) * exact_mean + exact_unweighted_aux_coef * aux_mean
                    append_to_dict(metrics, {exact_aux_metric_name: aux_mean})
                else:
                    step_loss = exact_mean
            elif uniform_step_token_count > 0.0:
                step_loss = uniform_step_weighted_sum / uniform_step_token_count
            else:
                step_loss = 0.0
            append_to_dict(metrics, {"actor/distill_loss": step_loss})
        elif theorem_unnormalized_rembudget_tvd:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                theorem_step_t = torch.tensor(
                    [theorem_step_block_sum, theorem_step_sample_count, theorem_step_token_count],
                    dtype=torch.float32,
                    device=torch.device(self.device_name, get_device_id()),
                )
                torch.distributed.all_reduce(theorem_step_t, op=torch.distributed.ReduceOp.SUM)
                theorem_step_block_sum = float(theorem_step_t[0].item())
                theorem_step_sample_count = float(theorem_step_t[1].item())
                theorem_step_token_count = float(theorem_step_t[2].item())
            theorem_step_denom = theorem_step_sample_count
            if self.rembudget_tvd_backward_length_normalize:
                theorem_step_denom = theorem_step_token_count
            step_loss = theorem_step_block_sum / theorem_step_denom if theorem_step_denom > 0.0 else 0.0
            append_to_dict(metrics, {"actor/distill_loss": step_loss})
        append_to_dict(metrics, {"distill/exact_unweighted_aux_coef": exact_unweighted_aux_coef})
        append_to_dict(metrics, {"distill/did_update": float(did_update_step)})
        self.actor_optimizer.zero_grad()
        return metrics
