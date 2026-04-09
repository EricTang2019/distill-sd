from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.autograd.function import once_differentiable


@dataclass
class BlockCountOmegaResult:
    teacher_logprobs: torch.Tensor
    student_logprobs: torch.Tensor
    log_ratio_p_over_q: torch.Tensor
    ratio_p_over_q: torch.Tensor
    reject_prob: torch.Tensor
    occupancy_u: torch.Tensor
    future_block_count_U: torch.Tensor
    advantage_A: torch.Tensor
    omega: torch.Tensor
    alpha: torch.Tensor
    response_mask: torch.Tensor


@dataclass
class ForwardRemainingBudgetResult:
    teacher_logprobs: torch.Tensor
    student_logprobs: torch.Tensor
    log_ratio_p_over_q: torch.Tensor
    ratio_p_over_q: torch.Tensor
    reject_prob: torch.Tensor
    occupancy_u: torch.Tensor
    expected_position: torch.Tensor
    remaining_budget_weight: torch.Tensor
    mixed_weight: torch.Tensor
    response_mask: torch.Tensor


def _ensure_2d(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    if tensor.ndim != 2:
        raise ValueError(f"Expected rank-1 or rank-2 tensor, got shape={tuple(tensor.shape)}")
    return tensor, False


def compute_block_count_omega(
    *,
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    gamma: int,
    response_mask: Optional[torch.Tensor] = None,
    clamp_log_ratio: float = 60.0,
    dp_dtype: torch.dtype = torch.float64,
) -> BlockCountOmegaResult:
    """Compute exact one-rollout block-count DP weights from token logprobs.

    Inputs are token-level log-probabilities on a fixed teacher rollout x:
      - teacher_logprobs[n] = log q_n^x
      - student_logprobs[n] = log p_n^x

    The returned `alpha` matches the detached weighted-NLL prefactor in
    `expectation.tex`:
      alpha_n = omega_n * 1[p_n^x < q_n^x] * p_n^x / q_n^x
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    teacher_logprobs, squeeze = _ensure_2d(teacher_logprobs)
    student_logprobs, _ = _ensure_2d(student_logprobs)
    if teacher_logprobs.shape != student_logprobs.shape:
        raise ValueError(
            f"teacher/student logprob shape mismatch: {tuple(teacher_logprobs.shape)} vs "
            f"{tuple(student_logprobs.shape)}"
        )

    batch_size, max_len = teacher_logprobs.shape
    device = teacher_logprobs.device

    if response_mask is None:
        response_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    else:
        response_mask, _ = _ensure_2d(response_mask)
        response_mask = response_mask.to(device=device, dtype=torch.bool)
        if tuple(response_mask.shape) != (batch_size, max_len):
            raise ValueError(
                f"response_mask shape mismatch: expected {(batch_size, max_len)}, got {tuple(response_mask.shape)}"
            )

    teacher_logprobs = teacher_logprobs.to(dtype=dp_dtype)
    student_logprobs = student_logprobs.to(dtype=dp_dtype)

    log_ratio = (student_logprobs - teacher_logprobs).clamp(min=-clamp_log_ratio, max=clamp_log_ratio)
    ratio = torch.exp(log_ratio)
    reject_prob = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

    valid_mask = response_mask.to(dtype=dp_dtype)

    # Vectorize the teacher-rollout DP over the batch dimension so GPU execution
    # follows the forward/backward recursions from expectation.tex without a
    # Python loop over individual samples.
    occupancy_u = torch.zeros((batch_size, max_len, gamma + 1), dtype=dp_dtype, device=device)
    current_u = torch.zeros((batch_size, gamma + 1), dtype=dp_dtype, device=device)
    current_u[:, 0] = 1.0
    for token_idx in range(max_len):
        occupancy_u[:, token_idx, :] = current_u
        reject = reject_prob[:, token_idx].unsqueeze(-1)
        queried_mass = current_u[:, :gamma].sum(dim=-1, keepdim=True)
        next_u = torch.empty_like(current_u)
        next_u[:, :1] = current_u[:, gamma : gamma + 1] + reject * queried_mass
        next_u[:, 1 : gamma + 1] = (1.0 - reject) * current_u[:, :gamma]
        token_valid = response_mask[:, token_idx].unsqueeze(-1)
        current_u = torch.where(token_valid, next_u, current_u)
    occupancy_u = occupancy_u * valid_mask.unsqueeze(-1)

    future_U = torch.zeros((batch_size, max_len + 1, gamma + 1), dtype=dp_dtype, device=device)
    current_future = torch.zeros((batch_size, gamma + 1), dtype=dp_dtype, device=device)
    for token_idx in range(max_len - 1, -1, -1):
        reject = reject_prob[:, token_idx].unsqueeze(-1)
        current_candidate = torch.empty_like(current_future)
        current_candidate[:, gamma : gamma + 1] = 1.0 + current_future[:, :1]
        current_candidate[:, :gamma] = (
            reject * (1.0 + current_future[:, :1]) + (1.0 - reject) * current_future[:, 1 : gamma + 1]
        )
        token_valid = response_mask[:, token_idx].unsqueeze(-1)
        future_U[:, token_idx, :] = torch.where(token_valid, current_candidate, torch.zeros_like(current_candidate))
        current_future = torch.where(token_valid, current_candidate, current_future)

    advantage_A = (
        1.0 + future_U[:, 1:, 0].unsqueeze(-1) - future_U[:, 1:, 1 : gamma + 1]
    ) * valid_mask.unsqueeze(-1)
    omega = (occupancy_u[:, :, :gamma] * advantage_A).sum(dim=-1)

    indicator = (student_logprobs < teacher_logprobs).to(dtype=dp_dtype)
    alpha = omega * indicator * ratio

    if squeeze:
        teacher_logprobs = teacher_logprobs.squeeze(0)
        student_logprobs = student_logprobs.squeeze(0)
        log_ratio = log_ratio.squeeze(0)
        ratio = ratio.squeeze(0)
        reject_prob = reject_prob.squeeze(0)
        occupancy_u = occupancy_u.squeeze(0)
        future_U = future_U.squeeze(0)
        advantage_A = advantage_A.squeeze(0)
        omega = omega.squeeze(0)
        alpha = alpha.squeeze(0)
        response_mask = response_mask.squeeze(0)

    return BlockCountOmegaResult(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        log_ratio_p_over_q=log_ratio,
        ratio_p_over_q=ratio,
        reject_prob=reject_prob,
        occupancy_u=occupancy_u,
        future_block_count_U=future_U,
        advantage_A=advantage_A,
        omega=omega,
        alpha=alpha,
        response_mask=response_mask,
    )


def _compute_forward_remaining_budget_tensors(
    *,
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    gamma: int,
    response_mask: torch.Tensor,
    kl_floor_coef: float = 0.0,
    clamp_log_ratio: float = 60.0,
    dp_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, ...]:
    batch_size, max_len = teacher_logprobs.shape
    device = teacher_logprobs.device
    teacher_logprobs = teacher_logprobs.to(dtype=dp_dtype)
    student_logprobs = student_logprobs.to(dtype=dp_dtype)
    response_mask = response_mask.to(device=device, dtype=torch.bool)

    log_ratio = (student_logprobs - teacher_logprobs).clamp(min=-clamp_log_ratio, max=clamp_log_ratio)
    ratio = torch.exp(log_ratio)
    reject_prob = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

    occupancy_u = torch.zeros((batch_size, max_len, gamma + 1), dtype=dp_dtype, device=device)
    expected_position = torch.zeros((batch_size, max_len), dtype=dp_dtype, device=device)
    remaining_budget_weight = torch.zeros((batch_size, max_len), dtype=dp_dtype, device=device)

    current_u = torch.zeros((batch_size, gamma + 1), dtype=dp_dtype, device=device)
    current_u[:, 0] = 1.0
    state_index = torch.arange(gamma + 1, dtype=dp_dtype, device=device)
    valid_mask = response_mask.to(dtype=dp_dtype)

    for token_idx in range(max_len):
        occupancy_u[:, token_idx, :] = current_u
        expected_position[:, token_idx] = torch.sum(current_u * state_index.unsqueeze(0), dim=-1)
        remaining_budget_weight[:, token_idx] = torch.clamp(
            (float(gamma) - expected_position[:, token_idx]) / float(gamma),
            min=0.0,
            max=1.0,
        )

        reject = reject_prob[:, token_idx]
        next_u = torch.zeros_like(current_u)
        queried_mass = current_u[:, :gamma].sum(dim=-1)
        next_u[:, 0] = current_u[:, gamma] + reject * queried_mass
        next_u[:, 1 : gamma + 1] = (1.0 - reject).unsqueeze(-1) * current_u[:, :gamma]

        token_valid = valid_mask[:, token_idx].unsqueeze(-1)
        current_u = token_valid * next_u + (1.0 - token_valid) * current_u

    occupancy_u = occupancy_u * valid_mask.unsqueeze(-1)
    expected_position = expected_position * valid_mask
    remaining_budget_weight = remaining_budget_weight * valid_mask
    mixed_weight = (float(kl_floor_coef) + (1.0 - float(kl_floor_coef)) * remaining_budget_weight) * valid_mask

    return (
        teacher_logprobs,
        student_logprobs,
        log_ratio,
        ratio,
        reject_prob,
        occupancy_u,
        expected_position,
        remaining_budget_weight,
        mixed_weight,
    )


class _ForwardRemainingBudgetFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        teacher_logprobs: torch.Tensor,
        student_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
        gamma: int,
        kl_floor_coef: float,
        clamp_log_ratio: float,
        dp_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        outputs = _compute_forward_remaining_budget_tensors(
            teacher_logprobs=teacher_logprobs,
            student_logprobs=student_logprobs,
            gamma=int(gamma),
            response_mask=response_mask,
            kl_floor_coef=float(kl_floor_coef),
            clamp_log_ratio=float(clamp_log_ratio),
            dp_dtype=dp_dtype,
        )
        (
            teacher_logprobs_out,
            student_logprobs_out,
            log_ratio,
            ratio,
            reject_prob,
            occupancy_u,
            expected_position,
            remaining_budget_weight,
            mixed_weight,
        ) = outputs
        ctx.gamma = int(gamma)
        ctx.kl_floor_coef = float(kl_floor_coef)
        ctx.clamp_log_ratio = float(clamp_log_ratio)
        ctx.teacher_input_dtype = teacher_logprobs.dtype
        ctx.student_input_dtype = student_logprobs.dtype
        ctx.save_for_backward(
            log_ratio,
            ratio,
            reject_prob,
            occupancy_u,
            remaining_budget_weight,
            response_mask.to(dtype=torch.bool),
        )
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        (
            grad_teacher_out,
            grad_student_out,
            grad_log_ratio_out,
            grad_ratio_out,
            grad_reject_out,
            grad_occupancy_out,
            grad_expected_out,
            grad_remaining_budget_out,
            grad_mixed_out,
        ) = grad_outputs
        (
            log_ratio,
            ratio,
            reject_prob,
            occupancy_u,
            remaining_budget_weight,
            response_mask,
        ) = ctx.saved_tensors

        dtype = log_ratio.dtype
        device = log_ratio.device
        gamma = int(ctx.gamma)
        valid_mask = response_mask.to(dtype=dtype)
        state_index = torch.arange(gamma + 1, dtype=dtype, device=device).unsqueeze(0)

        batch_size, max_len = log_ratio.shape

        grad_teacher = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_student = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_log_ratio_total = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_ratio_total = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_reject_total = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_expected_total = torch.zeros((batch_size, max_len), dtype=dtype, device=device)
        grad_occupancy_total = torch.zeros_like(occupancy_u)

        if grad_teacher_out is not None:
            grad_teacher = grad_teacher + grad_teacher_out.to(dtype=dtype)
        if grad_student_out is not None:
            grad_student = grad_student + grad_student_out.to(dtype=dtype)
        if grad_log_ratio_out is not None:
            grad_log_ratio_total = grad_log_ratio_total + grad_log_ratio_out.to(dtype=dtype)
        if grad_ratio_out is not None:
            grad_ratio_total = grad_ratio_total + grad_ratio_out.to(dtype=dtype)
        if grad_reject_out is not None:
            grad_reject_total = grad_reject_total + grad_reject_out.to(dtype=dtype)
        if grad_occupancy_out is not None:
            grad_occupancy_total = grad_occupancy_total + grad_occupancy_out.to(dtype=dtype) * valid_mask.unsqueeze(-1)
        if grad_expected_out is not None:
            grad_expected_total = grad_expected_total + grad_expected_out.to(dtype=dtype) * valid_mask
        if grad_remaining_budget_out is not None:
            active_weight = ((remaining_budget_weight > 0.0) & (remaining_budget_weight < 1.0)).to(dtype=dtype)
            grad_expected_total = grad_expected_total - (
                grad_remaining_budget_out.to(dtype=dtype) * valid_mask * active_weight / float(gamma)
            )
        if grad_mixed_out is not None:
            active_weight = ((remaining_budget_weight > 0.0) & (remaining_budget_weight < 1.0)).to(dtype=dtype)
            grad_expected_total = grad_expected_total - (
                grad_mixed_out.to(dtype=dtype)
                * valid_mask
                * active_weight
                * (1.0 - float(ctx.kl_floor_coef))
                / float(gamma)
            )

        adj_current = torch.zeros((batch_size, gamma + 1), dtype=dtype, device=device)

        for token_idx in range(max_len - 1, -1, -1):
            token_valid = response_mask[:, token_idx].to(dtype=dtype)
            u_t = occupancy_u[:, token_idx, :]

            adj_u = adj_current * (1.0 - token_valid).unsqueeze(-1)
            adj_u = adj_u + grad_occupancy_total[:, token_idx, :]
            adj_u = adj_u + grad_expected_total[:, token_idx].unsqueeze(-1) * state_index

            adj_next = adj_current * token_valid.unsqueeze(-1)
            reject_t = reject_prob[:, token_idx]
            queried_mass = u_t[:, :gamma].sum(dim=-1)

            grad_reject_total[:, token_idx] = grad_reject_total[:, token_idx] + token_valid * (
                adj_next[:, 0] * queried_mass - torch.sum(adj_next[:, 1:] * u_t[:, :gamma], dim=-1)
            )

            adj_u[:, :gamma] = adj_u[:, :gamma] + (
                adj_next[:, 0].unsqueeze(-1) * reject_t.unsqueeze(-1)
                + adj_next[:, 1:] * (1.0 - reject_t).unsqueeze(-1)
            )
            adj_u[:, gamma] = adj_u[:, gamma] + adj_next[:, 0]
            adj_current = adj_u

        grad_ratio_total = grad_ratio_total - grad_reject_total * (reject_prob > 0.0).to(dtype=dtype)
        grad_log_ratio_total = grad_log_ratio_total + grad_ratio_total * ratio
        unclamped = ((log_ratio > -float(ctx.clamp_log_ratio)) & (log_ratio < float(ctx.clamp_log_ratio))).to(
            dtype=dtype
        )
        grad_delta = grad_log_ratio_total * unclamped
        grad_student = grad_student + grad_delta
        grad_teacher = grad_teacher - grad_delta

        return (
            grad_teacher.to(dtype=ctx.teacher_input_dtype),
            grad_student.to(dtype=ctx.student_input_dtype),
            None,
            None,
            None,
            None,
            None,
        )


def compute_forward_remaining_budget_weights(
    *,
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    gamma: int,
    response_mask: Optional[torch.Tensor] = None,
    kl_floor_coef: float = 0.0,
    clamp_log_ratio: float = 60.0,
    dp_dtype: torch.dtype = torch.float64,
) -> ForwardRemainingBudgetResult:
    """Compute forward-only remaining-budget weights from teacher/student token logprobs.

    This implements the forward occupancy DP from Proposition~forward-recursion and the
    normalized remaining-budget weight

        w_rem[n] = (gamma - E[S_n | prefix]) / gamma

    used by the forward-only dense KL surrogate in `expectation.tex`.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if not (0.0 <= float(kl_floor_coef) <= 1.0):
        raise ValueError(f"kl_floor_coef must lie in [0, 1], got {kl_floor_coef}")

    teacher_logprobs, squeeze = _ensure_2d(teacher_logprobs)
    student_logprobs, _ = _ensure_2d(student_logprobs)
    if teacher_logprobs.shape != student_logprobs.shape:
        raise ValueError(
            f"teacher/student logprob shape mismatch: {tuple(teacher_logprobs.shape)} vs "
            f"{tuple(student_logprobs.shape)}"
        )

    batch_size, max_len = teacher_logprobs.shape
    device = teacher_logprobs.device

    if response_mask is None:
        response_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    else:
        response_mask, _ = _ensure_2d(response_mask)
        response_mask = response_mask.to(device=device, dtype=torch.bool)
        if tuple(response_mask.shape) != (batch_size, max_len):
            raise ValueError(
                f"response_mask shape mismatch: expected {(batch_size, max_len)}, got {tuple(response_mask.shape)}"
            )

    use_custom_backward = torch.is_grad_enabled() and (
        teacher_logprobs.requires_grad or student_logprobs.requires_grad
    )
    if use_custom_backward:
        (
            teacher_logprobs,
            student_logprobs,
            log_ratio,
            ratio,
            reject_prob,
            occupancy_u,
            expected_position,
            remaining_budget_weight,
            mixed_weight,
        ) = _ForwardRemainingBudgetFunction.apply(
            teacher_logprobs,
            student_logprobs,
            response_mask,
            int(gamma),
            float(kl_floor_coef),
            float(clamp_log_ratio),
            dp_dtype,
        )
    else:
        (
            teacher_logprobs,
            student_logprobs,
            log_ratio,
            ratio,
            reject_prob,
            occupancy_u,
            expected_position,
            remaining_budget_weight,
            mixed_weight,
        ) = _compute_forward_remaining_budget_tensors(
            teacher_logprobs=teacher_logprobs,
            student_logprobs=student_logprobs,
            gamma=gamma,
            response_mask=response_mask,
            kl_floor_coef=kl_floor_coef,
            clamp_log_ratio=clamp_log_ratio,
            dp_dtype=dp_dtype,
        )

    if squeeze:
        teacher_logprobs = teacher_logprobs.squeeze(0)
        student_logprobs = student_logprobs.squeeze(0)
        log_ratio = log_ratio.squeeze(0)
        ratio = ratio.squeeze(0)
        reject_prob = reject_prob.squeeze(0)
        occupancy_u = occupancy_u.squeeze(0)
        expected_position = expected_position.squeeze(0)
        remaining_budget_weight = remaining_budget_weight.squeeze(0)
        mixed_weight = mixed_weight.squeeze(0)
        response_mask = response_mask.squeeze(0)

    return ForwardRemainingBudgetResult(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        log_ratio_p_over_q=log_ratio,
        ratio_p_over_q=ratio,
        reject_prob=reject_prob,
        occupancy_u=occupancy_u,
        expected_position=expected_position,
        remaining_budget_weight=remaining_budget_weight,
        mixed_weight=mixed_weight,
        response_mask=response_mask,
    )
