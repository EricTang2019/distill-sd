from __future__ import annotations

from dataclasses import dataclass

import torch

from recipe.waste_sd.block_count_dp import BlockCountOmegaResult, compute_block_count_omega


@dataclass
class ExactBlockCountWNLLResult:
    teacher_token_logprobs: torch.Tensor
    student_token_logprobs: torch.Tensor
    token_nll: torch.Tensor
    dp_result: BlockCountOmegaResult
    alpha: torch.Tensor
    sample_loss_sum: torch.Tensor
    total_loss_sum: torch.Tensor
    alpha_weight_sum: torch.Tensor
    contributing_samples: int
    loss: torch.Tensor


def gather_response_token_logprobs(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if responses.ndim != 2:
        raise ValueError(f"responses must have shape [B, T], got {tuple(responses.shape)}")
    if tuple(logits.shape[:2]) != tuple(responses.shape):
        raise ValueError(
            f"logits/responses shape mismatch: expected prefix dims {tuple(logits.shape[:2])}, got {tuple(responses.shape)}"
        )
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)


def compute_exact_block_count_wnll_from_logits(
    *,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    responses: torch.Tensor,
    gamma: int,
    response_mask: torch.Tensor | None = None,
    dp_dtype: torch.dtype = torch.float64,
) -> ExactBlockCountWNLLResult:
    """Exact detached weighted-NLL implementation from expectation.tex.

    The objective is the alpha-weighted token mean

        L = sum_{i,n} sg(alpha_{i,n}) * (-log p_theta(x_{i,n} | prefix))
            / sum_{i,n} sg(alpha_{i,n})

    where alpha is computed by the exact one-rollout forward/backward DP.
    """
    teacher_token_logprobs = gather_response_token_logprobs(teacher_logits, responses)
    student_token_logprobs = gather_response_token_logprobs(student_logits, responses)

    if response_mask is None:
        response_mask = torch.ones_like(responses, dtype=torch.bool, device=responses.device)
    else:
        response_mask = response_mask.to(device=responses.device, dtype=torch.bool)
        if tuple(response_mask.shape) != tuple(responses.shape):
            raise ValueError(
                f"response_mask shape mismatch: expected {tuple(responses.shape)}, got {tuple(response_mask.shape)}"
            )

    with torch.no_grad():
        dp_result = compute_block_count_omega(
            teacher_logprobs=teacher_token_logprobs.to(dtype=dp_dtype),
            student_logprobs=student_token_logprobs.detach().to(dtype=dp_dtype),
            gamma=gamma,
            response_mask=response_mask,
            dp_dtype=dp_dtype,
        )

    alpha = dp_result.alpha.to(device=student_logits.device, dtype=torch.float32)
    token_nll = -student_token_logprobs.float()
    mask_f = response_mask.to(dtype=torch.float32)
    detached_alpha = alpha.detach()
    sample_loss_sum = torch.sum(detached_alpha * token_nll * mask_f, dim=-1)
    alpha_weight_sum = torch.sum(alpha * mask_f)

    valid_sample_mask = response_mask.sum(dim=-1) > 0
    contributing_samples = int(valid_sample_mask.sum().item())
    if contributing_samples > 0:
        total_loss_sum = sample_loss_sum[valid_sample_mask].sum()
        if float(alpha_weight_sum.item()) > 0.0:
            loss = total_loss_sum / alpha_weight_sum
        else:
            loss = total_loss_sum * 0.0
    else:
        total_loss_sum = student_logits.new_zeros((), dtype=torch.float32)
        loss = total_loss_sum

    return ExactBlockCountWNLLResult(
        teacher_token_logprobs=teacher_token_logprobs,
        student_token_logprobs=student_token_logprobs,
        token_nll=token_nll,
        dp_result=dp_result,
        alpha=alpha,
        sample_loss_sum=sample_loss_sum,
        total_loss_sum=total_loss_sum,
        alpha_weight_sum=alpha_weight_sum,
        contributing_samples=contributing_samples,
        loss=loss,
    )
