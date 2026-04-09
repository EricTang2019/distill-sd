from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import torch

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.eval_exact_blocks_offline import (
    build_offline_eval_batches,
    load_model,
    resolve_dtype,
)
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw remaining-budget weight distributions on fixed offline teacher rollouts "
            "between two student checkpoints."
        )
    )
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--student-a-model-path", type=str, required=True, help="Usually the before-training model.")
    parser.add_argument("--student-b-model-path", type=str, required=True, help="Usually the after-training model.")
    parser.add_argument("--student-a-label", type=str, default="before")
    parser.add_argument("--student-b-label", type=str, default="after")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--data-file", type=str, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--truncation",
        type=str,
        default="right",
        choices=["error", "left", "right"],
    )
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--teacher-device", type=str, default="cuda")
    parser.add_argument("--student-a-device", type=str, default="")
    parser.add_argument("--student-b-device", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--histogram-bins", type=int, default=51)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-token-jsonl", type=str, default="")
    parser.add_argument("--tvd-vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--tvd-row-chunk-size", type=int, default=8)
    return parser.parse_args()


@dataclass
class StudentBatchMetrics:
    remaining_budget_weight: torch.Tensor
    token_tvd: torch.Tensor
    expected_position: torch.Tensor
    reject_prob: torch.Tensor
    state_probs: torch.Tensor
    student_token_logprobs: torch.Tensor
    theorem_token_contrib: torch.Tensor


def tensor_quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {k: 0.0 for k in ("p00", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100")}
    q = torch.tensor([0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00], dtype=torch.float32)
    quantiles = torch.quantile(values.float(), q)
    names = ["p00", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    return {name: float(val.item()) for name, val in zip(names, quantiles)}


def build_histogram(values: torch.Tensor, num_bins: int, *, xmin: float, xmax: float) -> dict[str, Any]:
    if num_bins <= 0:
        raise ValueError(f"histogram-bins must be positive, got {num_bins}")
    if values.numel() == 0:
        edges = torch.linspace(xmin, xmax, steps=num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {
            "xmin": xmin,
            "xmax": xmax,
            "num_bins": num_bins,
            "bin_edges": edges.tolist(),
            "bin_centers": centers.tolist(),
            "counts": [0.0] * num_bins,
            "density": [0.0] * num_bins,
        }
    counts = torch.histc(values.float(), bins=num_bins, min=xmin, max=xmax)
    edges = torch.linspace(xmin, xmax, steps=num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / max(float(counts.sum().item()), 1.0)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "num_bins": num_bins,
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "counts": counts.tolist(),
        "density": density.tolist(),
    }


def gather_response_token_logprobs_lowmem(
    logits: torch.Tensor,
    responses: torch.Tensor,
    *,
    row_chunk_size: int = 8,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if responses.ndim != 2:
        raise ValueError(f"responses must have shape [B, T], got {tuple(responses.shape)}")
    if tuple(logits.shape[:2]) != tuple(responses.shape):
        raise ValueError(f"logits/responses shape mismatch: {tuple(logits.shape[:2])} vs {tuple(responses.shape)}")
    if row_chunk_size <= 0:
        raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}")
    gathered = torch.empty_like(responses, dtype=torch.float32, device=logits.device)
    batch_size = logits.size(0)
    for row_start in range(0, batch_size, row_chunk_size):
        row_end = min(row_start + row_chunk_size, batch_size)
        logits_chunk = logits[row_start:row_end]
        responses_chunk = responses[row_start:row_end]
        selected = torch.gather(logits_chunk, dim=-1, index=responses_chunk.unsqueeze(-1)).squeeze(-1).to(dtype=torch.float32)
        log_norm = torch.logsumexp(logits_chunk, dim=-1).to(dtype=torch.float32)
        gathered[row_start:row_end] = selected - log_norm
    return gathered


def run_model_response_logits_lowmem(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)  # type: ignore[index]
    attention_mask = batch["attention_mask"].to(device)  # type: ignore[index]
    position_ids = batch["position_ids"].to(device)  # type: ignore[index]
    response_length = batch["responses"].size(-1)  # type: ignore[index]
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = outputs.logits[:, -response_length - 1 : -1, :]
    if temperature != 1.0:
        logits = logits / float(temperature)
    return logits


def summarize_values(
    values: torch.Tensor,
    *,
    histogram_bins: int,
    histogram_xmin: float | None = None,
    histogram_xmax: float | None = None,
) -> dict[str, Any]:
    values_f = values.float()
    xmin = float(histogram_xmin) if histogram_xmin is not None else float(values_f.min().item()) if values_f.numel() > 0 else 0.0
    xmax = float(histogram_xmax) if histogram_xmax is not None else float(values_f.max().item()) if values_f.numel() > 0 else 1.0
    if xmax <= xmin:
        xmax = xmin + 1e-6
    if values_f.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "quantiles": tensor_quantiles(values_f),
            "histogram": build_histogram(values_f, histogram_bins, xmin=xmin, xmax=xmax),
        }
    return {
        "count": int(values_f.numel()),
        "mean": float(values_f.mean().item()),
        "std": float(values_f.std(unbiased=False).item()),
        "quantiles": tensor_quantiles(values_f),
        "histogram": build_histogram(values_f, histogram_bins, xmin=xmin, xmax=xmax),
    }


def compute_token_tvd_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    chunk_size: int,
    row_chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError(f"tvd-vocab-chunk-size must be positive, got {chunk_size}")
    if row_chunk_size <= 0:
        raise ValueError(f"tvd-row-chunk-size must be positive, got {row_chunk_size}")
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            f"student/teacher logits shape mismatch: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    token_tvd = torch.zeros(student_logits.shape[:2], dtype=torch.float32, device=student_logits.device)
    batch_size = student_logits.size(0)
    vocab_size = student_logits.size(-1)
    for row_start in range(0, batch_size, row_chunk_size):
        row_end = min(row_start + row_chunk_size, batch_size)
        student_chunk = student_logits[row_start:row_end]
        teacher_chunk = teacher_logits[row_start:row_end]
        student_log_norm = torch.logsumexp(student_chunk, dim=-1, keepdim=True).to(dtype=torch.float32)
        teacher_log_norm = torch.logsumexp(teacher_chunk, dim=-1, keepdim=True).to(dtype=torch.float32)
        chunk_tvd = torch.zeros(student_chunk.shape[:2], dtype=torch.float32, device=student_logits.device)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            student_prob = torch.exp(student_chunk[..., start:end].to(dtype=torch.float32) - student_log_norm)
            teacher_prob = torch.exp(teacher_chunk[..., start:end].to(dtype=torch.float32) - teacher_log_norm)
            chunk_tvd += 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
        token_tvd[row_start:row_end] = chunk_tvd
    return token_tvd


def compute_student_batch_metrics(
    *,
    teacher_logits: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    student_model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    student_device: torch.device,
    gamma: int,
    temperature: float,
    tvd_vocab_chunk_size: int,
    tvd_row_chunk_size: int,
) -> StudentBatchMetrics:
    responses = batch["responses"].to(student_device)  # type: ignore[index]
    response_mask = batch["response_mask"].to(device=student_device, dtype=torch.bool)  # type: ignore[index]
    student_logits = run_model_response_logits_lowmem(
        student_model,
        batch,
        device=student_device,
        temperature=temperature,
    )
    teacher_logits_on_student = teacher_logits.to(student_device)
    teacher_token_logprobs_on_student = teacher_token_logprobs.to(device=student_device, dtype=torch.float64)
    token_tvd = compute_token_tvd_chunked(
        student_logits,
        teacher_logits_on_student,
        chunk_size=tvd_vocab_chunk_size,
        row_chunk_size=tvd_row_chunk_size,
    )
    student_token_logprobs = gather_response_token_logprobs_lowmem(
        student_logits,
        responses,
        row_chunk_size=tvd_row_chunk_size,
    )
    forward_result = compute_forward_remaining_budget_weights(
        teacher_logprobs=teacher_token_logprobs_on_student,
        student_logprobs=student_token_logprobs.to(dtype=torch.float64),
        gamma=gamma,
        response_mask=response_mask,
        kl_floor_coef=0.0,
    )
    valid_mask = response_mask
    remaining_budget_weight = forward_result.remaining_budget_weight[valid_mask].to(dtype=torch.float32).cpu()
    token_tvd_valid = token_tvd[valid_mask].to(dtype=torch.float32).cpu()
    expected_position = forward_result.expected_position[valid_mask].to(dtype=torch.float32).cpu()
    reject_prob = forward_result.reject_prob[valid_mask].to(dtype=torch.float32).cpu()
    state_probs = forward_result.occupancy_u[valid_mask].to(dtype=torch.float32).cpu()
    student_token_logprobs_valid = student_token_logprobs[valid_mask].to(dtype=torch.float32).cpu()
    theorem_token_contrib = (remaining_budget_weight * token_tvd_valid).to(dtype=torch.float32)
    return StudentBatchMetrics(
        remaining_budget_weight=remaining_budget_weight,
        token_tvd=token_tvd_valid,
        expected_position=expected_position,
        reject_prob=reject_prob,
        state_probs=state_probs,
        student_token_logprobs=student_token_logprobs_valid,
        theorem_token_contrib=theorem_token_contrib,
    )


def flatten_valid_token_metadata(batch: dict[str, torch.Tensor | list[str]]) -> list[dict[str, int | str]]:
    response_mask = batch["response_mask"].to(dtype=torch.bool)  # type: ignore[index]
    responses = batch["responses"]  # type: ignore[index]
    uids = batch["uid"]  # type: ignore[assignment]
    valid_coords = torch.nonzero(response_mask, as_tuple=False)
    metadata: list[dict[str, int | str]] = []
    for sample_idx, token_idx in valid_coords.tolist():
        metadata.append(
            {
                "uid": str(uids[sample_idx]),
                "sample_index": int(sample_idx),
                "token_index": int(token_idx),
                "response_token_id": int(responses[sample_idx, token_idx].item()),
            }
        )
    return metadata


def write_token_trace_records(
    *,
    handle: TextIO,
    metadata: list[dict[str, int | str]],
    teacher_token_logprobs: torch.Tensor,
    student_a_label: str,
    student_b_label: str,
    metrics_a: StudentBatchMetrics,
    metrics_b: StudentBatchMetrics,
) -> None:
    for idx, meta in enumerate(metadata):
        record = {
            **meta,
            "teacher_token_logprob": float(teacher_token_logprobs[idx].item()),
            f"{student_a_label}_student_token_logprob": float(metrics_a.student_token_logprobs[idx].item()),
            f"{student_b_label}_student_token_logprob": float(metrics_b.student_token_logprobs[idx].item()),
            f"{student_a_label}_tvd": float(metrics_a.token_tvd[idx].item()),
            f"{student_b_label}_tvd": float(metrics_b.token_tvd[idx].item()),
            f"{student_a_label}_remaining_budget_weight": float(metrics_a.remaining_budget_weight[idx].item()),
            f"{student_b_label}_remaining_budget_weight": float(metrics_b.remaining_budget_weight[idx].item()),
            f"{student_a_label}_expected_position": float(metrics_a.expected_position[idx].item()),
            f"{student_b_label}_expected_position": float(metrics_b.expected_position[idx].item()),
            f"{student_a_label}_reject_prob": float(metrics_a.reject_prob[idx].item()),
            f"{student_b_label}_reject_prob": float(metrics_b.reject_prob[idx].item()),
            f"{student_a_label}_theorem_token_contrib": float(metrics_a.theorem_token_contrib[idx].item()),
            f"{student_b_label}_theorem_token_contrib": float(metrics_b.theorem_token_contrib[idx].item()),
            f"{student_a_label}_state_probs": metrics_a.state_probs[idx].tolist(),
            f"{student_b_label}_state_probs": metrics_b.state_probs[idx].tolist(),
        }
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return 0.0
    x_f = x.float()
    y_f = y.float()
    x_centered = x_f - x_f.mean()
    y_centered = y_f - y_f.mean()
    denom = torch.sqrt(torch.sum(x_centered.square()) * torch.sum(y_centered.square()))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((torch.sum(x_centered * y_centered) / denom).item())


def compare_students(
    *,
    teacher_model: torch.nn.Module,
    student_a_model: torch.nn.Module,
    student_b_model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor | list[str]]],
    teacher_device: torch.device,
    student_a_device: torch.device,
    student_b_device: torch.device,
    gamma: int,
    temperature: float,
    disable_tqdm: bool,
    tvd_vocab_chunk_size: int,
    tvd_row_chunk_size: int,
    histogram_bins: int,
    student_a_label: str,
    student_b_label: str,
    output_token_jsonl: str = "",
) -> dict[str, Any]:
    weight_a_list: list[torch.Tensor] = []
    weight_b_list: list[torch.Tensor] = []
    tvd_a_list: list[torch.Tensor] = []
    tvd_b_list: list[torch.Tensor] = []
    contrib_a_list: list[torch.Tensor] = []
    contrib_b_list: list[torch.Tensor] = []
    expected_pos_a_list: list[torch.Tensor] = []
    expected_pos_b_list: list[torch.Tensor] = []
    reject_prob_a_list: list[torch.Tensor] = []
    reject_prob_b_list: list[torch.Tensor] = []
    student_logprob_a_list: list[torch.Tensor] = []
    student_logprob_b_list: list[torch.Tensor] = []

    batch_iter: list[dict[str, torch.Tensor | list[str]]] | Any = batches
    if not disable_tqdm:
        from tqdm.auto import tqdm

        batch_iter = tqdm(batches, desc="compare_rembudget_offline", dynamic_ncols=True)

    token_trace_handle: TextIO | None = None
    if output_token_jsonl:
        output_path = Path(output_token_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        token_trace_handle = output_path.open("w", encoding="utf-8")

    with torch.no_grad():
        for batch in batch_iter:
            teacher_logits = run_model_response_logits_lowmem(
                teacher_model,
                batch,
                device=teacher_device,
                temperature=temperature,
            )
            teacher_responses = batch["responses"].to(teacher_device)  # type: ignore[index]
            teacher_token_logprobs = gather_response_token_logprobs_lowmem(
                teacher_logits,
                teacher_responses,
                row_chunk_size=tvd_row_chunk_size,
            )
            metrics_a = compute_student_batch_metrics(
                teacher_logits=teacher_logits,
                teacher_token_logprobs=teacher_token_logprobs,
                student_model=student_a_model,
                batch=batch,
                student_device=student_a_device,
                gamma=gamma,
                temperature=temperature,
                tvd_vocab_chunk_size=tvd_vocab_chunk_size,
                tvd_row_chunk_size=tvd_row_chunk_size,
            )
            metrics_b = compute_student_batch_metrics(
                teacher_logits=teacher_logits,
                teacher_token_logprobs=teacher_token_logprobs,
                student_model=student_b_model,
                batch=batch,
                student_device=student_b_device,
                gamma=gamma,
                temperature=temperature,
                tvd_vocab_chunk_size=tvd_vocab_chunk_size,
                tvd_row_chunk_size=tvd_row_chunk_size,
            )
            weight_a_list.append(metrics_a.remaining_budget_weight)
            weight_b_list.append(metrics_b.remaining_budget_weight)
            tvd_a_list.append(metrics_a.token_tvd)
            tvd_b_list.append(metrics_b.token_tvd)
            contrib_a_list.append(metrics_a.theorem_token_contrib)
            contrib_b_list.append(metrics_b.theorem_token_contrib)
            expected_pos_a_list.append(metrics_a.expected_position)
            expected_pos_b_list.append(metrics_b.expected_position)
            reject_prob_a_list.append(metrics_a.reject_prob)
            reject_prob_b_list.append(metrics_b.reject_prob)
            student_logprob_a_list.append(metrics_a.student_token_logprobs)
            student_logprob_b_list.append(metrics_b.student_token_logprobs)

            if token_trace_handle is not None:
                valid_teacher_logprobs = teacher_token_logprobs[
                    batch["response_mask"].to(device=teacher_device, dtype=torch.bool)  # type: ignore[index]
                ].to(dtype=torch.float32).cpu()
                metadata = flatten_valid_token_metadata(batch)
                write_token_trace_records(
                    handle=token_trace_handle,
                    metadata=metadata,
                    teacher_token_logprobs=valid_teacher_logprobs,
                    student_a_label=student_a_label,
                    student_b_label=student_b_label,
                    metrics_a=metrics_a,
                    metrics_b=metrics_b,
                )

    if token_trace_handle is not None:
        token_trace_handle.close()

    weights_a = torch.cat(weight_a_list, dim=0) if weight_a_list else torch.empty((0,), dtype=torch.float32)
    weights_b = torch.cat(weight_b_list, dim=0) if weight_b_list else torch.empty((0,), dtype=torch.float32)
    tvd_a = torch.cat(tvd_a_list, dim=0) if tvd_a_list else torch.empty((0,), dtype=torch.float32)
    tvd_b = torch.cat(tvd_b_list, dim=0) if tvd_b_list else torch.empty((0,), dtype=torch.float32)
    contrib_a = torch.cat(contrib_a_list, dim=0) if contrib_a_list else torch.empty((0,), dtype=torch.float32)
    contrib_b = torch.cat(contrib_b_list, dim=0) if contrib_b_list else torch.empty((0,), dtype=torch.float32)
    expected_pos_a = (
        torch.cat(expected_pos_a_list, dim=0) if expected_pos_a_list else torch.empty((0,), dtype=torch.float32)
    )
    expected_pos_b = (
        torch.cat(expected_pos_b_list, dim=0) if expected_pos_b_list else torch.empty((0,), dtype=torch.float32)
    )
    reject_prob_a = (
        torch.cat(reject_prob_a_list, dim=0) if reject_prob_a_list else torch.empty((0,), dtype=torch.float32)
    )
    reject_prob_b = (
        torch.cat(reject_prob_b_list, dim=0) if reject_prob_b_list else torch.empty((0,), dtype=torch.float32)
    )
    student_logprob_a = (
        torch.cat(student_logprob_a_list, dim=0) if student_logprob_a_list else torch.empty((0,), dtype=torch.float32)
    )
    student_logprob_b = (
        torch.cat(student_logprob_b_list, dim=0) if student_logprob_b_list else torch.empty((0,), dtype=torch.float32)
    )

    if weights_a.shape != weights_b.shape:
        raise ValueError(
            f"remaining-budget tensor shape mismatch: {tuple(weights_a.shape)} vs {tuple(weights_b.shape)}"
        )

    weight_diff = weights_b - weights_a
    tvd_diff = tvd_b - tvd_a
    theorem_diff = contrib_b - contrib_a
    result = {
        "gamma": int(gamma),
        "temperature": float(temperature),
        "num_valid_tokens": int(weights_a.numel()),
        student_a_label: {
            **summarize_values(weights_a, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0),
            "token_tvd": summarize_values(tvd_a, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0),
            "theorem_token_contrib": summarize_values(
                contrib_a, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0
            ),
            "expected_position": summarize_values(
                expected_pos_a, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=float(gamma)
            ),
            "reject_prob": summarize_values(
                reject_prob_a, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0
            ),
            "student_token_logprob": summarize_values(student_logprob_a, histogram_bins=histogram_bins),
            "corr_weight_tvd": pearson_corr(weights_a, tvd_a),
        },
        student_b_label: {
            **summarize_values(weights_b, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0),
            "token_tvd": summarize_values(tvd_b, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0),
            "theorem_token_contrib": summarize_values(
                contrib_b, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0
            ),
            "expected_position": summarize_values(
                expected_pos_b, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=float(gamma)
            ),
            "reject_prob": summarize_values(
                reject_prob_b, histogram_bins=histogram_bins, histogram_xmin=0.0, histogram_xmax=1.0
            ),
            "student_token_logprob": summarize_values(student_logprob_b, histogram_bins=histogram_bins),
            "corr_weight_tvd": pearson_corr(weights_b, tvd_b),
        },
        "delta": {
            "mean_after_minus_before": float(weight_diff.mean().item()) if weight_diff.numel() > 0 else 0.0,
            "std_after_minus_before": float(weight_diff.std(unbiased=False).item()) if weight_diff.numel() > 0 else 0.0,
            "quantiles_after_minus_before": tensor_quantiles(weight_diff),
            "fraction_after_gt_before": float((weight_diff > 0).float().mean().item()) if weight_diff.numel() > 0 else 0.0,
            "fraction_after_lt_before": float((weight_diff < 0).float().mean().item()) if weight_diff.numel() > 0 else 0.0,
            "histogram_after_minus_before": build_histogram(weight_diff, histogram_bins, xmin=-1.0, xmax=1.0),
            "token_tvd_after_minus_before": {
                "mean": float(tvd_diff.mean().item()) if tvd_diff.numel() > 0 else 0.0,
                "std": float(tvd_diff.std(unbiased=False).item()) if tvd_diff.numel() > 0 else 0.0,
                "quantiles": tensor_quantiles(tvd_diff),
            },
            "theorem_token_contrib_after_minus_before": {
                "mean": float(theorem_diff.mean().item()) if theorem_diff.numel() > 0 else 0.0,
                "std": float(theorem_diff.std(unbiased=False).item()) if theorem_diff.numel() > 0 else 0.0,
                "quantiles": tensor_quantiles(theorem_diff),
            },
        },
        "output_token_jsonl": output_token_jsonl,
    }
    return result


def main() -> None:
    args = parse_args()
    teacher_device = torch.device(args.teacher_device)
    student_a_device = torch.device(args.student_a_device or args.teacher_device)
    student_b_device = torch.device(args.student_b_device or args.teacher_device)
    torch_dtype = resolve_dtype(args.dtype)
    tokenizer_path = args.tokenizer_path or args.teacher_model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    batches = build_offline_eval_batches(
        tokenizer=tokenizer,
        data_files=list(args.data_file),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        truncation=args.truncation,
    )

    teacher_model = load_model(
        model_path=args.teacher_model_path,
        device=teacher_device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    student_a_model = load_model(
        model_path=args.student_a_model_path,
        device=student_a_device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    student_b_model = load_model(
        model_path=args.student_b_model_path,
        device=student_b_device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    for model in (teacher_model, student_a_model, student_b_model):
        for param in model.parameters():
            param.requires_grad_(False)

    result = compare_students(
        teacher_model=teacher_model,
        student_a_model=student_a_model,
        student_b_model=student_b_model,
        batches=batches,
        teacher_device=teacher_device,
        student_a_device=student_a_device,
        student_b_device=student_b_device,
        gamma=args.gamma,
        temperature=args.temperature,
        disable_tqdm=args.disable_tqdm,
        tvd_vocab_chunk_size=args.tvd_vocab_chunk_size,
        tvd_row_chunk_size=args.tvd_row_chunk_size,
        histogram_bins=args.histogram_bins,
        student_a_label=args.student_a_label,
        student_b_label=args.student_b_label,
        output_token_jsonl=args.output_token_jsonl,
    )
    result["data_file"] = list(args.data_file)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
