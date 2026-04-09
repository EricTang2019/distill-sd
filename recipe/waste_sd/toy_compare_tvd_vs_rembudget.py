from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.offline_rollout_dataset import OfflineTeacherRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Toy experiment on a fixed offline teacher-rollout subset. "
            "Train student copies with unweighted TVD vs. rembudget-weighted TVD for a small number of steps, "
            "then evaluate exact pathwise expected blocks and theorem-style RB-TV values on the same subset."
        )
    )
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--student-model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--data-file", type=str, nargs="+", required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["unweighted_tvd", "rembudget_tvd"],
        choices=["unweighted_tvd", "rembudget_tvd", "rembudget_weighted_mean_tvd"],
        help=(
            "Training objectives to compare. "
            "'rembudget_tvd' optimizes mean(w_rem * TVD). "
            "'rembudget_weighted_mean_tvd' matches the detached weighted-mean variant."
        ),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size used for forward/backward.")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="token_mean",
        choices=["token_mean", "sample_mean"],
        help=(
            "How to aggregate per-token objectives for unweighted_tvd and rembudget_tvd. "
            "weighted_mean rembudget ignores this and uses its detached weighted denominator."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--truncation",
        type=str,
        default="right",
        choices=["error", "left", "right"],
    )
    parser.add_argument(
        "--subset-mode",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to choose the fixed toy subset from the offline rollout file(s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--teacher-device", type=str, default="cuda")
    parser.add_argument("--student-device", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[])
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples], dim=0),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in samples], dim=0),
        "position_ids": torch.stack([sample["position_ids"] for sample in samples], dim=0),
        "responses": torch.stack([sample["responses"] for sample in samples], dim=0),
        "response_mask": torch.stack([sample["response_mask"] for sample in samples], dim=0),
        "uid": [str(sample["uid"]) for sample in samples],
    }


@dataclass
class PreparedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    responses: torch.Tensor
    response_mask: torch.Tensor
    uids: list[str]

    @property
    def sample_count(self) -> int:
        return int(self.responses.size(0))

    @property
    def valid_token_count(self) -> int:
        return int(self.response_mask.sum().item())


def materialize_batches(
    *,
    data_files: list[str],
    tokenizer_path: str,
    max_samples: int,
    batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    truncation: str,
    subset_mode: str,
    seed: int,
) -> tuple[list[PreparedBatch], int, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    dataset_cfg = OmegaConf.create(
        {
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "truncation": truncation,
            "shuffle": subset_mode == "random",
            "seed": seed,
            "use_shm": False,
        }
    )
    dataset = OfflineTeacherRolloutDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=dataset_cfg,
        max_samples=max_samples,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    batches: list[PreparedBatch] = []
    total_samples = 0
    total_valid_tokens = 0
    for batch in loader:
        prepared = PreparedBatch(
            input_ids=batch["input_ids"].contiguous(),
            attention_mask=batch["attention_mask"].contiguous(),
            position_ids=batch["position_ids"].contiguous(),
            responses=batch["responses"].contiguous(),
            response_mask=batch["response_mask"].contiguous(),
            uids=list(batch["uid"]),
        )
        batches.append(prepared)
        total_samples += prepared.sample_count
        total_valid_tokens += prepared.valid_token_count
    return batches, total_samples, total_valid_tokens


def gather_response_token_logprobs_via_logsumexp(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if responses.ndim != 2:
        raise ValueError(f"responses must have shape [B, T], got {tuple(responses.shape)}")
    if tuple(logits.shape[:2]) != tuple(responses.shape):
        raise ValueError(f"logits/responses shape mismatch: {tuple(logits.shape[:2])} vs {tuple(responses.shape)}")
    logits_f = logits.float()
    selected = torch.gather(logits_f, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits_f, dim=-1)


def compute_token_tvd(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_prob = torch.softmax(student_logits.float(), dim=-1)
    teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
    return 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)


def compute_exact_pathwise_blocks_from_forward_result(forward_result: Any) -> torch.Tensor:
    """Compute exact expected block count on a fixed teacher rollout from the forward occupancy DP.

    For each token position n, a new block is queried iff:
      - the occupancy state before token n is gamma, or
      - the state is < gamma and the sampled teacher token is rejected.

    Therefore the per-token query probability is

        u_n(gamma) + r_n * sum_{s=0}^{gamma-1} u_n(s),

    and summing this over valid tokens yields the exact pathwise expected number
    of blocks B(x; theta) for the realized teacher rollout x.
    """

    dp_dtype = forward_result.occupancy_u.dtype
    response_mask = forward_result.response_mask.to(dtype=dp_dtype)
    occupancy_u = forward_result.occupancy_u
    reject_prob = forward_result.reject_prob.to(dtype=dp_dtype)
    queried_mass = occupancy_u[..., :-1].sum(dim=-1)
    query_prob = occupancy_u[..., -1] + reject_prob * queried_mass
    return torch.sum(query_prob * response_mask, dim=-1)


def slice_response_logits(full_logits: torch.Tensor, response_length: int, *, temperature: float) -> torch.Tensor:
    if response_length <= 0:
        raise ValueError(f"response_length must be positive, got {response_length}")
    logits = full_logits[:, -response_length - 1 : -1, :].float()
    if temperature != 1.0:
        logits = logits / float(temperature)
    return logits


def run_model_response_logits(
    model: AutoModelForCausalLM,
    batch: PreparedBatch,
    *,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    position_ids = batch.position_ids.to(device)
    response_length = batch.responses.size(-1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return slice_response_logits(outputs.logits, response_length=response_length, temperature=temperature)


def aggregate_token_objective(
    token_values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    aggregation: str,
    total_valid_tokens: int,
    total_samples: int,
) -> torch.Tensor:
    masked_values = token_values * response_mask.to(dtype=token_values.dtype)
    if aggregation == "token_mean":
        denom = max(int(total_valid_tokens), 1)
        return masked_values.sum() / float(denom)
    if aggregation == "sample_mean":
        denom = max(int(total_samples), 1)
        return masked_values.sum(dim=-1).sum() / float(denom)
    raise ValueError(f"Unsupported aggregation mode {aggregation!r}")


@dataclass
class EvalMetrics:
    exact_pathwise_mean_blocks: float
    theorem_rb_tvd_mean_blocks: float
    unweighted_tvd_token_mean: float
    unweighted_tvd_sample_mean: float
    rembudget_tvd_token_mean: float
    rembudget_tvd_sample_mean: float
    rembudget_weighted_mean_tvd: float
    mean_remaining_budget_weight: float
    mean_reject_prob: float
    total_samples: int
    total_valid_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "exact_pathwise_mean_blocks": self.exact_pathwise_mean_blocks,
            "theorem_rb_tvd_mean_blocks": self.theorem_rb_tvd_mean_blocks,
            "unweighted_tvd_token_mean": self.unweighted_tvd_token_mean,
            "unweighted_tvd_sample_mean": self.unweighted_tvd_sample_mean,
            "rembudget_tvd_token_mean": self.rembudget_tvd_token_mean,
            "rembudget_tvd_sample_mean": self.rembudget_tvd_sample_mean,
            "rembudget_weighted_mean_tvd": self.rembudget_weighted_mean_tvd,
            "mean_remaining_budget_weight": self.mean_remaining_budget_weight,
            "mean_reject_prob": self.mean_reject_prob,
            "total_samples": self.total_samples,
            "total_valid_tokens": self.total_valid_tokens,
        }


class WandbToyLogger:
    def __init__(self, run: Any) -> None:
        self.run = run
        self.global_step = 0

    def log_record(self, *, method: str, method_step: int, record: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "global_step": int(self.global_step),
            "method_step": int(method_step),
            "method_name": method,
        }
        for key, value in record.items():
            if key == "step" or value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                payload[f"{method}/{key}"] = value
        self.run.log(payload)
        self.global_step += 1

    def finish(self, *, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]) -> WandbToyLogger | None:
    project = str(args.wandb_project).strip()
    mode = str(args.wandb_mode).strip()
    if not project or mode == "disabled":
        return None

    import wandb

    run = wandb.init(
        project=project,
        name=str(args.wandb_run_name).strip() or None,
        entity=str(args.wandb_entity).strip() or None,
        group=str(args.wandb_group).strip() or None,
        tags=list(args.wandb_tags),
        mode=mode,
        config=config,
        dir=os.environ.get("WANDB_DIR", None),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    return WandbToyLogger(run)


@torch.no_grad()
def evaluate_subset(
    *,
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    batches: list[PreparedBatch],
    gamma: int,
    temperature: float,
    teacher_device: torch.device,
    student_device: torch.device,
    show_progress: bool = False,
    progress_desc: str = "eval",
) -> EvalMetrics:
    student_model.eval()
    teacher_model.eval()
    total_samples = 0
    total_valid_tokens = 0
    exact_pathwise_sum = 0.0
    theorem_blocks_sum = 0.0
    unweighted_tvd_sum = 0.0
    rembudget_tvd_sum = 0.0
    rembudget_weight_sum = 0.0
    reject_prob_sum = 0.0

    batch_iter = batches
    if show_progress:
        batch_iter = tqdm(batches, desc=progress_desc, leave=False, dynamic_ncols=True)

    for batch in batch_iter:
        total_samples += batch.sample_count
        total_valid_tokens += batch.valid_token_count

        teacher_logits = run_model_response_logits(
            teacher_model,
            batch,
            device=teacher_device,
            temperature=temperature,
        )
        teacher_responses = batch.responses.to(teacher_device)
        teacher_token_logprobs = gather_response_token_logprobs_via_logsumexp(teacher_logits, teacher_responses)

        teacher_logits_on_student = teacher_logits.to(student_device)
        responses = batch.responses.to(student_device)
        response_mask = batch.response_mask.to(device=student_device, dtype=torch.bool)
        student_logits = run_model_response_logits(
            student_model,
            batch,
            device=student_device,
            temperature=temperature,
        )
        if teacher_logits_on_student.shape != student_logits.shape:
            raise ValueError(
                f"teacher/student logits shape mismatch: {tuple(teacher_logits_on_student.shape)} vs "
                f"{tuple(student_logits.shape)}"
            )

        token_tvd = compute_token_tvd(student_logits, teacher_logits_on_student)
        student_token_logprobs = gather_response_token_logprobs_via_logsumexp(student_logits, responses)
        forward_result = compute_forward_remaining_budget_weights(
            teacher_logprobs=teacher_token_logprobs.to(device=student_device, dtype=torch.float64),
            student_logprobs=student_token_logprobs.to(dtype=torch.float64),
            gamma=gamma,
            response_mask=response_mask,
            kl_floor_coef=0.0,
        )
        valid_f = response_mask.to(dtype=torch.float32)
        seq_len = valid_f.sum(dim=-1)
        theorem_blocks = seq_len / float(gamma + 1) + (
            float(gamma) / float(gamma + 1)
        ) * torch.sum(forward_result.remaining_budget_weight.to(dtype=torch.float32) * token_tvd * valid_f, dim=-1)
        pathwise_blocks = compute_exact_pathwise_blocks_from_forward_result(forward_result)

        exact_pathwise_sum += float(pathwise_blocks.sum().item())
        theorem_blocks_sum += float(theorem_blocks.sum().item())
        unweighted_tvd_sum += float(torch.sum(token_tvd * valid_f).item())
        rembudget_tvd_sum += float(
            torch.sum(forward_result.remaining_budget_weight.to(dtype=torch.float32) * token_tvd * valid_f).item()
        )
        rembudget_weight_sum += float(
            torch.sum(forward_result.remaining_budget_weight.to(dtype=torch.float32) * valid_f).item()
        )
        reject_prob_sum += float(torch.sum(forward_result.reject_prob.to(dtype=torch.float32) * valid_f).item())

    denom_samples = max(total_samples, 1)
    denom_tokens = max(total_valid_tokens, 1)
    rembudget_weighted_mean_tvd = rembudget_tvd_sum / rembudget_weight_sum if rembudget_weight_sum > 0.0 else 0.0
    return EvalMetrics(
        exact_pathwise_mean_blocks=exact_pathwise_sum / float(denom_samples),
        theorem_rb_tvd_mean_blocks=theorem_blocks_sum / float(denom_samples),
        unweighted_tvd_token_mean=unweighted_tvd_sum / float(denom_tokens),
        unweighted_tvd_sample_mean=unweighted_tvd_sum / float(denom_samples),
        rembudget_tvd_token_mean=rembudget_tvd_sum / float(denom_tokens),
        rembudget_tvd_sample_mean=rembudget_tvd_sum / float(denom_samples),
        rembudget_weighted_mean_tvd=rembudget_weighted_mean_tvd,
        mean_remaining_budget_weight=rembudget_weight_sum / float(denom_tokens),
        mean_reject_prob=reject_prob_sum / float(denom_tokens),
        total_samples=total_samples,
        total_valid_tokens=total_valid_tokens,
    )


def compute_global_rembudget_weight_sum(
    *,
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    batches: list[PreparedBatch],
    gamma: int,
    temperature: float,
    teacher_device: torch.device,
    student_device: torch.device,
    show_progress: bool = False,
) -> float:
    student_model.eval()
    teacher_model.eval()
    total_weight_sum = 0.0
    batch_iter = batches
    if show_progress:
        batch_iter = tqdm(batches, desc="rembudget denom", leave=False, dynamic_ncols=True)

    for batch in batch_iter:
        teacher_logits = run_model_response_logits(
            teacher_model,
            batch,
            device=teacher_device,
            temperature=temperature,
        )
        teacher_token_logprobs = gather_response_token_logprobs_via_logsumexp(
            teacher_logits,
            batch.responses.to(teacher_device),
        )
        student_logits = run_model_response_logits(
            student_model,
            batch,
            device=student_device,
            temperature=temperature,
        )
        responses = batch.responses.to(student_device)
        response_mask = batch.response_mask.to(device=student_device, dtype=torch.bool)
        student_token_logprobs = gather_response_token_logprobs_via_logsumexp(student_logits, responses)
        forward_result = compute_forward_remaining_budget_weights(
            teacher_logprobs=teacher_token_logprobs.to(device=student_device, dtype=torch.float64),
            student_logprobs=student_token_logprobs.to(dtype=torch.float64),
            gamma=gamma,
            response_mask=response_mask,
            kl_floor_coef=0.0,
        )
        total_weight_sum += float(
            torch.sum(forward_result.remaining_budget_weight.to(dtype=torch.float32) * response_mask.float()).item()
        )
    return total_weight_sum


def method_step_objective(
    *,
    method: str,
    token_tvd: torch.Tensor,
    response_mask: torch.Tensor,
    rembudget_weights: torch.Tensor | None,
    total_valid_tokens: int,
    total_samples: int,
    total_weight_sum: float | None,
    aggregation: str,
) -> torch.Tensor:
    valid_f = response_mask.to(dtype=token_tvd.dtype)
    if method == "unweighted_tvd":
        return aggregate_token_objective(
            token_tvd,
            response_mask,
            aggregation=aggregation,
            total_valid_tokens=total_valid_tokens,
            total_samples=total_samples,
        )
    if rembudget_weights is None:
        raise ValueError(f"rembudget_weights are required for method {method!r}")
    weighted_token_tvd = rembudget_weights.to(dtype=token_tvd.dtype) * token_tvd
    if method == "rembudget_tvd":
        return aggregate_token_objective(
            weighted_token_tvd,
            response_mask,
            aggregation=aggregation,
            total_valid_tokens=total_valid_tokens,
            total_samples=total_samples,
        )
    if method == "rembudget_weighted_mean_tvd":
        denom = max(float(total_weight_sum or 0.0), 0.0)
        if denom <= 0.0:
            return weighted_token_tvd.sum() * 0.0
        return torch.sum(weighted_token_tvd * valid_f) / denom
    raise ValueError(f"Unsupported method {method!r}")


def load_causal_lm(
    *,
    model_path: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    return model


def train_method(
    *,
    method: str,
    student_model_path: str,
    teacher_model: AutoModelForCausalLM,
    batches: list[PreparedBatch],
    total_samples: int,
    total_valid_tokens: int,
    steps: int,
    gamma: int,
    lr: float,
    weight_decay: float,
    adam_betas: tuple[float, float],
    adam_eps: float,
    aggregation: str,
    temperature: float,
    teacher_device: torch.device,
    student_device: torch.device,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    disable_tqdm: bool,
    wandb_logger: WandbToyLogger | None,
) -> dict[str, Any]:
    student_model = load_causal_lm(
        model_path=student_model_path,
        device=student_device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    student_model.train()
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=lr,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    history: list[dict[str, Any]] = []
    print(f"[toy_compare_tvd_vs_rembudget] method={method} initial eval...", flush=True)
    initial_eval = evaluate_subset(
        student_model=student_model,
        teacher_model=teacher_model,
        batches=batches,
        gamma=gamma,
        temperature=temperature,
        teacher_device=teacher_device,
        student_device=student_device,
        show_progress=not disable_tqdm,
        progress_desc=f"{method} eval@0",
    )
    history.append({"step": 0, "train_loss": None, **initial_eval.to_dict()})
    if wandb_logger is not None:
        wandb_logger.log_record(method=method, method_step=0, record=history[-1])

    step_iter = range(1, steps + 1)
    if not disable_tqdm:
        step_iter = tqdm(step_iter, desc=f"{method} train", dynamic_ncols=True)

    for step_idx in step_iter:
        student_model.train()
        optimizer.zero_grad(set_to_none=True)

        total_weight_sum = None
        if method == "rembudget_weighted_mean_tvd":
            with torch.no_grad():
                total_weight_sum = compute_global_rembudget_weight_sum(
                    student_model=student_model,
                    teacher_model=teacher_model,
                    batches=batches,
                    gamma=gamma,
                    temperature=temperature,
                    teacher_device=teacher_device,
                    student_device=student_device,
                    show_progress=not disable_tqdm,
                )

        batch_iter = batches
        if not disable_tqdm:
            batch_iter = tqdm(batches, desc=f"{method} step {step_idx}", leave=False, dynamic_ncols=True)

        for batch in batch_iter:
            with torch.no_grad():
                teacher_logits = run_model_response_logits(
                    teacher_model,
                    batch,
                    device=teacher_device,
                    temperature=temperature,
                )
                teacher_token_logprobs = gather_response_token_logprobs_via_logsumexp(
                    teacher_logits,
                    batch.responses.to(teacher_device),
                )

            teacher_logits_on_student = teacher_logits.to(student_device)
            responses = batch.responses.to(student_device)
            response_mask = batch.response_mask.to(device=student_device, dtype=torch.bool)
            student_logits = run_model_response_logits(
                student_model,
                batch,
                device=student_device,
                temperature=temperature,
            )

            rembudget_weights = None
            if method != "unweighted_tvd":
                student_token_logprobs = gather_response_token_logprobs_via_logsumexp(student_logits, responses)
                with torch.no_grad():
                    forward_result = compute_forward_remaining_budget_weights(
                        teacher_logprobs=teacher_token_logprobs.to(device=student_device, dtype=torch.float64),
                        student_logprobs=student_token_logprobs.detach().to(dtype=torch.float64),
                        gamma=gamma,
                        response_mask=response_mask,
                        kl_floor_coef=0.0,
                    )
                rembudget_weights = forward_result.remaining_budget_weight.to(dtype=torch.float32)

            token_tvd = compute_token_tvd(student_logits, teacher_logits_on_student)
            micro_objective = method_step_objective(
                method=method,
                token_tvd=token_tvd,
                response_mask=response_mask,
                rembudget_weights=rembudget_weights,
                total_valid_tokens=total_valid_tokens,
                total_samples=total_samples,
                total_weight_sum=total_weight_sum,
                aggregation=aggregation,
            )
            micro_objective.backward()

        optimizer.step()

        eval_metrics = evaluate_subset(
            student_model=student_model,
            teacher_model=teacher_model,
            batches=batches,
            gamma=gamma,
            temperature=temperature,
            teacher_device=teacher_device,
            student_device=student_device,
            show_progress=not disable_tqdm,
            progress_desc=f"{method} eval@{step_idx}",
        )
        if method == "unweighted_tvd":
            train_loss = (
                eval_metrics.unweighted_tvd_token_mean
                if aggregation == "token_mean"
                else eval_metrics.unweighted_tvd_sample_mean
            )
        elif method == "rembudget_weighted_mean_tvd":
            train_loss = eval_metrics.rembudget_weighted_mean_tvd
        else:
            train_loss = (
                eval_metrics.rembudget_tvd_token_mean
                if aggregation == "token_mean"
                else eval_metrics.rembudget_tvd_sample_mean
            )
        history.append({"step": step_idx, "train_loss": train_loss, **eval_metrics.to_dict()})
        if wandb_logger is not None:
            wandb_logger.log_record(method=method, method_step=step_idx, record=history[-1])
        if not disable_tqdm and hasattr(step_iter, "set_postfix"):
            step_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                exact_blocks=f"{eval_metrics.exact_pathwise_mean_blocks:.4f}",
            )

    final_metrics = history[-1]
    result = {
        "method": method,
        "history": history,
        "final": final_metrics,
    }

    del optimizer
    del student_model
    gc.collect()
    if student_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    teacher_device = torch.device(args.teacher_device)
    student_device = torch.device(args.student_device or args.teacher_device)
    torch_dtype = resolve_dtype(args.dtype)
    tokenizer_path = args.tokenizer_path or args.teacher_model_path

    batches, total_samples, total_valid_tokens = materialize_batches(
        data_files=args.data_file,
        tokenizer_path=tokenizer_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        truncation=args.truncation,
        subset_mode=args.subset_mode,
        seed=args.seed,
    )
    if total_samples <= 0:
        raise ValueError("No samples were loaded for the toy experiment.")

    run_config = {
        "teacher_model_path": args.teacher_model_path,
        "student_model_path": args.student_model_path,
        "tokenizer_path": tokenizer_path,
        "data_file": list(args.data_file),
        "methods": list(args.methods),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "aggregation": str(args.aggregation),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "adam_betas": [float(args.adam_betas[0]), float(args.adam_betas[1])],
        "adam_eps": float(args.adam_eps),
        "gamma": int(args.gamma),
        "max_samples": int(args.max_samples),
        "max_prompt_length": int(args.max_prompt_length),
        "max_response_length": int(args.max_response_length),
        "truncation": str(args.truncation),
        "subset_mode": str(args.subset_mode),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "dtype": str(args.dtype),
        "teacher_device": str(teacher_device),
        "student_device": str(student_device),
        "total_samples": int(total_samples),
        "total_valid_tokens": int(total_valid_tokens),
        "wandb_project": str(args.wandb_project),
        "wandb_run_name": str(args.wandb_run_name),
        "wandb_entity": str(args.wandb_entity),
        "wandb_group": str(args.wandb_group),
        "wandb_mode": str(args.wandb_mode),
        "wandb_tags": list(args.wandb_tags),
    }
    wandb_logger = maybe_init_wandb(args, run_config)

    teacher_model = load_causal_lm(
        model_path=args.teacher_model_path,
        device=teacher_device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    results: list[dict[str, Any]] = []
    exit_code = 0
    try:
        for method in args.methods:
            print(f"[toy_compare_tvd_vs_rembudget] running method={method}", flush=True)
            result = train_method(
                method=method,
                student_model_path=args.student_model_path,
                teacher_model=teacher_model,
                batches=batches,
                total_samples=total_samples,
                total_valid_tokens=total_valid_tokens,
                steps=args.steps,
                gamma=args.gamma,
                lr=args.lr,
                weight_decay=args.weight_decay,
                adam_betas=(float(args.adam_betas[0]), float(args.adam_betas[1])),
                adam_eps=args.adam_eps,
                aggregation=args.aggregation,
                temperature=args.temperature,
                teacher_device=teacher_device,
                student_device=student_device,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
                disable_tqdm=args.disable_tqdm,
                wandb_logger=wandb_logger,
            )
            results.append(result)
            final = result["final"]
            print(
                "[toy_compare_tvd_vs_rembudget] "
                f"method={method} "
                f"exact_blocks={final['exact_pathwise_mean_blocks']:.6f} "
                f"theorem_blocks={final['theorem_rb_tvd_mean_blocks']:.6f} "
                f"unweighted_tvd={final['unweighted_tvd_token_mean']:.6f} "
                f"rembudget_tvd={final['rembudget_tvd_token_mean']:.6f}",
                flush=True,
            )
            if wandb_logger is not None and wandb_logger.run is not None:
                wandb_logger.run.summary[f"{method}/final_exact_pathwise_mean_blocks"] = final[
                    "exact_pathwise_mean_blocks"
                ]
                wandb_logger.run.summary[f"{method}/final_theorem_rb_tvd_mean_blocks"] = final[
                    "theorem_rb_tvd_mean_blocks"
                ]
    finally:
        if sys.exc_info()[0] is not None:
            exit_code = 1
        del teacher_model
        gc.collect()
        if teacher_device.type == "cuda":
            torch.cuda.empty_cache()
        if wandb_logger is not None:
            wandb_logger.finish(exit_code=exit_code)

    payload = {
        "config": run_config,
        "results": results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[toy_compare_tvd_vs_rembudget] wrote {output_path}", flush=True)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
