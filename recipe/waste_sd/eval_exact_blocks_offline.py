from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.offline_rollout_dataset import OfflineTeacherRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact pathwise expected number of blocks and theorem-style "
            "RB-TV blocks on fixed offline teacher rollouts."
        )
    )
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--student-model-path", type=str, required=True)
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
    parser.add_argument("--student-device", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples], dim=0),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in samples], dim=0),
        "position_ids": torch.stack([sample["position_ids"] for sample in samples], dim=0),
        "responses": torch.stack([sample["responses"] for sample in samples], dim=0),
        "response_mask": torch.stack([sample["response_mask"] for sample in samples], dim=0),
        "uid": [str(sample["uid"]) for sample in samples],
    }


def build_offline_eval_batches(
    *,
    tokenizer: AutoTokenizer,
    data_files: list[str],
    batch_size: int,
    max_samples: int,
    max_prompt_length: int,
    max_response_length: int,
    truncation: str,
) -> list[dict[str, torch.Tensor | list[str]]]:
    dataset_cfg = OmegaConf.create(
        {
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "truncation": truncation,
            "shuffle": False,
            "use_shm": False,
        }
    )
    dataset = OfflineTeacherRolloutDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=dataset_cfg,
        max_samples=max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return list(dataloader)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


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
    return slice_response_logits(outputs.logits, response_length=response_length, temperature=temperature)


def load_model(
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
    model.eval()
    model.config.use_cache = False
    return model


def evaluate_exact_blocks_with_models(
    *,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor | list[str]]],
    teacher_device: torch.device,
    student_device: torch.device,
    gamma: int,
    temperature: float,
    disable_tqdm: bool,
    progress_desc: str = "eval_exact_blocks_offline",
) -> dict[str, float | int]:
    total_samples = 0
    total_valid_tokens = 0
    exact_pathwise_sum = 0.0
    theorem_blocks_sum = 0.0
    unweighted_tvd_sum = 0.0
    rembudget_tvd_sum = 0.0
    rembudget_weight_sum = 0.0
    reject_prob_sum = 0.0

    batch_iter: list[dict[str, torch.Tensor | list[str]]] | Any = batches
    if not disable_tqdm:
        batch_iter = tqdm(batches, desc=progress_desc, dynamic_ncols=True)

    with torch.no_grad():
        for batch in batch_iter:
            response_mask = batch["response_mask"]
            total_samples += int(batch["responses"].size(0))  # type: ignore[index]
            total_valid_tokens += int(response_mask.sum().item())  # type: ignore[union-attr]

            teacher_logits = run_model_response_logits(
                teacher_model,
                batch,
                device=teacher_device,
                temperature=temperature,
            )
            teacher_responses = batch["responses"].to(teacher_device)  # type: ignore[index]
            teacher_token_logprobs = gather_response_token_logprobs_via_logsumexp(teacher_logits, teacher_responses)

            teacher_logits_on_student = teacher_logits.to(student_device)
            responses = batch["responses"].to(student_device)  # type: ignore[index]
            response_mask = batch["response_mask"].to(device=student_device, dtype=torch.bool)  # type: ignore[index]
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
            exact_blocks = compute_exact_pathwise_blocks_from_forward_result(forward_result)

            exact_pathwise_sum += float(exact_blocks.sum().item())
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
    return {
        "num_samples": total_samples,
        "num_response_tokens": total_valid_tokens,
        "gamma": int(gamma),
        "temperature": float(temperature),
        "exact_pathwise_mean_blocks": exact_pathwise_sum / float(denom_samples),
        "theorem_rb_tvd_mean_blocks": theorem_blocks_sum / float(denom_samples),
        "unweighted_tvd_token_mean": unweighted_tvd_sum / float(denom_tokens),
        "unweighted_tvd_sample_mean": unweighted_tvd_sum / float(denom_samples),
        "rembudget_tvd_token_mean": rembudget_tvd_sum / float(denom_tokens),
        "rembudget_tvd_sample_mean": rembudget_tvd_sum / float(denom_samples),
        "rembudget_weighted_mean_tvd": rembudget_weighted_mean_tvd,
        "mean_remaining_budget_weight": rembudget_weight_sum / float(denom_tokens),
        "mean_reject_prob": reject_prob_sum / float(denom_tokens),
    }


def main() -> None:
    args = parse_args()
    teacher_device = torch.device(args.teacher_device)
    student_device = torch.device(args.student_device or args.teacher_device)
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
    student_model = load_model(
        model_path=args.student_model_path,
        device=student_device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    for param in student_model.parameters():
        param.requires_grad_(False)

    try:
        result = evaluate_exact_blocks_with_models(
            teacher_model=teacher_model,
            student_model=student_model,
            batches=batches,
            teacher_device=teacher_device,
            student_device=student_device,
            gamma=args.gamma,
            temperature=args.temperature,
            disable_tqdm=args.disable_tqdm,
        )
    finally:
        del teacher_model
        del student_model
        if teacher_device.type == "cuda":
            torch.cuda.empty_cache()
        if student_device.type == "cuda" and student_device != teacher_device:
            torch.cuda.empty_cache()

    result = {
        "teacher_model_path": args.teacher_model_path,
        "student_model_path": args.student_model_path,
        "data_file": list(args.data_file),
        **result,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
