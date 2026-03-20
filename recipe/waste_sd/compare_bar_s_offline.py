from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.exact_block_count_loss import gather_response_token_logprobs
from recipe.waste_sd.offline_rollout_dataset import OfflineTeacherRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare token-level expected_position (bar s_n) between two student checkpoints on offline rollouts."
    )
    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--model-a-path", type=str, required=True, help="Student checkpoint A (e.g. step300)")
    parser.add_argument("--model-b-path", type=str, required=True, help="Student checkpoint B (e.g. step450)")
    parser.add_argument("--data-file", type=str, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--truncation",
        type=str,
        default="right",
        choices=["error", "left", "right"],
    )
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--equal-atol", type=float, default=1e-12)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--histogram-bins", type=int, default=81)
    parser.add_argument("--output-hist-json", type=str, default="")
    parser.add_argument("--output-hist-svg", type=str, default="")
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


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_dataset(tokenizer, args: argparse.Namespace) -> DataLoader:
    dataset_cfg = OmegaConf.create(
        {
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "truncation": args.truncation,
            "shuffle": False,
            "use_shm": False,
        }
    )
    dataset = OfflineTeacherRolloutDataset(
        data_files=args.data_file,
        tokenizer=tokenizer,
        config=dataset_cfg,
        max_samples=args.max_samples,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)


def collect_token_logprobs(
    model_path: str,
    dataloader: DataLoader,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    temperature: float,
    trust_remote_code: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    token_logprobs_all: list[torch.Tensor] = []
    response_mask_all: list[torch.Tensor] = []

    try:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            responses = batch["responses"].to(device)
            response_mask = batch["response_mask"].to(device=device, dtype=torch.bool)
            response_length = responses.size(-1)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = outputs.logits[:, -response_length - 1 : -1, :].float()
                logits.div_(temperature)
                token_logprobs = gather_response_token_logprobs(logits, responses)

            token_logprobs_all.append(token_logprobs.cpu().to(dtype=torch.float64))
            response_mask_all.append(response_mask.cpu())
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(token_logprobs_all, dim=0), torch.cat(response_mask_all, dim=0)


def build_histogram(diff: torch.Tensor, num_bins: int) -> dict[str, list[float] | float | int]:
    if num_bins <= 0:
        raise ValueError(f"histogram-bins must be positive, got {num_bins}")
    if diff.numel() == 0:
        return {
            "num_bins": num_bins,
            "xmin": 0.0,
            "xmax": 0.0,
            "bin_edges": [0.0, 0.0],
            "bin_centers": [0.0],
            "counts": [0],
        }

    xmin = float(diff.min().item())
    xmax = float(diff.max().item())
    if math.isclose(xmin, xmax, abs_tol=1e-15):
        xmin -= 0.5
        xmax += 0.5
    counts = torch.histc(diff.float(), bins=num_bins, min=xmin, max=xmax)
    edges = torch.linspace(xmin, xmax, steps=num_bins + 1, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "num_bins": num_bins,
        "xmin": xmin,
        "xmax": xmax,
        "bin_edges": [float(v) for v in edges.tolist()],
        "bin_centers": [float(v) for v in centers.tolist()],
        "counts": [int(round(v)) for v in counts.tolist()],
    }


def render_histogram_svg(hist: dict[str, list[float] | float | int], title: str) -> str:
    width = 1000
    height = 600
    margin_left = 80
    margin_right = 30
    margin_top = 50
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    counts = hist["counts"]
    centers = hist["bin_centers"]
    xmin = float(hist["xmin"])
    xmax = float(hist["xmax"])
    ymax = max(counts) if counts else 1
    ymax = max(int(ymax), 1)

    def x_to_svg(x: float) -> float:
        if math.isclose(xmin, xmax):
            return margin_left + plot_w / 2.0
        return margin_left + (x - xmin) / (xmax - xmin) * plot_w

    def y_to_svg(y: float) -> float:
        return margin_top + plot_h - (y / ymax) * plot_h

    bars: list[str] = []
    if counts:
        bar_w = plot_w / len(counts)
        for idx, count in enumerate(counts):
            x = margin_left + idx * bar_w
            y = y_to_svg(count)
            h = margin_top + plot_h - y
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 1.0, 1.0):.2f}" height="{max(h, 0.0):.2f}" '
                'fill="#4f8bc9" opacity="0.85" />'
            )

    zero_x = x_to_svg(0.0) if xmin <= 0.0 <= xmax else None
    zero_line = (
        f'<line x1="{zero_x:.2f}" y1="{margin_top:.2f}" x2="{zero_x:.2f}" y2="{margin_top + plot_h:.2f}" '
        'stroke="#c0392b" stroke-width="2" stroke-dasharray="6,4" />'
        if zero_x is not None
        else ""
    )

    xticks = []
    for xval in [xmin, 0.0 if xmin <= 0.0 <= xmax else None, xmax]:
        if xval is None:
            continue
        xsvg = x_to_svg(float(xval))
        xticks.append(
            f'<line x1="{xsvg:.2f}" y1="{margin_top + plot_h:.2f}" x2="{xsvg:.2f}" y2="{margin_top + plot_h + 6:.2f}" '
            'stroke="#333" stroke-width="1" />'
            f'<text x="{xsvg:.2f}" y="{height - margin_bottom + 22:.2f}" text-anchor="middle" '
            'font-size="14" fill="#333">{:.4f}</text>'.format(float(xval))
        )

    yticks = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yval = frac * ymax
        ysvg = y_to_svg(yval)
        yticks.append(
            f'<line x1="{margin_left - 6:.2f}" y1="{ysvg:.2f}" x2="{margin_left:.2f}" y2="{ysvg:.2f}" '
            'stroke="#333" stroke-width="1" />'
            f'<text x="{margin_left - 12:.2f}" y="{ysvg + 5:.2f}" text-anchor="end" '
            'font-size="14" fill="#333">{int(round(yval))}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  <text x="{width / 2:.2f}" y="28" text-anchor="middle" font-size="24" font-family="Arial, sans-serif" fill="#111">{title}</text>
  <line x1="{margin_left:.2f}" y1="{margin_top + plot_h:.2f}" x2="{margin_left + plot_w:.2f}" y2="{margin_top + plot_h:.2f}" stroke="#333" stroke-width="1.5" />
  <line x1="{margin_left:.2f}" y1="{margin_top:.2f}" x2="{margin_left:.2f}" y2="{margin_top + plot_h:.2f}" stroke="#333" stroke-width="1.5" />
  {''.join(yticks)}
  {''.join(xticks)}
  {''.join(bars)}
  {zero_line}
  <text x="{width / 2:.2f}" y="{height - 20:.2f}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111">bar_s(step300) - bar_s(step450)</text>
  <text x="24" y="{height / 2:.2f}" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111" transform="rotate(-90, 24, {height / 2:.2f})">Token count</text>
</svg>
"""


def main() -> None:
    args = parse_args()

    torch_dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_a_path,
        trust_remote_code=args.trust_remote_code,
    )
    dataloader = load_dataset(tokenizer, args)

    teacher_logprobs, response_mask = collect_token_logprobs(
        args.teacher_model_path,
        dataloader,
        device=device,
        torch_dtype=torch_dtype,
        temperature=args.temperature,
        trust_remote_code=args.trust_remote_code,
    )
    student_a_logprobs, response_mask_a = collect_token_logprobs(
        args.model_a_path,
        dataloader,
        device=device,
        torch_dtype=torch_dtype,
        temperature=args.temperature,
        trust_remote_code=args.trust_remote_code,
    )
    student_b_logprobs, response_mask_b = collect_token_logprobs(
        args.model_b_path,
        dataloader,
        device=device,
        torch_dtype=torch_dtype,
        temperature=args.temperature,
        trust_remote_code=args.trust_remote_code,
    )

    if not torch.equal(response_mask, response_mask_a) or not torch.equal(response_mask, response_mask_b):
        raise RuntimeError("response_mask mismatch across teacher/student passes")

    result_a = compute_forward_remaining_budget_weights(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_a_logprobs,
        gamma=args.gamma,
        response_mask=response_mask,
        kl_floor_coef=0.0,
        dp_dtype=torch.float64,
    )
    result_b = compute_forward_remaining_budget_weights(
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_b_logprobs,
        gamma=args.gamma,
        response_mask=response_mask,
        kl_floor_coef=0.0,
        dp_dtype=torch.float64,
    )

    valid_mask = response_mask.bool()
    expected_a = result_a.expected_position[valid_mask]
    expected_b = result_b.expected_position[valid_mask]
    diff = expected_a - expected_b

    greater_count = int((diff > args.equal_atol).sum().item())
    less_count = int((diff < -args.equal_atol).sum().item())
    equal_count = int((diff.abs() <= args.equal_atol).sum().item())
    total_valid_tokens = int(valid_mask.sum().item())

    output = {
        "teacher_model_path": args.teacher_model_path,
        "model_a_path": args.model_a_path,
        "model_b_path": args.model_b_path,
        "data_file": args.data_file,
        "max_samples": args.max_samples,
        "gamma": args.gamma,
        "equal_atol": args.equal_atol,
        "total_valid_tokens": total_valid_tokens,
        "model_a_bar_s_gt_model_b_count": greater_count,
        "model_a_bar_s_lt_model_b_count": less_count,
        "model_a_bar_s_eq_model_b_count": equal_count,
        "model_a_bar_s_gt_model_b_frac": greater_count / total_valid_tokens if total_valid_tokens else 0.0,
        "model_a_bar_s_lt_model_b_frac": less_count / total_valid_tokens if total_valid_tokens else 0.0,
        "model_a_bar_s_eq_model_b_frac": equal_count / total_valid_tokens if total_valid_tokens else 0.0,
        "model_a_bar_s_mean": float(expected_a.mean().item()) if total_valid_tokens else 0.0,
        "model_b_bar_s_mean": float(expected_b.mean().item()) if total_valid_tokens else 0.0,
        "bar_s_diff_mean_a_minus_b": float(diff.mean().item()) if total_valid_tokens else 0.0,
        "bar_s_diff_abs_mean": float(diff.abs().mean().item()) if total_valid_tokens else 0.0,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hist = build_histogram(diff, num_bins=args.histogram_bins)
    if args.output_hist_json:
        hist_path = Path(args.output_hist_json)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_hist_svg:
        svg_path = Path(args.output_hist_svg)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(
            render_histogram_svg(hist, title="Distribution of bar_s(step300) - bar_s(step450)"),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
