from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.waste_sd.exact_block_count_loss import gather_response_token_logprobs
from recipe.waste_sd.offline_rollout_dataset import OfflineTeacherRolloutDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute student_nll_all_response_tokens_mean for offline rollout data."
    )
    parser.add_argument("--model-path", type=str, required=True, help="HF model/checkpoint path")
    parser.add_argument(
        "--data-file",
        type=str,
        nargs="+",
        required=True,
        help="Offline rollout dataset path(s): .jsonl/.json/.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument(
        "--truncation",
        type=str,
        default="right",
        choices=["error", "left", "right"],
        help="Offline dataset truncation behavior",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
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


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()

    torch_dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    total_nll_sum = 0.0
    total_token_count = 0
    total_samples = 0

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
            logits.div_(args.temperature)
            token_logprobs = gather_response_token_logprobs(logits, responses)
            token_nll = -token_logprobs.float()

        valid_token_nll = token_nll[response_mask]
        total_nll_sum += float(valid_token_nll.sum().item())
        total_token_count += int(valid_token_nll.numel())
        total_samples += int(input_ids.size(0))

    mean_nll = total_nll_sum / total_token_count if total_token_count > 0 else 0.0
    result = {
        "model_path": args.model_path,
        "data_file": args.data_file,
        "num_samples": total_samples,
        "num_response_tokens": total_token_count,
        "student_nll_all_response_tokens_mean": mean_nll,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
