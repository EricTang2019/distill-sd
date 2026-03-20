#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_rollout_dir(rollout_dir: Path) -> dict[str, float | int | str]:
    files = sorted(rollout_dir.glob("*.jsonl"))

    total_samples = 0
    traced_samples = 0
    total_blocks = 0
    total_accepted_tokens = 0
    total_response_valid_len = 0

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_samples += 1
                obj = json.loads(line)

                if "spec_accept_lens" in obj and isinstance(obj["spec_accept_lens"], list):
                    traced_samples += 1
                    lens = []
                    for x in obj["spec_accept_lens"]:
                        parsed = _safe_int(x)
                        if parsed is not None:
                            lens.append(parsed)
                    total_blocks += len(lens)
                    total_accepted_tokens += sum(lens)

                response_valid_len = _safe_int(obj.get("response_valid_len"))
                if response_valid_len is not None:
                    total_response_valid_len += response_valid_len

    avg_blocks_per_sample = (total_blocks / total_samples) if total_samples > 0 else 0.0
    avg_blocks_per_traced = (total_blocks / traced_samples) if traced_samples > 0 else 0.0
    avg_accepted_tokens_per_sample = (
        total_accepted_tokens / total_samples if total_samples > 0 else 0.0
    )
    avg_response_valid_len_per_sample = (
        total_response_valid_len / total_samples if total_samples > 0 else 0.0
    )

    return {
        "rollout_dir": str(rollout_dir),
        "jsonl_files": len(files),
        "total_samples": total_samples,
        "samples_with_spec_trace": traced_samples,
        "total_blocks": total_blocks,
        "avg_blocks_per_sample": avg_blocks_per_sample,
        "avg_blocks_per_traced_sample": avg_blocks_per_traced,
        "total_accepted_tokens": total_accepted_tokens,
        "total_response_valid_len": total_response_valid_len,
        "avg_accepted_tokens_per_sample": avg_accepted_tokens_per_sample,
        "avg_response_valid_len_per_sample": avg_response_valid_len_per_sample,
    }


def parse_algo_inputs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --input '{item}'. Expected format: name=/path/to/rollout_dir")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --input '{item}'. Empty name or path.")
        parsed.append((name, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize speculative block counts from rollout JSONL files.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Algorithm rollout dir in form: name=/path/to/rollout_dir (repeatable)",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    algo_inputs = parse_algo_inputs(args.input)
    rows: list[dict[str, float | int | str]] = []
    for algo_name, rollout_dir in algo_inputs:
        summary = summarize_rollout_dir(rollout_dir)
        summary["algorithm"] = algo_name
        rows.append(summary)

    print(
        "algorithm\ttotal_samples\tsamples_with_spec_trace\ttotal_blocks\tavg_blocks_per_sample\t"
        "avg_blocks_per_traced_sample\ttotal_accepted_tokens\ttotal_response_valid_len\t"
        "avg_accepted_tokens_per_sample\tavg_response_valid_len_per_sample"
    )
    for row in rows:
        print(
            f"{row['algorithm']}\t{row['total_samples']}\t{row['samples_with_spec_trace']}\t"
            f"{row['total_blocks']}\t{row['avg_blocks_per_sample']:.6f}\t"
            f"{row['avg_blocks_per_traced_sample']:.6f}\t{row['total_accepted_tokens']}\t"
            f"{row['total_response_valid_len']}\t{row['avg_accepted_tokens_per_sample']:.6f}\t"
            f"{row['avg_response_valid_len_per_sample']:.6f}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
