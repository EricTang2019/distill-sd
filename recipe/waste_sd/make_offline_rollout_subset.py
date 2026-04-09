from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fixed JSONL subset from offline teacher-rollout data without "
            "changing record contents. This is useful for toy runs that should reuse "
            "the standard waste_sd trainer."
        )
    )
    parser.add_argument(
        "--data-file",
        type=str,
        nargs="+",
        required=True,
        help="Offline rollout dataset path(s): .jsonl/.json/.parquet",
    )
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument(
        "--subset-mode",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to choose the fixed subset from the concatenated input records.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_records(data_files: list[str], *, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for data_file in data_files:
        path = Path(data_file)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Failed to parse JSONL line {line_no} from {path}: {exc}") from exc
                    records.append(record)
                    if max_records is not None and len(records) >= max_records:
                        return records
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected top-level list in {path}, got {type(payload).__name__}")
            remaining = None if max_records is None else max(max_records - len(records), 0)
            records.extend(payload if remaining is None else payload[:remaining])
            if max_records is not None and len(records) >= max_records:
                return records
        elif suffix == ".parquet":
            dataframe = pd.read_parquet(path)
            payload = dataframe.to_dict(orient="records")
            remaining = None if max_records is None else max(max_records - len(records), 0)
            records.extend(payload if remaining is None else payload[:remaining])
            if max_records is not None and len(records) >= max_records:
                return records
        else:
            raise ValueError(f"Unsupported input format: {path}")
    return records


def _select_subset(records: list[dict[str, Any]], *, max_samples: int, subset_mode: str, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {max_samples}")
    total = len(records)
    if max_samples >= total:
        return records
    if subset_mode == "first":
        indices = np.arange(max_samples)
    elif subset_mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=max_samples, replace=False)
        indices.sort()
    else:
        raise ValueError(f"Unsupported subset mode {subset_mode!r}")
    return [records[int(i)] for i in indices.tolist()]


def main() -> None:
    args = parse_args()
    load_limit = args.max_samples if args.subset_mode == "first" else None
    records = _load_records(list(args.data_file), max_records=load_limit)
    subset = _select_subset(
        records,
        max_samples=args.max_samples,
        subset_mode=args.subset_mode,
        seed=args.seed,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in subset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "data_file": list(args.data_file),
        "output_jsonl": str(output_path),
        "subset_mode": args.subset_mode,
        "seed": int(args.seed),
        "num_loaded_records": len(records),
        "num_output_records": len(subset),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
