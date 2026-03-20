from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class OfflineTeacherRolloutDataset(Dataset):
    """Offline dataset backed by teacher rollouts dumped from waste_sd.

    Expected fields per sample:
      - prompt
      - response
      - prompt_ids
      - response_ids

    The dataset directly uses stored token IDs instead of re-tokenizing text so that
    offline distillation is aligned with the original teacher rollout.
    """

    REQUIRED_KEYS = ("prompt", "response", "prompt_ids", "response_ids")

    def __init__(
        self,
        data_files: str | list[str] | ListConfig,
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        del processor  # offline token-id dataset does not require multimodal preprocessing

        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = list(data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)
        self.max_prompt_length = int(config.get("max_prompt_length", 1024))
        self.max_response_length = int(config.get("max_response_length", 1024))
        self.truncation = str(config.get("truncation", "error")).lower()
        self.use_shm = bool(config.get("use_shm", False))
        self.cache_dir = str(config.get("cache_dir", "~/.cache/verl/rlhf"))
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"Unsupported truncation mode: {self.truncation!r}")

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else (eos_token_id if eos_token_id is not None else 0))

        self._download()
        self._read_records()

    def _download(self) -> None:
        for idx, data_file in enumerate(self.data_files):
            self.data_files[idx] = copy_to_local(data_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_records(self) -> None:
        records: list[dict[str, Any]] = []
        for data_file in self.data_files:
            suffix = Path(data_file).suffix.lower()
            if suffix == ".jsonl":
                with open(data_file, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Failed to parse JSONL line {line_no} from {data_file}: {exc}") from exc
                        records.append(record)
            elif suffix == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    records.extend(payload)
                else:
                    raise ValueError(f"Expected top-level list in {data_file}, got {type(payload).__name__}")
            elif suffix == ".parquet":
                dataframe = pd.read_parquet(data_file)
                records.extend(dataframe.to_dict(orient="records"))
            else:
                raise ValueError(f"Unsupported offline rollout dataset format: {data_file}")

        total = len(records)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            records = [records[int(i)] for i in indices.tolist()]

        self.records = [self._normalize_record(record) for record in records]

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        missing = [key for key in self.REQUIRED_KEYS if key not in record]
        if missing:
            raise ValueError(
                "Offline teacher-rollout samples must contain keys "
                f"{self.REQUIRED_KEYS}, missing={missing}, available={sorted(record.keys())}"
            )

        prompt_ids = self._coerce_token_ids(record["prompt_ids"], key="prompt_ids")
        response_ids = self._coerce_token_ids(record["response_ids"], key="response_ids")

        normalized = {
            "uid": None if record.get("uid", None) is None else str(record["uid"]),
            "prompt": str(record["prompt"]),
            "response": str(record["response"]),
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
        }
        return normalized

    @staticmethod
    def _coerce_token_ids(values: Any, *, key: str) -> list[int]:
        if values is None:
            raise ValueError(f"Offline rollout sample field {key!r} cannot be None")
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Offline rollout sample field {key!r} must be a list of ints, got {type(values).__name__}")
        token_ids: list[int] = []
        for value in values:
            try:
                token_ids.append(int(value))
            except Exception as exc:
                raise ValueError(f"Offline rollout sample field {key!r} contains non-integer token {value!r}") from exc
        return token_ids

    def __len__(self) -> int:
        return len(self.records)

    def _truncate(self, token_ids: list[int], *, max_length: int, is_prompt: bool) -> list[int]:
        if len(token_ids) <= max_length:
            return token_ids
        if self.truncation == "error":
            raise ValueError(
                f"Offline rollout sample length {len(token_ids)} exceeds max_length={max_length} "
                f"for {'prompt' if is_prompt else 'response'}"
            )
        if self.truncation == "left":
            return token_ids[-max_length:]
        return token_ids[:max_length]

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        prompt_ids_list = self._truncate(record["prompt_ids"], max_length=self.max_prompt_length, is_prompt=True)
        response_ids_list = self._truncate(record["response_ids"], max_length=self.max_response_length, is_prompt=False)

        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
        response_ids = torch.tensor(response_ids_list, dtype=torch.long)

        prompt_len = int(prompt_ids.numel())
        response_len = int(response_ids.numel())

        prompts = torch.full((self.max_prompt_length,), self.pad_token_id, dtype=torch.long)
        prompt_attention_mask = torch.zeros((self.max_prompt_length,), dtype=torch.long)
        if prompt_len > 0:
            prompts[-prompt_len:] = prompt_ids
            prompt_attention_mask[-prompt_len:] = 1

        responses = torch.full((self.max_response_length,), self.pad_token_id, dtype=torch.long)
        response_mask = torch.zeros((self.max_response_length,), dtype=torch.bool)
        if response_len > 0:
            responses[:response_len] = response_ids
            response_mask[:response_len] = True

        attention_mask = torch.cat((prompt_attention_mask, response_mask.to(dtype=torch.long)), dim=0)
        input_ids = torch.cat((prompts, responses), dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)

        sample = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "uid": record["uid"] if record["uid"] is not None else f"offline_{item}",
        }
        return sample
