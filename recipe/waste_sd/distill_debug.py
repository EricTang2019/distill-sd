# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

logger = logging.getLogger(__name__)
DEFAULT_DISTILL_DEBUG_DIR = "/work5/jingwut/On-Policy-Distillation/verl/outputs/waste_sd_debug/default"


def _normalize_accept_lens(spec_accept_lens: Optional[Iterable[int]]) -> list[int]:
    if spec_accept_lens is None:
        return []
    out: list[int] = []
    for x in spec_accept_lens:
        out.append(int(x))
    return out


def _build_block_trace(accept_lens: list[int], valid_tokens: int) -> dict[str, Any]:
    mapped_total = 0
    block_indices: list[int] = []
    within_block_offsets: list[int] = []
    block_lengths: list[int] = []

    for block_idx, block_len in enumerate(accept_lens):
        if block_len <= 0:
            continue
        mapped_total += block_len
        for offset in range(block_len):
            if len(block_indices) >= valid_tokens:
                break
            block_indices.append(block_idx)
            within_block_offsets.append(offset)
            block_lengths.append(block_len)
        if len(block_indices) >= valid_tokens:
            break

    while len(block_indices) < valid_tokens:
        block_indices.append(-1)
        within_block_offsets.append(-1)
        block_lengths.append(-1)

    return {
        "mapped_tokens": mapped_total,
        "alignment_ok": bool(mapped_total == valid_tokens),
        "block_index": block_indices,
        "within_block_offset": within_block_offsets,
        "block_length": block_lengths,
    }


def _token_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_type: str,
    response_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    student_logp = torch.log_softmax(student_logits, dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits, dim=-1)

    if loss_type == "exact_block_count_wnll":
        if response_ids is None:
            raise ValueError("response_ids are required to debug exact_block_count_wnll")
        return -torch.gather(student_logp, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    if loss_type == "teacher_greedy_nll":
        teacher_greedy_ids = teacher_logits.argmax(dim=-1)
        return -torch.gather(student_logp, dim=-1, index=teacher_greedy_ids.unsqueeze(-1)).squeeze(-1)
    if loss_type == "fkl":
        teacher_prob = torch.exp(teacher_logp)
        return torch.sum(teacher_prob * (teacher_logp - student_logp), dim=-1)
    if loss_type == "rkl":
        student_prob = torch.exp(student_logp)
        return torch.sum(student_prob * (student_logp - teacher_logp), dim=-1)
    if loss_type == "tvd":
        student_prob = torch.exp(student_logp)
        teacher_prob = torch.exp(teacher_logp)
        return 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
    raise ValueError(f"Unsupported loss_type {loss_type}")


@dataclass
class DistillDebugConfig:
    enable: bool = False
    dump_dir: str = DEFAULT_DISTILL_DEBUG_DIR
    max_samples_per_step: int = 2
    max_tokens_per_sample: int = 128
    include_topk: bool = True
    logits_topk: int = 5


class DistillDebugRecorder:
    """Optional JSONL dump for validating SD alignment and loss computation."""

    def __init__(self, cfg: Any):
        cfg = cfg or {}
        self.cfg = DistillDebugConfig(
            enable=bool(cfg.get("enable", False)),
            dump_dir=str(cfg.get("dump_dir", DEFAULT_DISTILL_DEBUG_DIR)),
            max_samples_per_step=max(int(cfg.get("max_samples_per_step", 2)), 1),
            max_tokens_per_sample=max(int(cfg.get("max_tokens_per_sample", 128)), 1),
            include_topk=bool(cfg.get("include_topk", True)),
            logits_topk=max(int(cfg.get("logits_topk", 5)), 1),
        )
        self.enabled = self.cfg.enable
        self._sample_count_per_step: dict[int, int] = {}
        self._worker_tag = f"pid{os.getpid()}"
        if self.enabled:
            logger.warning(
                "Waste-SD debug dump enabled: dir=%s, max_samples_per_step=%d, max_tokens_per_sample=%d",
                self.cfg.dump_dir,
                self.cfg.max_samples_per_step,
                self.cfg.max_tokens_per_sample,
            )

    def _try_acquire_slot(self, step: int) -> bool:
        current = self._sample_count_per_step.get(step, 0)
        if current >= self.cfg.max_samples_per_step:
            return False
        self._sample_count_per_step[step] = current + 1
        return True

    def maybe_record_sample(
        self,
        *,
        step: int,
        uid: Any,
        sample_index: int,
        loss_type: str,
        gamma: int,
        strict: bool,
        temperature: float,
        spec_accept_lens: Optional[Iterable[int]],
        response_ids: Optional[torch.Tensor],
        token_weights: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        if not self._try_acquire_slot(step):
            return

        with torch.no_grad():
            valid_tokens = int(token_weights.numel())
            if valid_tokens <= 0:
                return
            take_tokens = min(valid_tokens, self.cfg.max_tokens_per_sample)

            weights_cpu = token_weights[:take_tokens].detach().to(device="cpu", dtype=torch.float32)
            student_cpu = student_logits[:take_tokens].detach().to(device="cpu", dtype=torch.float32)
            teacher_cpu = teacher_logits[:take_tokens].detach().to(device="cpu", dtype=torch.float32)
            response_ids_cpu = (
                response_ids[:take_tokens].detach().to(device="cpu", dtype=torch.long)
                if response_ids is not None
                else None
            )
            response_ids_list = (
                response_ids_cpu.tolist()
                if response_ids_cpu is not None
                else []
            )

            token_div = _token_divergence(student_cpu, teacher_cpu, loss_type=loss_type, response_ids=response_ids_cpu)
            weighted_token_div = token_div * weights_cpu
            weighted_sum = float(weighted_token_div.sum().item())
            weight_sum = float(weights_cpu.sum().item())
            weighted_mean = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            accept_lens = _normalize_accept_lens(spec_accept_lens)
            block_trace = _build_block_trace(accept_lens, valid_tokens=take_tokens)

            payload: dict[str, Any] = {
                "step": int(step),
                "worker": self._worker_tag,
                "uid": None if uid is None else str(uid),
                "sample_index": int(sample_index),
                "loss_type": str(loss_type),
                "gamma": int(gamma),
                "strict": bool(strict),
                "temperature": float(temperature),
                "spec_accept_lens": accept_lens,
                "token_count_total": valid_tokens,
                "token_count_dumped": take_tokens,
                "response_token_ids": response_ids_list,
                "weights": weights_cpu.tolist(),
                "token_divergence": token_div.tolist(),
                "weighted_token_divergence": weighted_token_div.tolist(),
                "weighted_sum": weighted_sum,
                "weight_sum": weight_sum,
                "weighted_mean": weighted_mean,
                "block_trace": block_trace,
            }

            if self.cfg.include_topk:
                k = min(self.cfg.logits_topk, student_cpu.shape[-1])
                student_logp = torch.log_softmax(student_cpu, dim=-1)
                teacher_logp = torch.log_softmax(teacher_cpu, dim=-1)
                student_top_logp, student_top_idx = torch.topk(student_logp, k=k, dim=-1)
                teacher_top_logp, teacher_top_idx = torch.topk(teacher_logp, k=k, dim=-1)
                payload["student_topk_token_ids"] = student_top_idx.tolist()
                payload["student_topk_probs"] = torch.exp(student_top_logp).tolist()
                payload["teacher_topk_token_ids"] = teacher_top_idx.tolist()
                payload["teacher_topk_probs"] = torch.exp(teacher_top_logp).tolist()

        os.makedirs(self.cfg.dump_dir, exist_ok=True)
        if step >= 0:
            file_name = f"step_{step:08d}_{self._worker_tag}.jsonl"
        else:
            file_name = f"step_unknown_{self._worker_tag}.jsonl"
        file_path = os.path.join(self.cfg.dump_dir, file_name)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
