from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch


EXPECTED_VLLM_VERSION = "0.15.1"
GREEDY_TEMPERATURE = 0


@dataclass
class DraftProposalCacheEntry:
    token_ids: torch.Tensor
    probs: torch.Tensor


class StrictDraftProbsWorkerExtension:
    def enable_strict_draft_probs_patch(self) -> bool:
        import vllm

        if vllm.__version__ != EXPECTED_VLLM_VERSION:
            raise RuntimeError(
                "Strict draft_probs patch was validated against vLLM "
                f"{EXPECTED_VLLM_VERSION}, got {vllm.__version__}."
            )
        return apply_strict_draft_probs_patch(self.model_runner)

    def set_strict_draft_temperature(self, value: float | None) -> float | None:
        if value is None:
            self.model_runner._waste_sd_strict_draft_temperature = None
            return None
        draft_temperature = float(value)
        if not math.isfinite(draft_temperature) or draft_temperature < 0:
            raise ValueError(
                "Strict draft temperature must be a finite non-negative float, "
                f"got {draft_temperature!r}."
            )
        self.model_runner._waste_sd_strict_draft_temperature = draft_temperature
        return draft_temperature

    def set_strict_rejection_debug_jsonl(self, raw_path: str | None) -> str | None:
        resolved = _resolve_strict_debug_jsonl_path(raw_path)
        self.model_runner._waste_sd_strict_debug_jsonl = resolved
        return None if resolved is None else str(resolved)

    def get_strict_rejection_debug_jsonl(self) -> str | None:
        resolved = getattr(self.model_runner, "_waste_sd_strict_debug_jsonl", None)
        return None if resolved is None else str(resolved)

    def get_strict_rejection_debug_row_count(self) -> int:
        return int(getattr(self.model_runner, "_waste_sd_strict_debug_row_count", 0))


def _resolve_strict_debug_jsonl_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    suffix = "".join(path.suffixes) or ".jsonl"
    stem = path.name[: -len(suffix)] if suffix else path.name
    resolved = path.with_name(f"{stem}.pid{os.getpid()}{suffix}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _ensure_supported_sampling_metadata(sampling_metadata) -> None:
    if not sampling_metadata.no_penalties:
        raise NotImplementedError(
            "Strict draft_probs patch does not support frequency/presence/repetition penalties."
        )
    if sampling_metadata.allowed_token_ids_mask is not None:
        raise NotImplementedError(
            "Strict draft_probs patch does not support structured-output token masks."
        )
    if sampling_metadata.bad_words_token_ids:
        raise NotImplementedError(
            "Strict draft_probs patch does not support bad_words constraints."
        )
    logitsprocs = sampling_metadata.logitsprocs
    if logitsprocs is not None and (
        getattr(logitsprocs, "argmax_invariant", None) or getattr(logitsprocs, "non_argmax_invariant", None)
    ):
        raise NotImplementedError(
            "Strict draft_probs patch does not support custom logits processors."
        )


def _sample_draft_tokens_with_probs(
    logits: torch.Tensor,
    sampling_metadata,
    *,
    draft_temperature_override: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    _ensure_supported_sampling_metadata(sampling_metadata)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape={tuple(logits.shape)}")

    working_logits = logits.to(torch.float32).clone()
    if draft_temperature_override is None:
        assert sampling_metadata.temperature is not None
        temperature = sampling_metadata.temperature.to(device=working_logits.device, dtype=torch.float32)
    else:
        temperature = torch.full(
            (working_logits.shape[0],),
            float(draft_temperature_override),
            dtype=torch.float32,
            device=working_logits.device,
        )

    is_greedy = temperature == GREEDY_TEMPERATURE
    if bool(is_greedy.all()):
        token_ids = working_logits.argmax(dim=-1)
        probs = torch.nn.functional.one_hot(token_ids, num_classes=working_logits.shape[-1]).to(torch.float32)
        return token_ids, probs
    if bool(is_greedy.any()):
        temperature = torch.where(is_greedy, torch.ones_like(temperature), temperature)

    working_logits.div_(temperature.unsqueeze(-1))

    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    if top_k is not None or top_p is not None:
        working_logits = apply_top_k_top_p(working_logits, top_k, top_p)

    probs = working_logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    for req_idx, generator in sampling_metadata.generators.items():
        q[req_idx].exponential_(generator=generator)
    token_ids = probs.div(q).argmax(dim=-1)
    if bool(is_greedy.any()):
        greedy_token_ids = probs.argmax(dim=-1)
        token_ids = torch.where(is_greedy, greedy_token_ids, token_ids)
        greedy_probs = torch.nn.functional.one_hot(greedy_token_ids, num_classes=probs.shape[-1]).to(torch.float32)
        probs = torch.where(is_greedy.unsqueeze(-1), greedy_probs, probs)
    return token_ids, probs


def _clear_strict_draft_cache(cache: dict[str, DraftProposalCacheEntry], req_ids: list[str] | set[str]) -> None:
    for req_id in req_ids:
        cache.pop(req_id, None)


def _flatten_strict_draft_probs(
    *,
    req_ids: list[str],
    num_draft_tokens: list[int],
    cache: dict[str, DraftProposalCacheEntry],
    expected_draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, list[str]]:
    if len(req_ids) != len(num_draft_tokens):
        raise ValueError(
            f"Request/cache alignment mismatch: len(req_ids)={len(req_ids)} vs len(num_draft_tokens)={len(num_draft_tokens)}"
        )

    if sum(num_draft_tokens) == 0:
        return None, []

    flat_probs: list[torch.Tensor] = []
    consumed_req_ids: list[str] = []
    offset = 0
    for req_id, draft_len in zip(req_ids, num_draft_tokens, strict=False):
        if draft_len <= 0:
            continue
        entry = cache.get(req_id)
        if entry is None:
            raise RuntimeError(f"Missing strict draft_probs cache entry for req_id={req_id}")
        if entry.token_ids.shape[0] < draft_len or entry.probs.shape[0] < draft_len:
            raise RuntimeError(
                f"Cached draft proposal for req_id={req_id} is too short: "
                f"cached={entry.token_ids.shape[0]} requested={draft_len}"
            )
        expected_slice = expected_draft_token_ids[offset : offset + draft_len].to(dtype=entry.token_ids.dtype)
        cached_slice = entry.token_ids[:draft_len]
        if not torch.equal(cached_slice, expected_slice):
            raise RuntimeError(
                f"Strict draft token alignment mismatch for req_id={req_id}: "
                f"cached={cached_slice.tolist()} expected={expected_slice.tolist()}"
            )
        flat_probs.append(entry.probs[:draft_len])
        consumed_req_ids.append(req_id)
        offset += draft_len

    if offset != expected_draft_token_ids.shape[0]:
        raise RuntimeError(
            "Strict draft_probs flatten mismatch: consumed "
            f"{offset} tokens but expected {expected_draft_token_ids.shape[0]}"
        )
    return torch.cat(flat_probs, dim=0).contiguous(), consumed_req_ids


def _single_step_rejection_debug_row(
    *,
    req_id: str,
    draft_token_id: int,
    q_row: torch.Tensor,
    p_row: torch.Tensor,
    output_token_ids: list[int],
) -> dict[str, Any]:
    q_draft = float(q_row[draft_token_id].item())
    if q_draft <= 0:
        raise RuntimeError(
            f"Strict rejection debug expected positive q for sampled token, got q={q_draft} req_id={req_id}"
        )
    p_draft = float(p_row[draft_token_id].item())
    exact_accept_rate = float(torch.minimum(q_row, p_row).sum().item())
    output_token_id = int(output_token_ids[0]) if output_token_ids else None
    return {
        "req_id": req_id,
        "draft_token_id": int(draft_token_id),
        "output_token_id": output_token_id,
        "accepted": output_token_id == int(draft_token_id) if output_token_id is not None else None,
        "q_draft": q_draft,
        "p_draft": p_draft,
        "accept_prob_for_sampled_token": min(1.0, p_draft / q_draft),
        "exact_accept_rate": exact_accept_rate,
        "output_token_ids": [int(token_id) for token_id in output_token_ids],
    }


def _append_debug_rows(debug_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with debug_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_log_strict_rejection_debug(
    self,
    *,
    spec_decode_metadata,
    sampling_metadata,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor | None,
    sampler_output,
) -> None:
    debug_path = getattr(self, "_waste_sd_strict_debug_jsonl", None)
    if debug_path is None:
        return
    if draft_probs is None:
        raise RuntimeError("Strict rejection debug requires draft_probs, but draft_probs is None.")
    if logits is None:
        raise RuntimeError("Strict rejection debug requires logits, but logits is None.")
    if any(draft_len not in (0, 1) for draft_len in spec_decode_metadata.num_draft_tokens):
        raise RuntimeError(
            "Strict rejection debug currently only supports num_draft_tokens in {0, 1} per request."
        )

    from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, apply_sampling_constraints

    raw_target_logits = logits[spec_decode_metadata.target_logits_indices].to(torch.float32).clone()
    target_logits = self.rejection_sampler.apply_logits_processors(
        raw_target_logits,
        sampling_metadata,
        spec_decode_metadata,
    )
    target_logits = apply_sampling_constraints(
        target_logits,
        spec_decode_metadata.cu_num_draft_tokens,
        sampling_metadata,
    )
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

    rows: list[dict[str, Any]] = []
    offset = 0
    req_ids = list(self.input_batch.req_ids)
    for req_idx, (req_id, draft_len) in enumerate(zip(req_ids, spec_decode_metadata.num_draft_tokens, strict=False)):
        if draft_len == 0:
            continue
        draft_token_id = int(spec_decode_metadata.draft_token_ids[offset].item())
        output_row = sampler_output.sampled_token_ids[req_idx]
        valid_output = output_row[output_row != PLACEHOLDER_TOKEN_ID].tolist()
        rows.append(
            _single_step_rejection_debug_row(
                req_id=req_id,
                draft_token_id=draft_token_id,
                q_row=draft_probs[offset],
                p_row=target_probs[offset],
                output_token_ids=valid_output,
            )
        )
        offset += draft_len

    if offset != draft_probs.shape[0]:
        raise RuntimeError(
            f"Strict rejection debug consumed {offset} draft rows but draft_probs has {draft_probs.shape[0]} rows."
        )

    _append_debug_rows(debug_path, rows)
    self._waste_sd_strict_debug_row_count += len(rows)


def _strict_sample(self, logits: torch.Tensor | None, spec_decode_metadata):
    sampling_metadata = self.input_batch.sampling_metadata
    self.input_batch.update_async_output_token_ids()
    if spec_decode_metadata is None:
        return self._waste_sd_orig_sample(logits, spec_decode_metadata)

    if self.use_async_scheduling and self._draft_token_req_ids is not None:
        draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
        self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

    draft_probs = None
    consumed_req_ids: list[str] = []
    if not sampling_metadata.all_greedy:
        draft_probs, consumed_req_ids = _flatten_strict_draft_probs(
            req_ids=list(self.input_batch.req_ids),
            num_draft_tokens=spec_decode_metadata.num_draft_tokens,
            cache=self._waste_sd_strict_draft_cache,
            expected_draft_token_ids=spec_decode_metadata.draft_token_ids,
        )
    sampler_output = self.rejection_sampler(
        spec_decode_metadata,
        draft_probs,
        logits,
        sampling_metadata,
    )
    _maybe_log_strict_rejection_debug(
        self,
        spec_decode_metadata=spec_decode_metadata,
        sampling_metadata=sampling_metadata,
        draft_probs=draft_probs,
        logits=logits,
        sampler_output=sampler_output,
    )
    _clear_strict_draft_cache(self._waste_sd_strict_draft_cache, consumed_req_ids)
    return sampler_output


def _strict_propose_draft_token_ids(self, *args, **kwargs):
    draft_token_ids = self._waste_sd_orig_propose_draft_token_ids(*args, **kwargs)
    if not isinstance(draft_token_ids, torch.Tensor):
        raise TypeError(
            "Strict draft_probs patch expects padded drafter batches and torch.Tensor draft_token_ids."
        )

    drafter = self.drafter
    cached_token_ids = getattr(drafter, "_waste_sd_last_draft_token_ids", None)
    cached_probs = getattr(drafter, "_waste_sd_last_draft_probs", None)
    if cached_token_ids is None or cached_probs is None:
        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata is None and len(args) >= 3:
            sampling_metadata = args[2]
        if sampling_metadata is None or not sampling_metadata.all_greedy:
            raise RuntimeError("Strict draft_probs patch expected cached proposer probs but found none.")
        _clear_strict_draft_cache(self._waste_sd_strict_draft_cache, list(self.input_batch.req_ids))
        return draft_token_ids

    req_ids = list(self.input_batch.req_ids)
    if draft_token_ids.shape[0] != len(req_ids):
        raise RuntimeError(
            f"Strict draft_probs patch expected {len(req_ids)} proposal rows, got {draft_token_ids.shape[0]}"
        )
    if cached_token_ids.shape[:2] != draft_token_ids.shape[:2]:
        raise RuntimeError(
            "Strict draft proposer cache shape mismatch: "
            f"cached={tuple(cached_token_ids.shape)} produced={tuple(draft_token_ids.shape)}"
        )
    if cached_probs.shape[:2] != draft_token_ids.shape[:2]:
        raise RuntimeError(
            "Strict draft proposer prob-cache shape mismatch: "
            f"cached={tuple(cached_probs.shape)} produced={tuple(draft_token_ids.shape)}"
        )
    if not torch.equal(cached_token_ids.to(dtype=draft_token_ids.dtype), draft_token_ids):
        raise RuntimeError("Strict draft proposer token cache mismatch after proposal.")

    for row_idx, req_id in enumerate(req_ids):
        self._waste_sd_strict_draft_cache[req_id] = DraftProposalCacheEntry(
            token_ids=cached_token_ids[row_idx].to(dtype=torch.int32).contiguous().clone(),
            probs=cached_probs[row_idx].to(dtype=torch.float32).contiguous().clone(),
        )

    drafter._waste_sd_last_draft_token_ids = None
    drafter._waste_sd_last_draft_probs = None
    return draft_token_ids


def _strict_update_states(self, scheduler_output):
    _clear_strict_draft_cache(self._waste_sd_strict_draft_cache, scheduler_output.finished_req_ids)
    _clear_strict_draft_cache(self._waste_sd_strict_draft_cache, scheduler_output.preempted_req_ids)
    return self._waste_sd_orig_update_states(scheduler_output)


def _strict_draft_propose(
    self,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    last_token_indices: torch.Tensor | None,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    from vllm.forward_context import set_forward_context
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
    from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID

    batch_size = common_attn_metadata.batch_size()

    if self.method == "eagle3":
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size

    num_tokens, last_token_indices, common_attn_metadata = self.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        last_token_indices=last_token_indices,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )

    assert self.runner is not None

    if self.attn_metadata_builder is None:
        attn_metadata_builder = self._get_attention_metadata_builder()
    else:
        attn_metadata_builder = self.attn_metadata_builder

    attn_metadata = attn_metadata_builder.build_for_drafting(
        common_attn_metadata=common_attn_metadata, draft_index=0
    )
    if self.draft_indexer_metadata_builder:
        draft_indexer_metadata = self.draft_indexer_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0,
        )
    else:
        draft_indexer_metadata = None
    per_layer_attn_metadata = {}
    for layer_name in self.attn_layer_names:
        per_layer_attn_metadata[layer_name] = attn_metadata
    for layer_name in self.indexer_layer_names:
        assert draft_indexer_metadata is not None
        per_layer_attn_metadata[layer_name] = draft_indexer_metadata

    num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
        num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens
    )
    cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(num_tokens_dp_padded)
    num_input_tokens = batch_desc.num_tokens
    if num_tokens_across_dp is not None:
        num_tokens_across_dp[self.dp_rank] = num_input_tokens

    if self.pass_hidden_states_to_model:
        self.hidden_states[:num_tokens] = target_hidden_states

    if self.supports_mm_inputs:
        mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
        self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
            self.input_ids[:num_tokens],
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        input_ids = None
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
    else:
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

    model_kwargs = {
        "input_ids": input_ids,
        "positions": self._get_positions(num_input_tokens),
        "inputs_embeds": inputs_embeds,
    }
    if self.pass_hidden_states_to_model:
        model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

    with set_forward_context(
        per_layer_attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=self._get_slot_mapping(num_input_tokens, common_attn_metadata.slot_mapping),
    ):
        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

    sample_hidden_states = last_hidden_states[last_token_indices]
    logits = self.model.compute_logits(sample_hidden_states)
    draft_token_ids, draft_probs = _sample_draft_tokens_with_probs(
        logits,
        sampling_metadata,
        draft_temperature_override=getattr(self.runner, "_waste_sd_strict_draft_temperature", None),
    )

    if self.num_speculative_tokens == 1:
        self._waste_sd_last_draft_token_ids = draft_token_ids.view(-1, 1).to(dtype=torch.int32).contiguous()
        self._waste_sd_last_draft_probs = draft_probs.view(batch_size, 1, -1).to(dtype=torch.float32).contiguous()
        return draft_token_ids.view(-1, 1)

    if self.uses_mrope:
        positions = self.positions[:, last_token_indices]
    else:
        positions = self.positions[last_token_indices]
    if self.method in (
        "deepseek_mtp",
        "ernie_mtp",
        "longcat_flash_mtp",
        "pangu_ultra_moe_mtp",
    ):
        hidden_states = self.hidden_states[last_token_indices]
    else:
        hidden_states = hidden_states[last_token_indices]

    if isinstance(attn_metadata, TreeAttentionMetadata):
        raise NotImplementedError("Strict draft_probs patch does not support tree-attention drafting.")

    if self.allowed_attn_types is not None and not isinstance(attn_metadata, self.allowed_attn_types):
        raise ValueError(
            f"Unsupported attention metadata type for speculative decoding with num_speculative_tokens > 1: "
            f"{type(attn_metadata)}. Supported types are: {self.allowed_attn_types}"
        )

    draft_token_ids_list = [draft_token_ids]
    draft_probs_list = [draft_probs]

    batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
        num_tokens_unpadded=batch_size, num_tokens_padded=batch_size
    )
    cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(batch_size_dp_padded)
    input_batch_size = batch_desc.num_tokens
    if batch_size_across_dp is not None:
        batch_size_across_dp[self.dp_rank] = input_batch_size

    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()

    if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata._num_computed_tokens_cpu = None

    for token_index in range(self.num_speculative_tokens - 1):
        input_ids = draft_token_ids_list[-1].int()
        if self.uses_mrope:
            positions += 1
            exceeds_max_model_len = positions[0] >= self.max_model_len
            clamped_positions = torch.where(
                exceeds_max_model_len.unsqueeze(0),
                torch.zeros_like(positions),
                positions,
            )
        else:
            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        common_attn_metadata.seq_lens += 1
        common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
        common_attn_metadata.max_seq_len = min(common_attn_metadata.max_seq_len + 1, self.max_model_len)
        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1

        block_size = attn_metadata_builder.kv_cache_spec.block_size
        if self.uses_mrope:
            block_numbers = clamped_positions[0] // block_size
        else:
            block_numbers = clamped_positions // block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(-1)
        if self.uses_mrope:
            common_attn_metadata.slot_mapping = block_ids * block_size + clamped_positions[0] % block_size
        else:
            common_attn_metadata.slot_mapping = block_ids * block_size + clamped_positions % block_size
        common_attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=token_index + 1
        )
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        self.input_ids[:batch_size] = input_ids
        self._set_positions(batch_size, clamped_positions)
        self.hidden_states[:batch_size] = hidden_states
        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
            input_ids = None
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(input_batch_size),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=input_batch_size,
            num_tokens_across_dp=batch_size_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(input_batch_size, common_attn_metadata.slot_mapping),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        hidden_states = hidden_states[:batch_size]
        logits = self.model.compute_logits(last_hidden_states[:batch_size])
        draft_token_ids, draft_probs = _sample_draft_tokens_with_probs(
            logits,
            sampling_metadata,
            draft_temperature_override=getattr(self.runner, "_waste_sd_strict_draft_temperature", None),
        )
        draft_token_ids_list.append(draft_token_ids)
        draft_probs_list.append(draft_probs)

    draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
    self._waste_sd_last_draft_token_ids = draft_token_ids.to(dtype=torch.int32).contiguous()
    self._waste_sd_last_draft_probs = torch.stack(draft_probs_list, dim=1).to(dtype=torch.float32).contiguous()
    return draft_token_ids


def apply_strict_draft_probs_patch(model_runner) -> bool:
    spec_config = getattr(model_runner, "speculative_config", None)
    if spec_config is None or getattr(spec_config, "method", None) != "draft_model":
        raise RuntimeError("Strict draft_probs patch only supports speculative_config.method='draft_model'.")
    if getattr(spec_config, "disable_padded_drafter_batch", False):
        raise RuntimeError("Strict draft_probs patch requires padded drafter batches.")
    if getattr(model_runner, "use_async_scheduling", False):
        raise RuntimeError("Strict draft_probs patch only supports synchronous scheduling.")
    if getattr(model_runner, "_waste_sd_strict_draft_probs_patch_applied", False):
        return False
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        raise RuntimeError("Strict draft_probs patch expected a draft proposer on model_runner.")

    model_runner._waste_sd_strict_draft_cache = {}
    model_runner._waste_sd_strict_draft_temperature = None
    model_runner._waste_sd_strict_debug_jsonl = _resolve_strict_debug_jsonl_path(
        os.environ.get("WASTE_SD_STRICT_REJECTION_DEBUG_JSONL")
    )
    model_runner._waste_sd_strict_debug_row_count = 0
    model_runner._waste_sd_orig_sample = model_runner._sample
    model_runner._waste_sd_orig_propose_draft_token_ids = model_runner.propose_draft_token_ids
    model_runner._waste_sd_orig_update_states = model_runner._update_states
    model_runner._sample = MethodType(_strict_sample, model_runner)
    model_runner.propose_draft_token_ids = MethodType(_strict_propose_draft_token_ids, model_runner)
    model_runner._update_states = MethodType(_strict_update_states, model_runner)

    drafter._waste_sd_last_draft_token_ids = None
    drafter._waste_sd_last_draft_probs = None
    drafter.propose = MethodType(_strict_draft_propose, drafter)

    model_runner._waste_sd_strict_draft_probs_patch_applied = True
    return True
