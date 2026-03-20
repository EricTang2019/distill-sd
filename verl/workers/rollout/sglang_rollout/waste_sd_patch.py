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

import logging
import os
import types
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_ENABLE_ENV = "VERL_SGLANG_WASTE_SD_PATCH"
_STRICT_ENV = "VERL_SGLANG_WASTE_SD_STRICT"
_DEBUG_SIG_ENV = "VERL_SGLANG_SD_DEBUG_SIG"
_VERIFY_COUNTER_ENV = "VERL_SGLANG_SD_VERIFY_COUNTER"
_PATCH_MARKER = "_verl_waste_sd_patch_applied"


def _env_enabled() -> bool:
    value = os.getenv(_ENABLE_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _strict_mode() -> bool:
    value = os.getenv(_STRICT_ENV, "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _debug_sig_enabled() -> bool:
    value = os.getenv(_DEBUG_SIG_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _verify_counter_enabled() -> bool:
    value = os.getenv(_VERIFY_COUNTER_ENV, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_model_path(model_runner: Any) -> str:
    model_cfg = getattr(model_runner, "model_config", None)
    for key in ("model_path", "path", "hf_config_path"):
        value = getattr(model_cfg, key, None)
        if isinstance(value, str) and value:
            return value
    return "<unknown>"


def _model_param_signature(model_runner: Any) -> tuple[float, float, int]:
    model = getattr(model_runner, "model", None)
    if model is None:
        return (float("nan"), float("nan"), 0)
    first_param = None
    for p in model.parameters():
        first_param = p
        break
    if first_param is None:
        return (float("nan"), float("nan"), 0)
    with torch.no_grad():
        flat = first_param.detach().view(-1)
        n = int(min(flat.numel(), 64))
        if n <= 0:
            return (float("nan"), float("nan"), 0)
        head = flat[:n].float()
        return (float(head.sum().item()), float(head.mean().item()), n)


def _capture_spec_accept_trace(self: Any, result: Any, batch: Any, strict: bool) -> None:
    accept_lens = getattr(result, "accept_lens", None)
    if accept_lens is None:
        return

    if isinstance(accept_lens, torch.Tensor):
        accept_lens_list = accept_lens.tolist()
    else:
        accept_lens_list = list(accept_lens)

    if len(accept_lens_list) != len(batch.reqs):
        msg = (
            "Waste-SD patch alignment error: len(accept_lens) != len(batch.reqs), "
            f"got {len(accept_lens_list)} vs {len(batch.reqs)}."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s Skip this capture step.", msg)
        return

    for req_idx, req in enumerate(batch.reqs):
        accepted_len = int(accept_lens_list[req_idx])
        if accepted_len <= 0:
            continue

        if not hasattr(req, "_verl_spec_accept_lens_history"):
            req._verl_spec_accept_lens_history = []
        req._verl_spec_accept_lens_history.append(accepted_len)


def _append_spec_accept_trace_from_verify(
    batch: Any, accept_length_per_req_cpu: Any, strict: bool, *, source: str
) -> None:
    """Append per-block emitted lengths (accepted_draft + 1 teacher) to request history."""
    reqs = list(getattr(batch, "reqs", []))
    accept_list = [int(x) for x in accept_length_per_req_cpu]
    if len(accept_list) != len(reqs):
        msg = (
            "Waste-SD patch alignment error in "
            f"{source}: len(accept_length_per_req_cpu)={len(accept_list)} vs len(reqs)={len(reqs)}."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s Skip this capture step.", msg)
        return

    for req_idx, req in enumerate(reqs):
        emitted_len = int(accept_list[req_idx]) + 1
        if emitted_len <= 0:
            continue
        if not hasattr(req, "_verl_spec_accept_lens_history"):
            req._verl_spec_accept_lens_history = []
        req._verl_spec_accept_lens_history.append(emitted_len)


def _extract_reqs_from_forward_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """Best-effort extraction of req objects from target forward call args."""
    candidates = []
    if args:
        candidates.append(args[0])
    if "batch" in kwargs:
        candidates.append(kwargs["batch"])
    for candidate in candidates:
        reqs = getattr(candidate, "reqs", None)
        if reqs is not None:
            return list(reqs)
    return []


def apply_waste_sd_patch() -> None:
    """Patch SGLang internals to expose strict SD acceptance trace only."""
    if not _env_enabled():
        return

    try:
        from sglang.srt.managers.io_struct import BatchTokenIDOutput
        from sglang.srt.managers.scheduler_output_processor_mixin import SchedulerOutputProcessorMixin
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
    except Exception as e:
        logger.warning("Skip waste-sd patch: failed to import required sglang modules: %s", e)
        return

    # Optional hooks. Missing optional modules should not disable the whole patch.
    DetokenizerManager = None
    multi_tokenizer_mixin_mod = None
    EAGLEWorker = None
    try:
        from sglang.srt.managers.detokenizer_manager import DetokenizerManager as _DetokenizerManager

        DetokenizerManager = _DetokenizerManager
    except Exception as e:
        logger.warning("Waste-SD patch: detokenizer hook unavailable: %s", e)
    try:
        from sglang.srt.managers import multi_tokenizer_mixin as _multi_tokenizer_mixin_mod

        multi_tokenizer_mixin_mod = _multi_tokenizer_mixin_mod
    except Exception as e:
        logger.warning("Waste-SD patch: multi-tokenizer hook unavailable: %s", e)
    try:
        from sglang.srt.speculative.eagle_worker import EAGLEWorker as _EAGLEWorker

        EAGLEWorker = _EAGLEWorker
    except Exception as e:
        logger.warning("Waste-SD patch: eagle hook unavailable: %s", e)

    if getattr(SchedulerOutputProcessorMixin, _PATCH_MARKER, False):
        return

    strict = _strict_mode()
    # Keep runtime verify counters always enabled once Waste-SD patch is active.
    # This is diagnostics-only metadata/logging and does not change decoding logic.
    verify_counter_enabled = True
    original_resolve = SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids
    original_prefill = SchedulerOutputProcessorMixin.process_batch_result_prefill
    original_decode = SchedulerOutputProcessorMixin.process_batch_result_decode
    original_stream = SchedulerOutputProcessorMixin.stream_output_generation
    original_handle_batch_output = TokenizerManager._handle_batch_output
    original_detok_handle_batch_token_id_out = (
        DetokenizerManager.handle_batch_token_id_out if DetokenizerManager is not None else None
    )
    original_eagle_verify = EAGLEWorker.verify if EAGLEWorker is not None else None
    original_eagle_forward_batch_generation = (
        EAGLEWorker.forward_batch_generation if EAGLEWorker is not None else None
    )
    original_multi_handle_output_by_index = (
        getattr(multi_tokenizer_mixin_mod, "_handle_output_by_index", None)
        if multi_tokenizer_mixin_mod is not None
        else None
    )

    def wrapped_resolve(self: Any, result: Any, batch: Any):
        predict_tokens = original_resolve(self, result, batch)
        try:
            _capture_spec_accept_trace(self, result, batch, strict=strict)
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in _resolve_spec_overlap_token_ids: %s", e)
            if strict:
                raise
        return predict_tokens

    def wrapped_decode(self: Any, batch: Any, result: Any):
        # For non-v2 speculative paths (e.g., some standalone configurations),
        # `_resolve_spec_overlap_token_ids` is not called, so capture `accept_lens`
        # here before decode processing.
        try:
            spec_algorithm = getattr(batch, "spec_algorithm", None)
            is_spec = spec_algorithm is not None and (not spec_algorithm.is_none())
            if is_spec and not getattr(batch, "is_v2_eagle", False):
                # EAGLE/STANDALONE non-v2 path is captured in `wrapped_eagle_verify`.
                # Skip `result.accept_lens` capture here to avoid duplicate block trace entries.
                if hasattr(spec_algorithm, "is_standalone") and spec_algorithm.is_standalone():
                    return original_decode(self, batch, result)
                if hasattr(spec_algorithm, "is_eagle") and spec_algorithm.is_eagle():
                    return original_decode(self, batch, result)
                # Do not crash the SGLang scheduler process here. If accept_lens
                # cannot be captured on this path, keep metadata empty and let the
                # trainer-side strict alignment check fail with a clear error.
                if strict and getattr(result, "accept_lens", None) is None and original_eagle_verify is None:
                    logger.error(
                        "Waste-SD strict mode: non-v2 speculative path has no accept_lens "
                        "and eagle verify hook is unavailable; leaving spec_accept_lens empty."
                    )
                _capture_spec_accept_trace(self, result, batch, strict=False)
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in process_batch_result_decode hook: %s", e)
            if strict:
                raise
        return original_decode(self, batch, result)

    def wrapped_prefill(self: Any, batch: Any, result: Any):
        reqs = list(getattr(batch, "reqs", []))
        before_output_lens = {id(req): len(getattr(req, "output_ids", [])) for req in reqs}

        out = original_prefill(self, batch, result)

        try:
            spec_algorithm = getattr(batch, "spec_algorithm", None)
            is_spec = spec_algorithm is not None and (not spec_algorithm.is_none())
            if not is_spec or getattr(batch, "is_v2_eagle", False):
                return out

            is_standalone = hasattr(spec_algorithm, "is_standalone") and spec_algorithm.is_standalone()
            is_eagle = hasattr(spec_algorithm, "is_eagle") and spec_algorithm.is_eagle()
            if not (is_standalone or is_eagle):
                return out

            # Non-v2 speculative prefill can emit the first target token directly in prefill
            # (without verify/accept_lens), so record it as target-only block(s) of length 1.
            for req in reqs:
                before_len = before_output_lens.get(id(req), 0)
                after_len = len(getattr(req, "output_ids", []))
                emitted = max(after_len - before_len, 0)
                if emitted <= 0:
                    continue
                if not hasattr(req, "_verl_spec_accept_lens_history"):
                    req._verl_spec_accept_lens_history = []
                req._verl_spec_accept_lens_history.extend([1] * emitted)
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in process_batch_result_prefill hook: %s", e)
            if strict:
                raise
        return out

    def wrapped_stream(self: Any, reqs: list[Any], return_logprob: bool, skip_req: Optional[Any] = None):
        req_by_rid = {getattr(req, "rid", None): req for req in reqs}
        original_send_output = self.send_to_detokenizer.send_output

        def wrapped_send_output(recv_obj: Any):
            try:
                if isinstance(recv_obj, BatchTokenIDOutput):
                    rids = list(getattr(recv_obj, "rids", []))
                    finished_reasons = getattr(recv_obj, "finished_reasons", None)
                    use_final_only = finished_reasons is not None and len(finished_reasons) == len(rids)
                    spec_accept_lens = []
                    verify_counters = []
                    has_payload = False

                    for idx, rid in enumerate(rids):
                        req = req_by_rid.get(rid)
                        if req is None:
                            spec_accept_lens.append(None)
                            verify_counters.append(None)
                            continue

                        if use_final_only and finished_reasons[idx] is None:
                            spec_accept_lens.append(None)
                            verify_counters.append(None)
                            continue

                        has_payload = True
                        spec_accept_lens.append(list(getattr(req, "_verl_spec_accept_lens_history", [])))
                        verify_counters.append(
                            {
                                "spec_verify_ct": int(getattr(req, "spec_verify_ct", 0)),
                                "spec_accepted_tokens": int(getattr(req, "spec_accepted_tokens", 0)),
                                "target_forward_total_calls": int(getattr(req, "_verl_target_forward_total_calls", 0)),
                                "target_forward_verify_true_calls": int(
                                    getattr(req, "_verl_target_verify_true_calls", 0)
                                ),
                                "target_forward_verify_false_calls": int(
                                    getattr(req, "_verl_target_verify_false_calls", 0)
                                ),
                                "verify_hook_calls": int(getattr(req, "_verl_verify_hook_calls", 0)),
                            }
                        )

                    if has_payload:
                        setattr(recv_obj, "verl_spec_accept_lens", spec_accept_lens)
                        setattr(recv_obj, "verl_verify_counters", verify_counters)
            except Exception as e:  # pragma: no cover - patch safety
                logger.warning("Waste-SD patch failed in stream_output_generation send_output hook: %s", e)
                if strict:
                    raise

            return original_send_output(recv_obj)

        self.send_to_detokenizer.send_output = wrapped_send_output
        try:
            return original_stream(self, reqs, return_logprob, skip_req)
        finally:
            self.send_to_detokenizer.send_output = original_send_output

    def wrapped_handle_batch_output(self: Any, recv_obj: Any):
        rids = list(getattr(recv_obj, "rids", []))
        rid_to_state = {rid: self.rid_to_state.get(rid, None) for rid in rids}
        rid_to_out_len = {rid: len(state.out_list) if state is not None else 0 for rid, state in rid_to_state.items()}

        original_handle_batch_output(self, recv_obj)

        spec_accept_lens = getattr(recv_obj, "verl_spec_accept_lens", None)
        verify_counters = getattr(recv_obj, "verl_verify_counters", None)
        if spec_accept_lens is None and verify_counters is None:
            return

        for idx, rid in enumerate(rids):
            state = rid_to_state.get(rid)
            if state is None:
                continue
            start = rid_to_out_len.get(rid, 0)
            for entry in state.out_list[start:]:
                meta_info = entry.setdefault("meta_info", {})
                if spec_accept_lens is not None and idx < len(spec_accept_lens):
                    meta_info["spec_accept_lens"] = spec_accept_lens[idx]
                if verify_counters is not None and idx < len(verify_counters) and verify_counters[idx] is not None:
                    vc = verify_counters[idx]
                    meta_info["spec_verify_ct"] = vc.get("spec_verify_ct", 0)
                    meta_info["spec_accepted_tokens"] = vc.get("spec_accepted_tokens", 0)
                    meta_info["target_forward_total_calls"] = vc.get("target_forward_total_calls", 0)
                    meta_info["target_forward_verify_true_calls"] = vc.get("target_forward_verify_true_calls", 0)
                    meta_info["target_forward_verify_false_calls"] = vc.get("target_forward_verify_false_calls", 0)
                    meta_info["target_forward_verify_true_called"] = vc.get("target_forward_verify_true_calls", 0) > 0
                    meta_info["verify_hook_calls"] = vc.get("verify_hook_calls", 0)

    def wrapped_eagle_verify(self: Any, batch: Any, spec_info: Any):
        if original_eagle_verify is None:
            return None
        try:
            self._verl_verify_call_count_current = int(getattr(self, "_verl_verify_call_count_current", 0)) + 1
            for req in list(getattr(batch, "reqs", [])):
                req._verl_verify_hook_calls = int(getattr(req, "_verl_verify_hook_calls", 0)) + 1
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed to count eagle verify hook calls: %s", e)
            if strict:
                raise
        out = original_eagle_verify(self, batch, spec_info)
        try:
            # `res.accept_length_per_req_cpu` stores accepted draft tokens per request.
            # Convert to emitted tokens per block by adding 1 target token.
            verify_output = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
            accept_length_per_req_cpu = getattr(verify_output, "accept_length_per_req_cpu", None)
            if accept_length_per_req_cpu is not None:
                emitted_accept_lens = [int(x) + 1 for x in accept_length_per_req_cpu]
                self._verl_last_accept_lens = emitted_accept_lens
                _append_spec_accept_trace_from_verify(
                    batch,
                    accept_length_per_req_cpu,
                    strict=strict,
                    source="eagle verify hook",
                )
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in eagle verify hook: %s", e)
            if strict:
                raise
        return out

    def wrapped_eagle_forward_batch_generation(self: Any, batch: Any):
        if original_eagle_forward_batch_generation is None:
            return None
        # Avoid leaking previous step's cache into this step.
        self._verl_last_accept_lens = None
        self._verl_verify_call_count_current = 0
        verify_counter = verify_counter_enabled
        target_forward_counters = {"total": 0, "verify_true": 0, "verify_false": 0}
        original_target_forward = None
        if verify_counter:
            try:
                original_target_forward = self.target_worker.forward_batch_generation

                def wrapped_target_forward(target_self: Any, *args: Any, **kwargs: Any):
                    is_verify = bool(kwargs.get("is_verify", False))
                    target_forward_counters["total"] += 1
                    if is_verify:
                        target_forward_counters["verify_true"] += 1
                    else:
                        target_forward_counters["verify_false"] += 1
                    reqs_in_call = _extract_reqs_from_forward_call(args, kwargs)
                    for req in reqs_in_call:
                        req._verl_target_forward_total_calls = int(getattr(req, "_verl_target_forward_total_calls", 0)) + 1
                        if is_verify:
                            req._verl_target_verify_true_calls = int(
                                getattr(req, "_verl_target_verify_true_calls", 0)
                            ) + 1
                        else:
                            req._verl_target_verify_false_calls = int(
                                getattr(req, "_verl_target_verify_false_calls", 0)
                            ) + 1
                    return original_target_forward(*args, **kwargs)

                self.target_worker.forward_batch_generation = types.MethodType(wrapped_target_forward, self.target_worker)
            except Exception as e:  # pragma: no cover - patch safety
                logger.warning("Waste-SD patch failed to enable target-forward counter hook: %s", e)
                if strict:
                    raise

        debug_sig = _debug_sig_enabled()
        if debug_sig:
            try:
                draft_sig_before = _model_param_signature(self.model_runner)
                target_sig_before = _model_param_signature(self.target_worker.model_runner)
                debug_count = int(getattr(self, "_verl_debug_sig_count", 0))
                if debug_count < 12:
                    logger.warning(
                        "Waste-SD debug sig[%d] before forward: draft_path=%s draft_sig=%s | target_path=%s target_sig=%s",
                        debug_count,
                        _safe_model_path(self.model_runner),
                        draft_sig_before,
                        _safe_model_path(self.target_worker.model_runner),
                        target_sig_before,
                    )
                prev_target_sig = getattr(self, "_verl_debug_prev_target_sig", None)
                if prev_target_sig is not None and target_sig_before != prev_target_sig:
                    logger.warning(
                        "Waste-SD debug sig target changed across requests: prev=%s now=%s",
                        prev_target_sig,
                        target_sig_before,
                    )
                self._verl_debug_prev_target_sig = target_sig_before
                self._verl_debug_sig_count = debug_count + 1
            except Exception as e:  # pragma: no cover - debug aid
                logger.warning("Waste-SD debug sig probe failed: %s", e)

        try:
            result = original_eagle_forward_batch_generation(self, batch)
        finally:
            if verify_counter and original_target_forward is not None:
                try:
                    self.target_worker.forward_batch_generation = original_target_forward
                except Exception as e:  # pragma: no cover - patch safety
                    logger.warning("Waste-SD patch failed to restore target-forward counter hook: %s", e)
                    if strict:
                        raise
        try:
            if result is None:
                return result

            if verify_counter:
                reqs = list(getattr(batch, "reqs", []))
                rids = [getattr(req, "rid", "<unknown>") for req in reqs]
                verify_hook_calls = int(getattr(self, "_verl_verify_call_count_current", 0))
                for req in reqs:
                    req._verl_verify_hook_calls = int(getattr(req, "_verl_verify_hook_calls", 0))
                debug_count = int(getattr(self, "_verl_verify_counter_log_count", 0))
                if debug_count < 64:
                    logger.warning(
                        "Waste-SD verify counter[%d]: rids=%s target_forward(total=%d, is_verify_true=%d, is_verify_false=%d), verify_hook_calls=%d",
                        debug_count,
                        rids,
                        int(target_forward_counters["total"]),
                        int(target_forward_counters["verify_true"]),
                        int(target_forward_counters["verify_false"]),
                        verify_hook_calls,
                    )
                self._verl_verify_counter_log_count = debug_count + 1

            if getattr(result, "accept_lens", None) is not None:
                return result

            spec_algorithm = getattr(batch, "spec_algorithm", None)
            is_spec = spec_algorithm is not None and (not spec_algorithm.is_none())
            if not is_spec or getattr(batch, "is_v2_eagle", False):
                return result

            accept_lens = getattr(self, "_verl_last_accept_lens", None)
            reqs = getattr(batch, "reqs", [])
            if accept_lens is None:
                if verify_counter:
                    rids = [getattr(req, "rid", "<unknown>") for req in reqs]
                    logger.warning(
                        "Waste-SD verify counter(strict-fallback): rids=%s target_forward(total=%d, is_verify_true=%d, is_verify_false=%d), verify_hook_calls=%d",
                        rids,
                        int(target_forward_counters["total"]),
                        int(target_forward_counters["verify_true"]),
                        int(target_forward_counters["verify_false"]),
                        int(getattr(self, "_verl_verify_call_count_current", 0)),
                    )
                if strict:
                    logger.error(
                        "Waste-SD strict mode: non-v2 speculative decode did not expose accept lengths; "
                        "leaving spec_accept_lens empty for trainer-side strict check."
                    )
                return result
            if len(accept_lens) != len(reqs):
                msg = (
                    "Waste-SD patch alignment error in eagle forward hook: "
                    f"len(accept_lens)={len(accept_lens)} vs len(reqs)={len(reqs)}."
                )
                if strict:
                    raise RuntimeError(msg)
                logger.warning("%s Skip this capture step.", msg)
                return result

            result.accept_lens = torch.tensor(accept_lens, dtype=torch.int32, device="cpu")
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in eagle forward hook: %s", e)
            if strict:
                raise
        return result

    def wrapped_detok_handle_batch_token_id_out(self: Any, recv_obj: Any):
        if original_detok_handle_batch_token_id_out is None:
            return None
        out_obj = original_detok_handle_batch_token_id_out(self, recv_obj)
        try:
            spec_accept_lens = getattr(recv_obj, "verl_spec_accept_lens", None)
            verify_counters = getattr(recv_obj, "verl_verify_counters", None)
            if spec_accept_lens is not None and out_obj is not None:
                setattr(out_obj, "verl_spec_accept_lens", spec_accept_lens)
            if verify_counters is not None and out_obj is not None:
                setattr(out_obj, "verl_verify_counters", verify_counters)
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in detokenizer hook: %s", e)
            if strict:
                raise
        return out_obj

    def wrapped_multi_handle_output_by_index(output: Any, i: int):
        if original_multi_handle_output_by_index is None:
            return None
        new_output = original_multi_handle_output_by_index(output, i)
        try:
            spec_accept_lens = getattr(output, "verl_spec_accept_lens", None)
            if spec_accept_lens is not None and i < len(spec_accept_lens):
                setattr(new_output, "verl_spec_accept_lens", [spec_accept_lens[i]])
            verify_counters = getattr(output, "verl_verify_counters", None)
            if verify_counters is not None and i < len(verify_counters):
                setattr(new_output, "verl_verify_counters", [verify_counters[i]])
        except Exception as e:  # pragma: no cover - patch safety
            logger.warning("Waste-SD patch failed in multi-tokenizer output split hook: %s", e)
            if strict:
                raise
        return new_output

    SchedulerOutputProcessorMixin._resolve_spec_overlap_token_ids = wrapped_resolve
    SchedulerOutputProcessorMixin.process_batch_result_prefill = wrapped_prefill
    SchedulerOutputProcessorMixin.process_batch_result_decode = wrapped_decode
    SchedulerOutputProcessorMixin.stream_output_generation = wrapped_stream
    TokenizerManager._handle_batch_output = wrapped_handle_batch_output
    if DetokenizerManager is not None and original_detok_handle_batch_token_id_out is not None:
        DetokenizerManager.handle_batch_token_id_out = wrapped_detok_handle_batch_token_id_out
    if EAGLEWorker is not None and original_eagle_verify is not None:
        EAGLEWorker.verify = wrapped_eagle_verify
    if EAGLEWorker is not None and original_eagle_forward_batch_generation is not None:
        EAGLEWorker.forward_batch_generation = wrapped_eagle_forward_batch_generation
    if original_multi_handle_output_by_index is not None:
        multi_tokenizer_mixin_mod._handle_output_by_index = wrapped_multi_handle_output_by_index
    setattr(SchedulerOutputProcessorMixin, _PATCH_MARKER, True)
    setattr(TokenizerManager, _PATCH_MARKER, True)
    logger.warning(
        "Applied Waste-SD SGLang patch (env %s=1, strict=%s, eagle=%s, detok=%s, multi_tokenizer=%s, verify_counter=%s).",
        _ENABLE_ENV,
        strict,
        original_eagle_verify is not None,
        original_detok_handle_batch_token_id_out is not None,
        original_multi_handle_output_by_index is not None,
        verify_counter_enabled,
    )
