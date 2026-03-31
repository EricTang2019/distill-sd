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
import math
import os
from typing import Any

import torch

from recipe.waste_sd.block_count_dp import compute_forward_remaining_budget_weights
from recipe.waste_sd.distill_debug import DistillDebugRecorder
from recipe.waste_sd.distill_losses import get_distill_loss_fn
from recipe.waste_sd.exact_block_count_loss import compute_exact_block_count_wnll_from_logits
from recipe.waste_sd.waste_weighting import build_strict_weights
from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SUPPORTED_WEIGHTING_MODES = {"waste", "uniform_mean", "remaining_budget_forward"}


def validate_distill_objective_config(
    loss_type: str,
    weighting_mode: str,
    kl_floor_coef: float,
    rembudget_unweighted_kl_coef: float = 0.0,
    exact_unweighted_kl_coef: float = 0.0,
) -> None:
    if weighting_mode not in _SUPPORTED_WEIGHTING_MODES:
        raise ValueError(
            "distill.weighting_mode must be one of "
            f"{sorted(_SUPPORTED_WEIGHTING_MODES)!r}, got {weighting_mode!r}."
        )
    if loss_type == "teacher_greedy_nll" and weighting_mode != "uniform_mean":
        raise ValueError(
            "distill.loss_type='teacher_greedy_nll' currently requires "
            "distill.weighting_mode='uniform_mean' so the objective remains plain "
            "teacher-greedy token MLE on sampled prefixes. "
            f"Got weighting_mode={weighting_mode!r}."
        )
    if loss_type == "exact_block_count_wnll" and weighting_mode != "uniform_mean":
        raise ValueError(
            "distill.loss_type='exact_block_count_wnll' currently requires "
            "distill.weighting_mode='uniform_mean'. The exact theorem-backed objective "
            "is a detached weighted sampled-token NLL with alpha-weighted token-mean aggregation, "
            "so waste/remaining-budget weighting modes do not apply on top. "
            f"Got weighting_mode={weighting_mode!r}."
        )
    if weighting_mode == "remaining_budget_forward" and loss_type not in {"fkl", "tvd"}:
        raise ValueError(
            "distill.weighting_mode='remaining_budget_forward' currently requires "
            "distill.loss_type in {'fkl', 'tvd'}. "
            f"Got loss_type={loss_type!r}."
        )
    if not (0.0 <= kl_floor_coef <= 1.0):
        raise ValueError(f"distill.kl_floor_coef must be in [0, 1], got {kl_floor_coef}.")
    if not (0.0 <= rembudget_unweighted_kl_coef <= 1.0):
        raise ValueError(
            "distill.rembudget_unweighted_kl_coef must be in [0, 1], "
            f"got {rembudget_unweighted_kl_coef}."
        )
    if rembudget_unweighted_kl_coef > 0.0 and not (
        weighting_mode == "remaining_budget_forward" and loss_type in {"fkl", "tvd"}
    ):
        raise ValueError(
            "distill.rembudget_unweighted_kl_coef only applies to "
            "distill.weighting_mode='remaining_budget_forward' with "
            "distill.loss_type in {'fkl', 'tvd'}. "
            f"Got weighting_mode={weighting_mode!r}, loss_type={loss_type!r}."
        )
    if not (0.0 <= exact_unweighted_kl_coef <= 1.0):
        raise ValueError(
            "distill.exact_unweighted_kl_coef must be in [0, 1], "
            f"got {exact_unweighted_kl_coef}."
        )
    if exact_unweighted_kl_coef > 0.0 and loss_type != "exact_block_count_wnll":
        raise ValueError(
            "distill.exact_unweighted_kl_coef only applies to "
            "distill.loss_type='exact_block_count_wnll'. "
            f"Got loss_type={loss_type!r}."
        )


class DataParallelWasteSDDistillActor(DataParallelPPOActor):
    """FSDP actor used by strict waste-aware SD distillation."""

    def __init__(
        self,
        config,
        actor_module: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        distill_config: Any,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.distill_config = distill_config
        self.loss_type = str(distill_config.get("loss_type", "fkl")).lower()
        self.q_source = str(distill_config.get("q_source", "local_ref")).lower()
        self.weighting_mode = str(distill_config.get("weighting_mode", "waste")).lower()
        self.gamma = int(distill_config.get("gamma", 1))
        self.kl_floor_coef = float(distill_config.get("kl_floor_coef", 0.0))
        self.rembudget_unweighted_kl_coef = float(distill_config.get("rembudget_unweighted_kl_coef", 0.0))
        self.exact_unweighted_kl_coef = float(distill_config.get("exact_unweighted_kl_coef", 0.0))
        self.strict = bool(distill_config.get("strict", True))
        self.forward_path = str(distill_config.get("forward_path", "auto")).lower()
        legacy_full_block_flag = distill_config.get("full_block_participate", None)
        if legacy_full_block_flag is not None:
            logger.warning(
                "distill.full_block_participate is deprecated and ignored. "
                "Full blocks are always handled by the strict weight formula."
            )
        if self.q_source != "local_ref":
            raise ValueError(
                "waste_sd actor now requires distill.q_source=local_ref. "
                f"Got {self.q_source}."
            )
        validate_distill_objective_config(
            loss_type=self.loss_type,
            weighting_mode=self.weighting_mode,
            kl_floor_coef=self.kl_floor_coef,
            rembudget_unweighted_kl_coef=self.rembudget_unweighted_kl_coef,
            exact_unweighted_kl_coef=self.exact_unweighted_kl_coef,
        )
        self.teacher_module: torch.nn.Module | None = None
        self.loss_fn = None if self.loss_type == "exact_block_count_wnll" else get_distill_loss_fn(self.loss_type)
        self.unweighted_fkl_loss_fn = get_distill_loss_fn("fkl")
        self.debug_recorder = DistillDebugRecorder(distill_config.get("debug", {}))
        self._local_update_step = 0
        self._warned_forward_path_fallback = False
        self._sync_debug = os.getenv("WASTE_SD_DISTILL_SYNC_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _forward_response_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        *,
        module: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if self.forward_path == "remove_padding":
            # Keep forward_path flag for future experiments but use padded forward in v1 for robustness.
            if not self._warned_forward_path_fallback:
                logger.warning("forward_path=remove_padding is not fully supported in waste_sd v1, fallback to padded.")
                self._warned_forward_path_fallback = True
        elif self.forward_path not in {"auto", "padded_only", "remove_padding"}:
            raise ValueError(f"Unsupported distill.forward_path={self.forward_path}")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        response_length = micro_batch["responses"].size(-1)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        model = module or self.actor_module
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = outputs.logits[:, -response_length - 1 : -1, :]
            logits = logits.float()
            logits.div_(temperature)
        return logits

    def _get_non_tensor_value(self, data: DataProto, key: str, index: int):
        if key not in data.non_tensor_batch:
            return None
        values = data.non_tensor_batch[key]
        if index >= len(values):
            return None
        return values[index]

    @staticmethod
    def _reduce_list_metric(metric_key: str, values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        if "max" in metric_key:
            return float(max(values))
        if "min" in metric_key:
            return float(min(values))
        return float(sum(float(x) for x in values) / len(values))

    def _reduce_sample_metrics(self, sample_metrics: dict[str, Any]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in sample_metrics.items():
            if isinstance(value, list):
                reduced[key] = self._reduce_list_metric(key, value)
            else:
                reduced[key] = float(value)
        return reduced

    def _init_micro_metrics(self, *, local_has_contrib: bool, contributing_samples: int) -> dict[str, float]:
        micro_metrics: dict[str, float] = {
            "distill/zero_contrib_micro_batches": 1.0 if not local_has_contrib else 0.0,
            "distill/rank_zero_contrib_while_global_nonzero": 0.0,
            "distill/weighted_divergence_mean": 0.0,
            f"distill/weighted_{self.loss_type}_mean": 0.0,
            "distill/contributing_samples": float(contributing_samples),
            "distill/micro_weight_sum": 0.0,
            "distill/weight_nonzero_ratio": 0.0,
            "distill/weight_mean": 0.0,
            "distill/weight_max": 0.0,
            "distill/weight_sum": 0.0,
            "distill/strict_alignment_ok": 1.0,
            f"distill/{self.loss_type}_token_div_mean": 0.0,
            f"distill/{self.loss_type}_token_div_max": 0.0,
            "distill/student_nll_all_response_tokens_mean": 0.0,
            "distill/token_count": 0.0,
        }
        if self.loss_type == "fkl":
            micro_metrics["distill/weighted_kl_mean"] = 0.0
        if self.weighting_mode != "uniform_mean":
            micro_metrics["actor/distill_loss"] = 0.0
        return micro_metrics

    @staticmethod
    def _gather_response_token_logprobs(logits: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return torch.gather(log_probs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        return torch.sum(values * mask_f, dim=-1) / denom

    @staticmethod
    def _weighted_token_mean(weighted_sum: torch.Tensor, weight_sum: torch.Tensor) -> torch.Tensor:
        if float(weight_sum.item()) > 0.0:
            return weighted_sum / weight_sum
        return weighted_sum * 0.0

    def _student_token_nll_stats(
        self,
        student_logits: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[float, float]:
        token_logp = self._gather_response_token_logprobs(student_logits, responses)
        token_nll = -token_logp[response_mask]
        if token_nll.numel() == 0:
            return 0.0, 0.0
        return token_nll.mean().item(), token_nll.max().item()

    def update_policy_distill(self, data: DataProto) -> dict[str, Any]:
        current_step = int(data.meta_info.get("global_steps", self._local_update_step))
        self._local_update_step += 1
        self.actor_module.train()
        if self.teacher_module is None:
            raise RuntimeError(
                "waste_sd requires a local teacher model in actor worker. "
                "Ensure ref model is initialized and attached as actor.teacher_module."
            )
        self.teacher_module.eval()

        if self.config.get("use_dynamic_bsz", False):
            raise NotImplementedError("waste_sd v1 does not support use_dynamic_bsz=True for distill update.")

        temperature = float(data.meta_info["temperature"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"Invalid distillation temperature={temperature}. "
                "Expected a positive finite value."
            )
        pad_token_id = int(data.meta_info.get("pad_token_id", 0))

        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = [
            key
            for key in ["spec_accept_lens", "uid", "multi_modal_inputs"]
            if key in data.non_tensor_batch
        ]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batch_size = int(self.config.ppo_mini_batch_size)
        micro_batch_size = int(self.config.ppo_micro_batch_size_per_gpu)
        mini_batches = data.split(mini_batch_size)

        metrics: dict[str, Any] = {"actor/distill_loss": []}
        did_update_step = False
        uniform_step_weighted_sum = 0.0
        uniform_step_weight_sum = 0.0
        exact_step_loss_sum = 0.0
        exact_step_weight_sum = 0.0
        exact_kl_step_weighted_sum = 0.0
        exact_kl_step_weight_sum = 0.0

        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(micro_batch_size)
            grad_accum_steps = max(len(micro_batches), 1)
            uniform_total_weight_sum = 0.0
            exact_total_weight_sum = 0.0
            trainable_params = [param for param in self.actor_module.parameters() if param.requires_grad]
            exact_grad_buffers: list[torch.Tensor | None] = [None] * len(trainable_params)
            local_has_contrib_flags: list[bool] = []
            micro_records: list[dict[str, Any]] = []
            sync_device = None
            if self.weighting_mode == "uniform_mean" and self.loss_type != "exact_block_count_wnll":
                # Exact token-level mean across the whole mini-batch:
                #   L = (sum_i weighted_sum_i) / (sum_i weight_sum_i)
                # For uniform_mean, weight_sum_i is the number of valid response tokens.
                for micro_batch_ref in micro_batches:
                    uniform_total_weight_sum += float(micro_batch_ref.batch["response_mask"].sum().item())

            self.actor_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}

                response_mask = model_inputs["response_mask"].bool()
                student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
                with torch.no_grad():
                    teacher_logits = self._forward_response_logits(
                        model_inputs,
                        temperature=temperature,
                        module=self.teacher_module,
                    )

                micro_weighted_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
                micro_weight_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
                contributing_samples = 0
                sample_level_metrics = {}

                bsz, response_len, _ = student_logits.shape
                exact_loss_result = None
                if self.loss_type == "exact_block_count_wnll":
                    exact_loss_result = compute_exact_block_count_wnll_from_logits(
                        teacher_logits=teacher_logits,
                        student_logits=student_logits,
                        responses=model_inputs["responses"],
                        gamma=self.gamma,
                        response_mask=response_mask,
                        dp_dtype=torch.float64,
                    )
                    contributing_samples = int(exact_loss_result.contributing_samples)
                    if contributing_samples > 0:
                        valid_token_counts = response_mask.sum(dim=-1)
                        valid_sample_mask = valid_token_counts > 0
                        valid_counts_f = valid_token_counts[valid_sample_mask].to(dtype=torch.float32)
                        valid_mask = response_mask[valid_sample_mask]
                        valid_mask_float = valid_mask.to(dtype=torch.float32)
                        valid_alpha = exact_loss_result.alpha[valid_sample_mask]
                        valid_alpha_sum = torch.sum(valid_alpha * valid_mask_float, dim=-1)
                        valid_alpha_mean = valid_alpha_sum / valid_counts_f
                        valid_alpha_nonzero_ratio = (
                            torch.sum((valid_alpha > 0).to(dtype=torch.float32) * valid_mask_float, dim=-1)
                            / valid_counts_f
                        )
                        valid_alpha_max = torch.max(valid_alpha.masked_fill(~valid_mask, 0.0), dim=-1).values
                        valid_omega_mean = self._masked_sample_mean(
                            exact_loss_result.dp_result.omega[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_reject_mean = self._masked_sample_mean(
                            exact_loss_result.dp_result.reject_prob[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_sample_loss = exact_loss_result.sample_loss_sum[valid_sample_mask].to(dtype=torch.float32)
                        valid_token_nll = exact_loss_result.token_nll[valid_sample_mask].to(dtype=torch.float32)
                        valid_token_nll_mean = self._masked_sample_mean(valid_token_nll, valid_mask)
                        valid_token_nll_max = torch.max(
                            valid_token_nll.masked_fill(~valid_mask, 0.0),
                            dim=-1,
                        ).values
                        sample_level_metrics = {
                            "distill/weight_nonzero_ratio": valid_alpha_nonzero_ratio.tolist(),
                            "distill/weight_mean": valid_alpha_mean.tolist(),
                            "distill/weight_max": valid_alpha_max.tolist(),
                            "distill/weight_sum": valid_alpha_sum.tolist(),
                            "distill/strict_alignment_ok": [1.0] * contributing_samples,
                            "distill/block_count_omega_mean": valid_omega_mean.tolist(),
                            "distill/reject_prob_mean": valid_reject_mean.tolist(),
                            "distill/student_nll_on_teacher_tokens_mean": valid_token_nll_mean.tolist(),
                            "distill/student_nll_on_teacher_tokens_max": valid_token_nll_max.tolist(),
                            "distill/sample_loss_sum": valid_sample_loss.tolist(),
                        }
                        if self.debug_recorder.enabled:
                            valid_indices = valid_sample_mask.nonzero(as_tuple=False).flatten().tolist()
                            for i in valid_indices:
                                sample_mask = response_mask[i]
                                sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                                sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                                sample_responses = model_inputs["responses"][i][sample_mask]
                                sample_alpha = exact_loss_result.alpha[i][sample_mask]
                                self.debug_recorder.maybe_record_sample(
                                    step=current_step,
                                    uid=sample_uid,
                                    sample_index=i,
                                    loss_type=self.loss_type,
                                    gamma=self.gamma,
                                    strict=self.strict,
                                    temperature=temperature,
                                    spec_accept_lens=sample_accept_lens,
                                    response_ids=sample_responses,
                                    token_weights=sample_alpha,
                                    student_logits=student_logits[i][sample_mask],
                                    teacher_logits=teacher_logits[i][sample_mask],
                                )
                elif self.weighting_mode == "uniform_mean":
                    # Off-policy baseline fast path: uniform weights on valid response tokens.
                    token_weights = response_mask.to(dtype=torch.float32)
                    valid_token_counts = response_mask.sum(dim=-1)
                    valid_sample_mask = valid_token_counts > 0
                    contributing_samples = int(valid_sample_mask.sum().item())
                    if contributing_samples > 0:
                        count = int(contributing_samples)
                        sample_level_metrics = {
                            "distill/weight_nonzero_ratio": [1.0] * count,
                            "distill/weight_mean": [1.0] * count,
                            "distill/weight_max": [1.0] * count,
                            "distill/weight_sum": valid_token_counts[valid_sample_mask].to(dtype=torch.float32).tolist(),
                            "distill/strict_alignment_ok": [1.0] * count,
                        }
                        if self.debug_recorder.enabled:
                            valid_indices = valid_sample_mask.nonzero(as_tuple=False).flatten().tolist()
                            for i in valid_indices:
                                sample_mask = response_mask[i]
                                valid_tokens = int(valid_token_counts[i].item())
                                sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                                sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                                sample_responses = None
                                if "responses" in model_inputs:
                                    sample_responses = model_inputs["responses"][i][sample_mask]
                                sample_weights = torch.ones(
                                    (valid_tokens,),
                                    dtype=torch.float32,
                                    device=student_logits.device,
                                )
                                self.debug_recorder.maybe_record_sample(
                                    step=current_step,
                                    uid=sample_uid,
                                    sample_index=i,
                                    loss_type=self.loss_type,
                                    gamma=self.gamma,
                                    strict=self.strict,
                                    temperature=temperature,
                                    spec_accept_lens=sample_accept_lens,
                                    response_ids=sample_responses,
                                    token_weights=sample_weights,
                                    student_logits=student_logits[i][sample_mask],
                                    teacher_logits=teacher_logits[i][sample_mask],
                                )
                elif self.weighting_mode == "waste":
                    token_weights = torch.zeros((bsz, response_len), dtype=torch.float32, device=student_logits.device)
                    for i in range(bsz):
                        sample_mask = response_mask[i]
                        valid_tokens = int(sample_mask.sum().item())
                        if valid_tokens <= 0:
                            continue

                        sample_accept_lens = self._get_non_tensor_value(micro_batch, "spec_accept_lens", i)
                        sample_uid = self._get_non_tensor_value(micro_batch, "uid", i)
                        if self.strict and (sample_accept_lens is None or len(sample_accept_lens) == 0):
                            raise ValueError(
                                "Strict alignment requires non-empty spec_accept_lens for every valid sample. "
                                f"Got spec_accept_lens={sample_accept_lens!r}, valid_tokens={valid_tokens}, uid={sample_uid!r}. "
                                "This usually means SGLang patch metadata did not propagate."
                            )
                        sample_weights, weight_metrics = build_strict_weights(
                            spec_accept_lens=sample_accept_lens,
                            response_valid_len=valid_tokens,
                            gamma=self.gamma,
                            strict=self.strict,
                            device=student_logits.device,
                        )
                        append_to_dict(sample_level_metrics, weight_metrics)
                        token_weights[i, sample_mask] = sample_weights

                        sample_responses = None
                        if "responses" in model_inputs:
                            sample_responses = model_inputs["responses"][i][sample_mask]
                        self.debug_recorder.maybe_record_sample(
                            step=current_step,
                            uid=sample_uid,
                            sample_index=i,
                            loss_type=self.loss_type,
                            gamma=self.gamma,
                            strict=self.strict,
                            temperature=temperature,
                            spec_accept_lens=sample_accept_lens,
                            response_ids=sample_responses,
                            token_weights=sample_weights,
                            student_logits=student_logits[i][sample_mask],
                            teacher_logits=teacher_logits[i][sample_mask],
                        )
                        contributing_samples += 1
                else:
                    with torch.no_grad():
                        teacher_token_logprobs = self._gather_response_token_logprobs(
                            teacher_logits,
                            model_inputs["responses"],
                        )
                        student_token_logprobs = self._gather_response_token_logprobs(
                            student_logits.detach(),
                            model_inputs["responses"],
                        )
                        forward_weight_result = compute_forward_remaining_budget_weights(
                            teacher_logprobs=teacher_token_logprobs,
                            student_logprobs=student_token_logprobs,
                            gamma=self.gamma,
                            response_mask=response_mask,
                            kl_floor_coef=self.kl_floor_coef,
                            dp_dtype=torch.float64,
                        )
                        token_weights = forward_weight_result.mixed_weight.to(
                            device=student_logits.device,
                            dtype=torch.float32,
                        )

                    valid_token_counts = response_mask.sum(dim=-1)
                    valid_sample_mask = valid_token_counts > 0
                    contributing_samples = int(valid_sample_mask.sum().item())
                    if contributing_samples > 0:
                        valid_counts_f = valid_token_counts[valid_sample_mask].to(dtype=torch.float32)
                        valid_weights = token_weights[valid_sample_mask]
                        valid_mask_float = response_mask[valid_sample_mask].to(dtype=torch.float32)
                        valid_weight_sum = torch.sum(valid_weights * valid_mask_float, dim=-1)
                        valid_weight_mean = valid_weight_sum / valid_counts_f
                        valid_weight_nonzero_ratio = (
                            torch.sum((valid_weights > 0).to(dtype=torch.float32) * valid_mask_float, dim=-1)
                            / valid_counts_f
                        )
                        valid_weight_max = torch.max(valid_weights.masked_fill(~response_mask[valid_sample_mask], 0.0), dim=-1).values
                        valid_mask = response_mask[valid_sample_mask]
                        valid_rem_mean = self._masked_sample_mean(
                            forward_weight_result.remaining_budget_weight[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_pos_mean = self._masked_sample_mean(
                            forward_weight_result.expected_position[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        valid_reject_mean = self._masked_sample_mean(
                            forward_weight_result.reject_prob[valid_sample_mask].to(dtype=torch.float32),
                            valid_mask,
                        )
                        sample_level_metrics = {
                            "distill/weight_nonzero_ratio": valid_weight_nonzero_ratio.tolist(),
                            "distill/weight_mean": valid_weight_mean.tolist(),
                            "distill/weight_max": valid_weight_max.tolist(),
                            "distill/weight_sum": valid_weight_sum.tolist(),
                            "distill/strict_alignment_ok": [1.0] * contributing_samples,
                            "distill/remaining_budget_weight_mean": valid_rem_mean.tolist(),
                            "distill/expected_position_mean": valid_pos_mean.tolist(),
                            "distill/reject_prob_mean": valid_reject_mean.tolist(),
                        }

                local_valid_tokens = int(response_mask.sum().item())
                local_has_contrib = contributing_samples > 0
                local_has_contrib_flags.append(local_has_contrib)
                sync_device = student_logits.device
                dist_rank = -1
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    dist_rank = torch.distributed.get_rank()

                micro_metrics = self._init_micro_metrics(
                    local_has_contrib=local_has_contrib,
                    contributing_samples=contributing_samples,
                )

                micro_objective = torch.zeros((), device=student_logits.device, dtype=torch.float32)

                if local_has_contrib:
                    if self.loss_type == "exact_block_count_wnll":
                        assert exact_loss_result is not None
                        micro_weighted_sum = exact_loss_result.total_loss_sum
                        micro_weight_sum = exact_loss_result.alpha_weight_sum
                        micro_weighted_mean = exact_loss_result.loss
                        exact_kl_coef = self.exact_unweighted_kl_coef
                        exact_coef = 1.0 - exact_kl_coef
                        valid_token_nll_flat = exact_loss_result.token_nll[response_mask].to(dtype=torch.float32)
                        micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                        micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = micro_weighted_mean.detach().item()
                        micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                        micro_metrics[f"distill/{self.loss_type}_token_div_mean"] = valid_token_nll_flat.mean().item()
                        micro_metrics[f"distill/{self.loss_type}_token_div_max"] = valid_token_nll_flat.max().item()
                        micro_metrics["distill/student_nll_all_response_tokens_mean"] = valid_token_nll_flat.mean().item()
                        micro_metrics["distill/token_count"] = float(response_mask.sum().item())
                        micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                        if exact_kl_coef > 0.0:
                            uniform_token_weights = response_mask.to(dtype=torch.float32)
                            kl_loss, kl_weighted_sum, kl_weight_sum, _ = self.unweighted_fkl_loss_fn(
                                student_logits,
                                {"logits": teacher_logits},
                                uniform_token_weights,
                                response_mask=response_mask,
                            )
                            micro_metrics["distill/unweighted_fkl_mean"] = kl_loss.detach().item()
                            micro_metrics["distill/mixed_exact_unweighted_fkl_mean"] = (
                                exact_coef * micro_weighted_mean.detach().item()
                                + exact_kl_coef * kl_loss.detach().item()
                            )
                            exact_kl_step_weighted_sum += float(kl_weighted_sum.detach().item())
                            exact_kl_step_weight_sum += float(kl_weight_sum.detach().item())
                            # Mixed exact+FKL under FSDP is treated as a convex combination of
                            # per-micro normalized objectives. The exact-only path below keeps the
                            # stricter global alpha-normalized aggregation, but mixing that with an
                            # additional KL backward via autograd.grad + backward on the same graph
                            # proved fragile under FSDP sharding.
                            micro_objective = exact_coef * micro_weighted_mean + exact_kl_coef * kl_loss
                        else:
                            micro_objective = micro_weighted_sum
                        exact_total_weight_sum += float(micro_weight_sum.detach().item())
                        exact_step_loss_sum += float(micro_weighted_sum.detach().item())
                        exact_step_weight_sum += float(micro_weight_sum.detach().item())
                    else:
                        all_token_nll_mean, all_token_nll_max = self._student_token_nll_stats(
                            student_logits=student_logits,
                            responses=model_inputs["responses"],
                            response_mask=response_mask,
                        )
                        if (
                            self.weighting_mode == "remaining_budget_forward"
                            and self.loss_type in {"fkl", "tvd"}
                            and self.rembudget_unweighted_kl_coef > 0.0
                        ):
                            # Reuse a single dense token-divergence graph, then reduce it two
                            # ways: remaining-budget weighted and plain unweighted. Calling the
                            # same dense loss twice on the same FSDP-backed logits graph can
                            # trigger fragile backward/view behavior, so keep this branch
                            # single-pass.
                            if self.loss_type == "fkl":
                                student_logp = torch.log_softmax(student_logits.float(), dim=-1)
                                teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
                                teacher_prob = torch.exp(teacher_logp)
                                token_div = torch.sum(
                                    torch.where(
                                        teacher_prob > 0,
                                        teacher_prob * (teacher_logp - student_logp),
                                        teacher_logp.new_zeros(()),
                                    ),
                                    dim=-1,
                                )
                            else:
                                student_prob = torch.softmax(student_logits.float(), dim=-1)
                                teacher_prob = torch.softmax(teacher_logits.float(), dim=-1)
                                token_div = 0.5 * torch.sum(torch.abs(student_prob - teacher_prob), dim=-1)
                            flat_mask = response_mask.reshape(-1)
                            token_div_flat = token_div.reshape(-1)[flat_mask]
                            weight_flat = token_weights.reshape(-1)[flat_mask].to(dtype=torch.float32)
                            if token_div_flat.numel() == 0:
                                micro_weighted_sum = student_logits.new_zeros((), dtype=torch.float32)
                                micro_weight_sum = student_logits.new_zeros((), dtype=torch.float32)
                                micro_weighted_mean = micro_weighted_sum
                                unweighted_loss = micro_weighted_sum
                                loss_metrics = {
                                    f"distill/{self.loss_type}_token_div_mean": 0.0,
                                    f"distill/{self.loss_type}_token_div_max": 0.0,
                                    "distill/token_count": 0.0,
                                }
                            else:
                                micro_weighted_sum = torch.sum(token_div_flat * weight_flat)
                                micro_weight_sum = torch.sum(weight_flat)
                                micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                                unweighted_loss = token_div_flat.mean()
                                loss_metrics = {
                                    f"distill/{self.loss_type}_token_div_mean": token_div_flat.mean().item(),
                                    f"distill/{self.loss_type}_token_div_max": token_div_flat.max().item(),
                                    "distill/token_count": float(token_div_flat.numel()),
                                }
                            append_to_dict(sample_level_metrics, loss_metrics)
                            micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                            if self.loss_type == "fkl":
                                micro_metrics["distill/weighted_kl_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))
                            mix_coef = self.rembudget_unweighted_kl_coef
                            micro_metrics[f"distill/unweighted_{self.loss_type}_mean"] = unweighted_loss.detach().item()
                            micro_metrics[f"distill/mixed_rembudget_unweighted_{self.loss_type}_mean"] = (
                                (1.0 - mix_coef) * micro_weighted_mean.detach().item()
                                + mix_coef * unweighted_loss.detach().item()
                            )
                            micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                            micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                            micro_objective = (1.0 - mix_coef) * micro_weighted_mean + mix_coef * unweighted_loss
                            micro_metrics["actor/distill_loss"] = micro_objective.detach().item()
                        else:
                            _, micro_weighted_sum, micro_weight_sum, loss_metrics = self.loss_fn(
                                student_logits,
                                {"logits": teacher_logits},
                                token_weights,
                                response_mask=response_mask,
                            )
                            append_to_dict(sample_level_metrics, loss_metrics)

                            micro_weighted_mean = self._weighted_token_mean(micro_weighted_sum, micro_weight_sum)
                            micro_metrics["distill/weighted_divergence_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics[f"distill/weighted_{self.loss_type}_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics["distill/micro_weight_sum"] = micro_weight_sum.detach().item()
                            if self.loss_type == "fkl":
                                micro_metrics["distill/weighted_kl_mean"] = micro_weighted_mean.detach().item()
                            micro_metrics["distill/student_nll_all_response_tokens_mean"] = all_token_nll_mean
                            micro_metrics["distill/student_nll_all_response_tokens_max"] = all_token_nll_max
                            micro_metrics.update(self._reduce_sample_metrics(sample_level_metrics))

                            if self.weighting_mode == "uniform_mean":
                                # Backprop per-micro weighted-sum contribution normalized by
                                # mini-batch total valid-token count to preserve exact token-level mean.
                                if uniform_total_weight_sum > 0.0:
                                    micro_objective = micro_weighted_sum / float(uniform_total_weight_sum)
                                else:
                                    micro_objective = micro_weighted_sum * 0.0
                                uniform_step_weighted_sum += float(micro_weighted_sum.detach().item())
                                uniform_step_weight_sum += float(micro_weight_sum.detach().item())
                            else:
                                # Optimize weighted token-mean objective for soft distillation:
                                #   (sum_n w_n * D_n) / (sum_n w_n)
                                # This keeps the weighting semantics at the token level instead of
                                # implicitly upweighting long responses via per-sample sums.
                                micro_objective = micro_weighted_mean
                                micro_metrics["actor/distill_loss"] = micro_objective.detach().item()

                if local_has_contrib:
                    if self.loss_type == "exact_block_count_wnll" and self.exact_unweighted_kl_coef > 0.0:
                        scaled_loss = micro_objective / grad_accum_steps
                    elif self.weighting_mode == "uniform_mean":
                        scaled_loss = micro_objective
                    else:
                        scaled_loss = micro_objective / grad_accum_steps
                else:
                    # Keep FSDP collectives aligned even when this rank has no valid tokens.
                    scaled_loss = student_logits.sum() * 0.0

                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_records.append(
                    {
                        "metrics": micro_metrics,
                        "local_has_contrib": local_has_contrib,
                        "local_valid_tokens": local_valid_tokens,
                        "contributing_samples": contributing_samples,
                        "micro_batch_size": bsz,
                        "dist_rank": dist_rank,
                    }
                )

            global_has_contrib_flags = list(local_has_contrib_flags)
            if (
                len(local_has_contrib_flags) > 0
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                global_flags_t = torch.tensor(
                    [1 if flag else 0 for flag in local_has_contrib_flags],
                    dtype=torch.int32,
                    device=sync_device,
                )
                torch.distributed.all_reduce(global_flags_t, op=torch.distributed.ReduceOp.SUM)
                global_has_contrib_flags = [bool(v > 0) for v in global_flags_t.tolist()]

            global_has_contrib = any(global_has_contrib_flags)
            for record, micro_global_has_contrib in zip(micro_records, global_has_contrib_flags):
                micro_metrics = record["metrics"]
                local_has_contrib = bool(record["local_has_contrib"])
                if (not local_has_contrib) and micro_global_has_contrib:
                    micro_metrics["distill/rank_zero_contrib_while_global_nonzero"] = 1.0
                    if self._sync_debug:
                        logger.warning(
                            "[distill-sync-debug] step=%s rank=%s local_valid_tokens=%s "
                            "contributing_samples=%s micro_batch_size=%s",
                            current_step,
                            int(record["dist_rank"]),
                            int(record["local_valid_tokens"]),
                            int(record["contributing_samples"]),
                            int(record["micro_batch_size"]),
                        )
                append_to_dict(metrics, micro_metrics)

            if global_has_contrib:
                if self.loss_type == "exact_block_count_wnll":
                    if self.exact_unweighted_kl_coef <= 0.0 and exact_total_weight_sum > 0.0:
                        grad_scale = 1.0 / float(exact_total_weight_sum)
                        for param in self.actor_module.parameters():
                            if param.grad is not None:
                                param.grad.mul_(grad_scale)
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
                did_update_step = True
            else:
                self.actor_optimizer.zero_grad()

        if self.weighting_mode == "uniform_mean":
            if self.loss_type == "exact_block_count_wnll":
                exact_mean = exact_step_loss_sum / exact_step_weight_sum if exact_step_weight_sum > 0.0 else 0.0
                if self.exact_unweighted_kl_coef > 0.0:
                    kl_mean = exact_kl_step_weighted_sum / exact_kl_step_weight_sum if exact_kl_step_weight_sum > 0.0 else 0.0
                    step_loss = (1.0 - self.exact_unweighted_kl_coef) * exact_mean + self.exact_unweighted_kl_coef * kl_mean
                    append_to_dict(metrics, {"distill/unweighted_fkl_mean": kl_mean})
                else:
                    step_loss = exact_mean
            elif uniform_step_weight_sum > 0.0:
                step_loss = uniform_step_weighted_sum / uniform_step_weight_sum
            else:
                step_loss = 0.0
            append_to_dict(metrics, {"actor/distill_loss": step_loss})
        append_to_dict(metrics, {"distill/did_update": float(did_update_step)})
        self.actor_optimizer.zero_grad()
        return metrics
