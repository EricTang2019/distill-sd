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
import os
import uuid
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip


def normalize_exact_blocks_eval_data_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        files: list[str] = []
        for item in value:
            item_str = str(item).strip()
            if item_str:
                files.append(item_str)
        return files
    raise ValueError(f"trainer.exact_blocks_eval_data_file must be a string or list[str], got {type(value)!r}")


def exact_blocks_eval_should_run(*, freq: int, data_files: list[str], global_step: int, is_last_step: bool) -> bool:
    if freq <= 0 or not data_files:
        return False
    return is_last_step or global_step % freq == 0


class WasteSDRayTrainer(RayPPOTrainer):
    """Strict waste-aware SD distillation trainer (SGLang rollout, FSDP actor update)."""
    _LOCAL_VERSION_KEY = "waste_sd_weight_version"
    _CHECKPOINT_CONTENTS_METADATA_FILENAME = "checkpoint_contents.json"
    _FULL_CHECKPOINT_CONTENTS = ["model", "optimizer", "extra"]
    _MODEL_ONLY_CHECKPOINT_CONTENTS = ["model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_cfg = self.config.get("distill", {})
        self.data_mode = str(self.distill_cfg.get("data_mode", "online_rollout")).lower()
        self.strict = bool(self.distill_cfg.get("strict", True))
        self.staleness_max_version_gap = int(self.distill_cfg.get("staleness_max_version_gap", 1))
        self.q_source = str(self.distill_cfg.get("q_source", "local_ref")).lower()
        self._rollout_weight_version = 0
        self._dropped_stale_samples_total = 0
        self._dropped_stale_batches = 0
        self._dropped_empty_response_samples_total = 0
        self._dropped_empty_response_batches = 0
        self._dropped_dp_align_samples_total = 0
        self._dropped_dp_align_batches = 0
        debug_cfg = self.distill_cfg.get("debug", {})
        self._distill_debug_enable = bool(debug_cfg.get("enable", False))
        self._distill_debug_dir = str(debug_cfg.get("dump_dir", "")).strip()
        self._dropped_debug_file = ""
        if self._distill_debug_enable and self._distill_debug_dir:
            os.makedirs(self._distill_debug_dir, exist_ok=True)
            self._dropped_debug_file = os.path.join(self._distill_debug_dir, "dropped_samples.jsonl")

        if self.use_critic:
            raise ValueError("waste_sd trainer only supports actor distillation in v1. Set critic.enable=False.")
        if self.q_source != "local_ref":
            raise ValueError(
                "waste_sd trainer now requires distill.q_source=local_ref. "
                f"Got {self.q_source}."
            )
        if self.data_mode not in {"online_rollout", "offline_teacher_rollout"}:
            raise ValueError(
                "distill.data_mode must be one of {'online_rollout', 'offline_teacher_rollout'}, "
                f"got {self.data_mode!r}."
            )
        self._exact_blocks_eval_data_files = normalize_exact_blocks_eval_data_files(
            self.config.trainer.get("exact_blocks_eval_data_file", None)
        )
        self._exact_blocks_eval_freq = int(self.config.trainer.get("exact_blocks_eval_freq", -1))

    def _stale_filter_enabled(self) -> bool:
        return True

    def _build_compare_config_metrics(self) -> dict[str, float]:
        loss_type = str(self.distill_cfg.get("loss_type", "")).lower()
        weighting_mode = str(self.distill_cfg.get("weighting_mode", "")).lower()
        rollout_target = str(self.distill_cfg.get("rollout_target", "local_ref")).lower()
        teacher_forward_backend = str(self.distill_cfg.get("teacher_forward_backend", "")).lower()
        return {
            "distill_config/data_mode_online_rollout": float(self.data_mode == "online_rollout"),
            "distill_config/data_mode_offline_teacher_rollout": float(self.data_mode == "offline_teacher_rollout"),
            "distill_config/rollout_target_actor": float(rollout_target == "actor"),
            "distill_config/rollout_target_local_ref": float(rollout_target == "local_ref"),
            "distill_config/loss_type_tvd": float(loss_type == "tvd"),
            "distill_config/loss_type_fkl": float(loss_type == "fkl"),
            "distill_config/weighting_mode_uniform_mean": float(weighting_mode == "uniform_mean"),
            "distill_config/weighting_mode_remaining_budget_forward": float(
                weighting_mode == "remaining_budget_forward"
            ),
            "distill_config/teacher_forward_backend_fsdp_ref": float(teacher_forward_backend == "fsdp_ref"),
            "distill_config/teacher_forward_backend_local_replica": float(
                teacher_forward_backend == "local_replica"
            ),
            "distill_config/log_current_batch_theorem_rb_tvd": float(
                bool(self.distill_cfg.get("log_current_batch_theorem_rb_tvd", False))
            ),
            "distill_config/log_post_update_batch_theorem_rb_tvd": float(
                bool(self.distill_cfg.get("log_post_update_batch_theorem_rb_tvd", False))
            ),
            "distill_config/off_policy_require_ref": float(bool(self.distill_cfg.get("off_policy_require_ref", False))),
            "distill_config/stale_filter_enabled": float(self._stale_filter_enabled()),
            "distill_config/staleness_max_version_gap": float(self.staleness_max_version_gap),
            "distill_config/gamma": float(int(self.distill_cfg.get("gamma", 1))),
            "distill_config/exact_blocks_eval_enabled": float(
                self._exact_blocks_eval_freq > 0 and bool(self._exact_blocks_eval_data_files)
            ),
        }

    def _maybe_run_exact_blocks_eval(self, *, logger, global_step: int, is_last_step: bool, temperature: float) -> dict[str, float]:
        if not exact_blocks_eval_should_run(
            freq=self._exact_blocks_eval_freq,
            data_files=self._exact_blocks_eval_data_files,
            global_step=global_step,
            is_last_step=is_last_step,
        ):
            return {}

        eval_request = DataProto(
            meta_info={
                "data_files": list(self._exact_blocks_eval_data_files),
                "batch_size": int(self.config.trainer.get("exact_blocks_eval_batch_size", 64)),
                "max_samples": int(self.config.trainer.get("exact_blocks_eval_max_samples", -1)),
                "max_prompt_length": int(self.config.trainer.get("exact_blocks_eval_max_prompt_length", 1024)),
                "max_response_length": int(self.config.trainer.get("exact_blocks_eval_max_response_length", 2048)),
                "truncation": str(self.config.trainer.get("exact_blocks_eval_truncation", "right")),
                "gamma": int(self.config.distill.get("gamma", 8)),
                "temperature": float(self.config.trainer.get("exact_blocks_eval_temperature", temperature)),
            }
        ).to("cpu")

        eval_output = self.actor_rollout_wg.evaluate_exact_blocks_offline(eval_request)
        return reduce_metrics(eval_output.meta_info["metrics"])

    def _use_offline_teacher_rollout_data(self) -> bool:
        return self.data_mode == "offline_teacher_rollout"

    def _resolve_checkpoint_root_dir(self) -> str:
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
        return checkpoint_root

    @classmethod
    def _normalize_checkpoint_contents(cls, contents: list[str]) -> list[str]:
        normalized = []
        for item in contents:
            value = str(item)
            if value not in normalized:
                normalized.append(value)
        if "model" not in normalized:
            raise ValueError(f"checkpoint contents must include 'model', got {normalized}")
        return normalized

    def _actor_checkpoint_local_path(self, global_step: int) -> str:
        return os.path.join(self._resolve_checkpoint_root_dir(), f"global_step_{int(global_step)}", "actor")

    @classmethod
    def _checkpoint_contents_metadata_path(cls, actor_local_path: str) -> str:
        return os.path.join(actor_local_path, cls._CHECKPOINT_CONTENTS_METADATA_FILENAME)

    @classmethod
    def _write_actor_checkpoint_metadata(cls, actor_local_path: str, contents: list[str]) -> None:
        if not os.path.isdir(actor_local_path):
            return
        normalized = cls._normalize_checkpoint_contents(contents)
        metadata = {
            "version": 1,
            "save_contents": normalized,
            "load_contents": normalized,
        }
        metadata_path = cls._checkpoint_contents_metadata_path(actor_local_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2, sort_keys=True)

    @classmethod
    def _prune_actor_checkpoint_non_model_state(cls, actor_local_path: str) -> None:
        if not os.path.isdir(actor_local_path):
            return
        for filename in os.listdir(actor_local_path):
            if filename.startswith(("optim_world_size_", "extra_state_world_size_")) and filename.endswith(".pt"):
                os.remove(os.path.join(actor_local_path, filename))
        cls._write_actor_checkpoint_metadata(actor_local_path, cls._MODEL_ONLY_CHECKPOINT_CONTENTS)

    def _iter_actor_checkpoint_steps(self) -> list[int]:
        checkpoint_root = self._resolve_checkpoint_root_dir()
        if not os.path.isdir(checkpoint_root):
            return []
        steps = []
        for entry in os.listdir(checkpoint_root):
            if not entry.startswith("global_step_"):
                continue
            try:
                global_step = int(entry.split("global_step_", 1)[1])
            except ValueError:
                continue
            actor_local_path = os.path.join(checkpoint_root, entry, "actor")
            if os.path.isdir(actor_local_path):
                steps.append(global_step)
        return sorted(steps)

    def _finalize_actor_checkpoint_storage_policy(self, latest_step: int) -> None:
        latest_actor_local_path = self._actor_checkpoint_local_path(latest_step)
        self._write_actor_checkpoint_metadata(latest_actor_local_path, self._FULL_CHECKPOINT_CONTENTS)

        for global_step in self._iter_actor_checkpoint_steps():
            if global_step == int(latest_step):
                continue
            self._prune_actor_checkpoint_non_model_state(self._actor_checkpoint_local_path(global_step))

    def _save_checkpoint(self):
        super()._save_checkpoint()
        self._finalize_actor_checkpoint_storage_policy(latest_step=self.global_steps)

    def _extract_prompt_response_ids(self, batch: DataProto) -> tuple[list[list[int]], list[list[int]]]:
        prompts = batch.batch["prompts"].detach().cpu()
        responses = batch.batch["responses"].detach().cpu()
        if "response_mask" in batch.batch:
            response_valid_lens = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
        else:
            response_valid_lens = [responses.shape[-1]] * len(batch)

        if "attention_mask" in batch.batch:
            total_valid_lens = batch.batch["attention_mask"].sum(dim=-1).cpu().tolist()
            prompt_valid_lens = [max(int(total) - int(resp), 0) for total, resp in zip(total_valid_lens, response_valid_lens)]
        else:
            prompt_valid_lens = [prompts.shape[-1]] * len(batch)

        prompt_ids = []
        response_ids = []
        for idx in range(len(batch)):
            prompt_len = int(prompt_valid_lens[idx])
            response_len = int(response_valid_lens[idx])
            prompt_ids.append(prompts[idx, -prompt_len:].tolist() if prompt_len > 0 else [])
            response_ids.append(responses[idx, :response_len].tolist() if response_len > 0 else [])
        return prompt_ids, response_ids

    def _prepare_offline_teacher_rollout_batch(self, batch: DataProto) -> DataProto:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if int(rollout_cfg.n) != 1:
            raise ValueError(
                "distill.data_mode=offline_teacher_rollout requires actor_rollout_ref.rollout.n == 1. "
                f"Got rollout.n={rollout_cfg.n}."
            )
        required_batch_keys = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask"}
        missing = [key for key in required_batch_keys if key not in batch.batch]
        if missing:
            raise ValueError(
                "Offline teacher-rollout batches must provide prompt/response tensors compatible with waste_sd. "
                f"Missing batch keys: {missing}"
            )

        batch.meta_info.update(
            {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "temperature": rollout_cfg.temperature,
                "offline_teacher_rollout": True,
            }
        )
        self._on_rollout_batch_ready(batch)
        return batch

    def _attach_weight_version(self, source_batch: DataProto, gen_batch: DataProto) -> None:
        """Attach rollout version to both source batch and rollout request batch.

        - source_batch keeps a trainer-local numeric version key that survives `batch.union(...)`.
        - gen_batch keeps the legacy `weight_version` key for compatibility with existing agent loops.
        """
        versions = np.full(len(source_batch), self._rollout_weight_version, dtype=np.int64)
        source_batch.non_tensor_batch[self._LOCAL_VERSION_KEY] = versions
        gen_batch.non_tensor_batch["weight_version"] = versions.astype(object)

    def _empty_like(self, batch: DataProto) -> DataProto:
        return batch.select_idxs(np.zeros(len(batch), dtype=bool))

    def _filter_stale_batch(self, batch: DataProto) -> tuple[DataProto, int, DataProto]:
        version_key = self._LOCAL_VERSION_KEY
        versions = batch.non_tensor_batch.get(version_key)
        if versions is None:
            version_key = "weight_version"
            versions = batch.non_tensor_batch.get(version_key)
        if versions is None:
            return batch, 0, self._empty_like(batch)
        uids = batch.non_tensor_batch.get("uid")

        if len(versions) != len(batch):
            if self.strict:
                raise ValueError(
                    f"{version_key} length mismatch: expected {len(batch)}, got {len(versions)}. "
                    "Set distill.strict=false to bypass hard failure."
                )
            return batch, 0, self._empty_like(batch)

        valid = np.ones(len(batch), dtype=bool)
        for i, version in enumerate(versions):
            uid = None
            if uids is not None and i < len(uids):
                uid = uids[i]
            if version is None:
                if self.strict:
                    raise ValueError(
                        f"Missing {version_key} for sample index={i}, uid={uid}. "
                        "This indicates rollout-to-trainer metadata misalignment."
                    )
                valid[i] = False
                continue
            try:
                sample_version = int(version)
            except Exception:
                if self.strict:
                    raise ValueError(
                        f"Invalid {version_key}={version!r} for sample index={i}, uid={uid}. "
                        "Expected an integer rollout version."
                    )
                valid[i] = False
                continue
            gap = self._rollout_weight_version - sample_version
            valid[i] = 0 <= gap <= self.staleness_max_version_gap

        dropped = int((~valid).sum())
        if dropped == 0:
            return batch, 0, self._empty_like(batch)
        dropped_batch = batch.select_idxs(~valid)
        if dropped == len(batch):
            return self._empty_like(batch), dropped, dropped_batch
        return batch.select_idxs(valid), dropped, dropped_batch

    def _filter_empty_response_batch(self, batch: DataProto) -> tuple[DataProto, int, DataProto]:
        if "response_mask" not in batch.batch:
            return batch, 0, self._empty_like(batch)
        valid_token_lens = batch.batch["response_mask"].sum(dim=-1)
        valid = (valid_token_lens > 0).cpu().numpy()
        dropped = int((~valid).sum())
        if dropped == 0:
            return batch, 0, self._empty_like(batch)
        dropped_batch = batch.select_idxs(~valid)
        if dropped == len(batch):
            return self._empty_like(batch), dropped, dropped_batch
        return batch.select_idxs(valid), dropped, dropped_batch

    def _align_batch_for_dp_update(self, batch: DataProto) -> tuple[DataProto, int, DataProto]:
        dp_size = int(getattr(self.actor_rollout_wg, "world_size", 1))
        if dp_size <= 1:
            return batch, 0, self._empty_like(batch)
        batch_size = len(batch)
        if batch_size == 0:
            return batch, 0, self._empty_like(batch)
        remainder = batch_size % dp_size
        if remainder == 0:
            return batch, 0, self._empty_like(batch)
        keep = batch_size - remainder
        if keep <= 0:
            return self._empty_like(batch), batch_size, batch
        keep_idx = np.arange(keep, dtype=np.int64)
        drop_idx = np.arange(keep, batch_size, dtype=np.int64)
        return batch.select_idxs(keep_idx), remainder, batch.select_idxs(drop_idx)

    def _dump_dropped_samples(self, *, step: int, reason: str, dropped_batch: DataProto) -> None:
        if not self._distill_debug_enable or not self._dropped_debug_file or len(dropped_batch) == 0:
            return

        prompts_text = []
        responses_text = []
        if "prompts" in dropped_batch.batch:
            prompts_text = self.tokenizer.batch_decode(dropped_batch.batch["prompts"], skip_special_tokens=True)
        if "responses" in dropped_batch.batch:
            responses_text = self.tokenizer.batch_decode(dropped_batch.batch["responses"], skip_special_tokens=True)

        uids = dropped_batch.non_tensor_batch.get("uid")
        accept_lens = dropped_batch.non_tensor_batch.get("spec_accept_lens")
        versions = dropped_batch.non_tensor_batch.get(self._LOCAL_VERSION_KEY)
        if versions is None:
            versions = dropped_batch.non_tensor_batch.get("weight_version")
        response_valid_lens = None
        if "response_mask" in dropped_batch.batch:
            response_valid_lens = dropped_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()

        with open(self._dropped_debug_file, "a", encoding="utf-8") as f:
            for i in range(len(dropped_batch)):
                uid = None
                if uids is not None and i < len(uids):
                    uid = None if uids[i] is None else str(uids[i])
                sample_accept_lens = None
                if accept_lens is not None and i < len(accept_lens):
                    item = accept_lens[i]
                    if item is not None:
                        if hasattr(item, "tolist"):
                            item = item.tolist()
                        sample_accept_lens = [int(v) for v in item]
                version = None
                if versions is not None and i < len(versions):
                    try:
                        version = int(versions[i])
                    except Exception:
                        version = None if versions[i] is None else str(versions[i])
                record = {
                    "step": int(step),
                    "reason": reason,
                    "uid": uid,
                    "prompt": prompts_text[i] if i < len(prompts_text) else None,
                    "response": responses_text[i] if i < len(responses_text) else None,
                    "response_valid_len": int(response_valid_lens[i]) if response_valid_lens is not None else None,
                    "spec_accept_lens": sample_accept_lens,
                    "rollout_weight_version": version,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _to_python_list(values):
        if values is None:
            return []
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    def _log_waste_sd_rollout_data(self, batch: DataProto, timing_raw: dict, rollout_data_dir: str) -> None:
        """Dump rollout generations and SD-specific metadata as JSONL."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

            n = len(inputs)
            scores = [0.0] * n
            gts = [None] * n

            extra: dict[str, list] = {}

            if "uid" in batch.non_tensor_batch:
                raw_uids = self._to_python_list(batch.non_tensor_batch["uid"])
                extra["uid"] = [None if x is None else str(x) for x in raw_uids]

            local_versions = batch.non_tensor_batch.get(self._LOCAL_VERSION_KEY, None)
            if local_versions is not None:
                raw_versions = self._to_python_list(local_versions)
                extra["waste_sd_weight_version"] = [None if x is None else int(x) for x in raw_versions]

            rollout_versions = batch.non_tensor_batch.get("weight_version", None)
            if rollout_versions is not None:
                raw_versions = self._to_python_list(rollout_versions)
                normalized_versions = []
                for x in raw_versions:
                    if x is None:
                        normalized_versions.append(None)
                        continue
                    try:
                        normalized_versions.append(int(x))
                    except Exception:
                        normalized_versions.append(str(x))
                extra["rollout_weight_version"] = normalized_versions

            if "spec_accept_lens" in batch.non_tensor_batch:
                raw_accept = self._to_python_list(batch.non_tensor_batch["spec_accept_lens"])
                normalized_accept = []
                for item in raw_accept:
                    if item is None:
                        normalized_accept.append(None)
                        continue
                    if hasattr(item, "tolist"):
                        item = item.tolist()
                    normalized_accept.append([int(v) for v in item])
                extra["spec_accept_lens"] = normalized_accept

            for key in (
                "spec_verify_ct",
                "spec_accepted_tokens",
                "target_forward_total_calls",
                "target_forward_verify_true_calls",
                "target_forward_verify_false_calls",
                "target_forward_verify_true_called",
                "verify_hook_calls",
            ):
                if key in batch.non_tensor_batch:
                    raw_vals = self._to_python_list(batch.non_tensor_batch[key])
                    if key.endswith("_called"):
                        extra[key] = [None if x is None else bool(x) for x in raw_vals]
                    else:
                        extra[key] = [None if x is None else int(x) for x in raw_vals]

            if "response_mask" in batch.batch:
                extra["response_valid_len"] = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
            prompt_ids, response_ids = self._extract_prompt_response_ids(batch)
            extra["prompt"] = inputs
            extra["response"] = outputs
            extra["prompt_ids"] = prompt_ids
            extra["response_ids"] = response_ids

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=gts,
                scores=scores,
                reward_extra_infos_dict=extra,
                dump_path=rollout_data_dir,
            )

    def _initialize_rollout_state(self) -> None:
        """Load rollout weights from actor and reset rollout version tracking."""
        if self._use_offline_teacher_rollout_data():
            self._rollout_weight_version = 0
            return
        self.checkpoint_manager.update_weights()
        self._rollout_weight_version = 0

    def _on_rollout_batch_ready(self, batch: DataProto) -> None:
        """Hook for rollout post-processing before actor update."""
        return

    def _maybe_update_rollout_weights(self, did_actor_update: bool, timing_raw: dict) -> None:
        """Synchronize actor weights to rollout if the actor was updated."""
        if self._use_offline_teacher_rollout_data():
            return
        if not did_actor_update:
            return
        with marked_timer("update_weights", timing_raw, color="purple"):
            self.checkpoint_manager.update_weights()
            self._rollout_weight_version += 1

    def init_workers(self):
        if not self._use_offline_teacher_rollout_data():
            return super().init_workers()

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if not self.hybrid_engine:
            raise NotImplementedError

        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        if self.use_critic:
            raise ValueError("waste_sd offline_teacher_rollout does not support critic workers.")
        if self.use_reference_policy and not self.ref_in_actor:
            raise ValueError(
                "waste_sd offline_teacher_rollout expects reference policy to be colocated in actor worker."
            )
        if self.use_reward_loop or self.use_rm:
            raise ValueError("waste_sd offline_teacher_rollout does not support reward loop workers.")

        all_wg = {}
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                worker_nsight_options = OmegaConf.select(
                    self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options"
                )
                assert worker_nsight_options is not None, (
                    "worker_nsight_options must be set when using nsys with profile_steps"
                )
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_nsight_options)

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = False
        self.async_rollout_manager = None
        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=[],
        )

    def _compute_block_metrics(self, batch: DataProto) -> dict[str, float]:
        total_samples = len(batch)
        traced_samples = 0
        total_blocks = 0
        total_accepted_tokens = 0
        total_response_valid_len = 0

        raw_verify_ct = batch.non_tensor_batch.get("spec_verify_ct", None)
        raw_accepted_tokens = batch.non_tensor_batch.get("spec_accepted_tokens", None)
        raw_accept = batch.non_tensor_batch.get("spec_accept_lens", None)
        response_valid_lens: list[int] = []
        if "response_mask" in batch.batch:
            response_valid_lens = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
            total_response_valid_len = int(sum(response_valid_lens))
        else:
            response_valid_lens = [0] * total_samples

        verify_ct_values = self._to_python_list(raw_verify_ct) if raw_verify_ct is not None else [None] * total_samples
        accepted_token_values = (
            self._to_python_list(raw_accepted_tokens) if raw_accepted_tokens is not None else [None] * total_samples
        )
        accept_trace_values = self._to_python_list(raw_accept) if raw_accept is not None else [None] * total_samples

        for idx in range(total_samples):
            sample_verify_ct = verify_ct_values[idx] if idx < len(verify_ct_values) else None
            sample_accepted_tokens = accepted_token_values[idx] if idx < len(accepted_token_values) else None
            sample_accept_lens = accept_trace_values[idx] if idx < len(accept_trace_values) else None
            valid_tokens = int(response_valid_lens[idx]) if idx < len(response_valid_lens) else 0

            has_raw_counters = sample_verify_ct is not None and sample_accepted_tokens is not None
            if has_raw_counters:
                traced_samples += 1
                total_blocks += int(sample_verify_ct)
                total_accepted_tokens += int(sample_accepted_tokens)
                continue

            if sample_accept_lens is not None:
                if hasattr(sample_accept_lens, "tolist"):
                    sample_accept_lens = sample_accept_lens.tolist()
                accepted_lens = []
                for value in sample_accept_lens:
                    try:
                        accepted_lens.append(int(value))
                    except Exception:
                        continue
                traced_samples += 1
                total_blocks += len(accepted_lens)
                total_accepted_tokens += sum(max(v - 1, 0) for v in accepted_lens)
                continue

            if self.strict and valid_tokens > 0:
                raise ValueError(
                    "Block eval strict mode requires per-sample speculative metadata. "
                    f"Missing spec_verify_ct/spec_accepted_tokens and spec_accept_lens for sample index={idx}, "
                    f"valid_tokens={valid_tokens}."
                )

        avg_blocks_per_sample = float(total_blocks / total_samples) if total_samples > 0 else 0.0
        avg_blocks_per_traced = float(total_blocks / traced_samples) if traced_samples > 0 else 0.0

        return {
            "block_eval/step_samples": float(total_samples),
            "block_eval/step_traced_samples": float(traced_samples),
            "block_eval/step_total_blocks": float(total_blocks),
            "block_eval/step_total_accepted_tokens": float(total_accepted_tokens),
            "block_eval/step_total_response_valid_len": float(total_response_valid_len),
            "block_eval/step_avg_blocks_per_sample": avg_blocks_per_sample,
            "block_eval/step_avg_blocks_per_traced_sample": avg_blocks_per_traced,
        }

    def _run_block_eval_only(self, logger, *, reset_global_step: bool = True) -> None:
        if self._use_offline_teacher_rollout_data():
            raise ValueError("trainer.block_eval_only is not supported with distill.data_mode=offline_teacher_rollout.")
        split = str(self.config.trainer.get("block_eval_split", "val")).lower()
        if split == "val":
            dataloader = self.val_dataloader
        elif split == "train":
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"trainer.block_eval_split must be 'val' or 'train', got {split!r}")

        if dataloader is None:
            raise ValueError(f"Block eval split={split!r} has no dataloader.")

        max_steps_cfg = self.config.trainer.get("block_eval_max_steps", None)
        if max_steps_cfg is not None:
            max_steps = int(max_steps_cfg)
        else:
            try:
                max_steps = int(len(dataloader))
            except TypeError:
                max_steps = int(self.total_training_steps)
        if max_steps <= 0:
            raise ValueError(f"trainer.block_eval_max_steps must be > 0, got {max_steps}")

        progress_bar = tqdm(total=max_steps, initial=0, desc=f"Block Eval ({split})")

        total_samples = 0.0
        total_traced_samples = 0.0
        total_blocks = 0.0
        total_accepted_tokens = 0.0
        total_response_valid_len = 0.0
        if reset_global_step:
            self.global_steps = 1

        for step_idx, batch_dict in enumerate(dataloader):
            if step_idx >= max_steps:
                break
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

            timing_raw: dict[str, float] = {}
            metrics: dict[str, float] = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            if "uid" not in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

            rollout_cfg = self.config.actor_rollout_ref.rollout
            batch.meta_info.update(
                {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": rollout_cfg.val_kwargs.do_sample,
                    "validate": True,
                    "top_p": rollout_cfg.val_kwargs.top_p,
                    "top_k": rollout_cfg.val_kwargs.top_k,
                    "temperature": rollout_cfg.val_kwargs.temperature,
                }
            )

            gen_batch = self._get_gen_batch(batch)
            self._attach_weight_version(batch, gen_batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=rollout_cfg.val_kwargs.n, interleave=True)

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    # Eval-only block tracing must avoid actor->rollout sync paths.
                    # actor_rollout_wg.generate_sequences() can trigger rollout.update_weights(),
                    # so require the async manager here.
                    if not self.async_rollout_mode:
                        raise RuntimeError(
                            "block_eval_only requires async_rollout_mode=True to avoid rollout "
                            "weight sync/update in actor_rollout_wg.generate_sequences()."
                        )
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                    gen_batch_output.meta_info.pop("timing", None)

                batch = batch.repeat(repeat_times=rollout_cfg.val_kwargs.n, interleave=True)
                batch = batch.union(gen_batch_output)
                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                self._on_rollout_batch_ready(batch)

                rollout_data_dir = self.config.trainer.get("block_eval_rollout_data_dir", None)
                if rollout_data_dir is None:
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_waste_sd_rollout_data(batch=batch, timing_raw=timing_raw, rollout_data_dir=rollout_data_dir)

            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            step_block_metrics = self._compute_block_metrics(batch)
            metrics.update(step_block_metrics)
            metrics["distill/rollout_weight_version"] = float(self._rollout_weight_version)
            metrics["training/global_step"] = self.global_steps
            metrics["training/epoch"] = 0.0
            metrics["block_eval/enabled"] = 1.0

            total_samples += step_block_metrics["block_eval/step_samples"]
            total_traced_samples += step_block_metrics["block_eval/step_traced_samples"]
            total_blocks += step_block_metrics["block_eval/step_total_blocks"]
            total_accepted_tokens += step_block_metrics["block_eval/step_total_accepted_tokens"]
            total_response_valid_len += step_block_metrics["block_eval/step_total_response_valid_len"]

            token_count = int(sum(batch.meta_info.get("global_token_num", [])))
            if token_count > 0:
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            else:
                metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
                metrics["block_eval/skip_timing_per_token_on_empty_batch"] = 1.0
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1

        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)

        avg_blocks_per_sample = float(total_blocks / total_samples) if total_samples > 0 else 0.0
        avg_blocks_per_traced = float(total_blocks / total_traced_samples) if total_traced_samples > 0 else 0.0
        summary_metrics = {
            "block_eval/total_samples": total_samples,
            "block_eval/total_traced_samples": total_traced_samples,
            "block_eval/total_blocks": total_blocks,
            "block_eval/total_accepted_tokens": total_accepted_tokens,
            "block_eval/total_response_valid_len": total_response_valid_len,
            "block_eval/avg_blocks_per_sample": avg_blocks_per_sample,
            "block_eval/avg_blocks_per_traced_sample": avg_blocks_per_traced,
            "block_eval/enabled": 1.0,
        }
        logger.log(data=summary_metrics, step=self.global_steps)
        pprint(summary_metrics)
        progress_bar.close()

    def _fit_block_eval_only(self, logger) -> None:
        """Run block-eval in an isolated path without rollout weight synchronization."""
        # Eval-only path does not need optimizer/lr-scheduler/rng restore.
        # Loading model-only significantly reduces resume latency.
        self.actor_rollout_wg.set_checkpoint_load_contents(["model"])

        # Load checkpoint state, but do not call checkpoint_manager.update_weights().
        self._load_checkpoint()
        self._rollout_weight_version = 0

        self._run_block_eval_only(logger=logger)

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        block_eval_only_cfg = self.config.trainer.get("block_eval_only", False)
        if isinstance(block_eval_only_cfg, str):
            block_eval_only = block_eval_only_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            block_eval_only = bool(block_eval_only_cfg)

        if block_eval_only and self._use_offline_teacher_rollout_data():
            raise ValueError("trainer.block_eval_only requires online rollout; offline teacher-rollout data mode is incompatible.")

        if block_eval_only:
            self._fit_block_eval_only(logger=logger)
            return

        # Training path (unchanged semantics): load checkpoint then sync actor->rollout.
        self._load_checkpoint()
        self._initialize_rollout_state()

        current_epoch = self.global_steps // len(self.train_dataloader)

        config_metrics = self._build_compare_config_metrics()
        config_metrics["training/global_step"] = self.global_steps
        config_metrics["training/epoch"] = float(current_epoch)
        logger.log(data=config_metrics, step=self.global_steps)

        if self.config.trainer.get("val_only", False):
            pprint("waste_sd trainer has no reward-based validation in v1. Skip val_only.")
            return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if "uid" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

                rollout_cfg = self.config.actor_rollout_ref.rollout
                batch.meta_info["temperature"] = rollout_cfg.temperature

                with marked_timer("step", timing_raw):
                    if self._use_offline_teacher_rollout_data():
                        timing_raw["gen"] = 0.0
                        batch = self._prepare_offline_teacher_rollout_batch(batch)
                    else:
                        gen_batch = self._get_gen_batch(batch)
                        self._attach_weight_version(batch, gen_batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch_output = gen_batch.repeat(repeat_times=rollout_cfg.n, interleave=True)
                        with marked_timer("gen", timing_raw, color="red"):
                            if self.async_rollout_mode:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        batch = batch.repeat(repeat_times=rollout_cfg.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        self._on_rollout_batch_ready(batch)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_waste_sd_rollout_data(batch=batch, timing_raw=timing_raw, rollout_data_dir=rollout_data_dir)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    filtered_batch, dropped_stale, dropped_stale_batch = self._filter_stale_batch(batch)
                    if dropped_stale > 0:
                        self._dropped_stale_samples_total += dropped_stale
                        self._dropped_stale_batches += 1
                        self._dump_dropped_samples(
                            step=self.global_steps,
                            reason="stale_version",
                            dropped_batch=dropped_stale_batch,
                        )
                    metrics["distill/dropped_stale_samples_step"] = float(dropped_stale)

                    filtered_batch, dropped_empty_response, dropped_empty_batch = self._filter_empty_response_batch(filtered_batch)
                    if dropped_empty_response > 0:
                        self._dropped_empty_response_samples_total += dropped_empty_response
                        self._dropped_empty_response_batches += 1
                        self._dump_dropped_samples(
                            step=self.global_steps,
                            reason="empty_response",
                            dropped_batch=dropped_empty_batch,
                        )
                    metrics["distill/dropped_empty_response_samples_step"] = float(dropped_empty_response)

                    filtered_batch, dropped_dp_align, dropped_dp_align_batch = self._align_batch_for_dp_update(filtered_batch)
                    if dropped_dp_align > 0:
                        self._dropped_dp_align_samples_total += dropped_dp_align
                        self._dropped_dp_align_batches += 1
                        self._dump_dropped_samples(
                            step=self.global_steps,
                            reason="dp_align_remainder",
                            dropped_batch=dropped_dp_align_batch,
                        )
                    metrics["distill/dropped_dp_align_samples_step"] = float(dropped_dp_align)

                    if len(filtered_batch) > 0:
                        filtered_batch.meta_info["global_token_num"] = (
                            torch.sum(filtered_batch.batch["attention_mask"], dim=-1).tolist()
                        )
                        filtered_batch.meta_info["global_steps"] = self.global_steps
                        filtered_batch.meta_info["temperature"] = rollout_cfg.temperature
                        filtered_batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor_distill(filtered_batch)
                        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_metrics)
                        did_actor_update = float(actor_metrics.get("distill/did_update", 0.0)) > 0.0
                    else:
                        filtered_batch = batch.select_idxs(np.zeros(len(batch), dtype=bool))
                        filtered_batch.meta_info["global_token_num"] = [0]
                        metrics["distill/skipped_step_all_filtered"] = 1.0
                        did_actor_update = False

                    self._maybe_update_rollout_weights(did_actor_update=did_actor_update, timing_raw=timing_raw)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["distill/rollout_weight_version"] = float(self._rollout_weight_version)
                metrics["distill/dropped_stale_samples_total"] = float(self._dropped_stale_samples_total)
                metrics["distill/dropped_stale_batches"] = float(self._dropped_stale_batches)
                metrics["distill/dropped_empty_response_samples_total"] = float(self._dropped_empty_response_samples_total)
                metrics["distill/dropped_empty_response_batches"] = float(self._dropped_empty_response_batches)
                metrics["distill/dropped_dp_align_samples_total"] = float(self._dropped_dp_align_samples_total)
                metrics["distill/dropped_dp_align_batches"] = float(self._dropped_dp_align_batches)

                n_gpus = self.resource_pool_manager.get_n_gpus()
                token_count = int(sum(filtered_batch.meta_info.get("global_token_num", [])))
                metrics.update(
                    self._maybe_run_exact_blocks_eval(
                        logger=logger,
                        global_step=self.global_steps,
                        is_last_step=is_last_step,
                        temperature=float(rollout_cfg.temperature),
                    )
                )
                if token_count > 0:
                    metrics.update(compute_timing_metrics(batch=filtered_batch, timing_raw=timing_raw))
                else:
                    metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
                    metrics["distill/skip_timing_per_token_on_empty_batch"] = 1.0
                metrics.update(compute_throughout_metrics(batch=filtered_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    block_eval_after_train_cfg = self.config.trainer.get("block_eval_after_train", False)
                    if isinstance(block_eval_after_train_cfg, str):
                        block_eval_after_train = block_eval_after_train_cfg.strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                            "on",
                        }
                    else:
                        block_eval_after_train = bool(block_eval_after_train_cfg)
                    if block_eval_after_train:
                        self._run_block_eval_only(logger=logger, reset_global_step=False)
                    progress_bar.close()
                    return

        progress_bar.close()
