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
import time
import warnings

import psutil
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

from recipe.waste_sd.dp_actor import DataParallelWasteSDDistillActor
from recipe.waste_sd.eval_exact_blocks_offline import build_offline_eval_batches, evaluate_exact_blocks_with_models
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.model import update_model_config
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WasteSDAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """FSDP actor+rollout worker with strict waste-aware distillation update path."""
    _CHECKPOINT_CONTENTS_METADATA_FILENAME = "checkpoint_contents.json"

    def _get_exact_blocks_eval_batches(
        self,
        *,
        data_files: list[str],
        batch_size: int,
        max_samples: int,
        max_prompt_length: int,
        max_response_length: int,
        truncation: str,
    ) -> list[dict[str, torch.Tensor | list[str]]]:
        cache_key = (
            tuple(str(path) for path in data_files),
            int(batch_size),
            int(max_samples),
            int(max_prompt_length),
            int(max_response_length),
            str(truncation),
        )
        if not hasattr(self, "_exact_blocks_eval_batches_cache"):
            self._exact_blocks_eval_batches_cache = {}
        cache = self._exact_blocks_eval_batches_cache
        if cache_key not in cache:
            cache[cache_key] = build_offline_eval_batches(
                tokenizer=self.tokenizer,
                data_files=list(data_files),
                batch_size=batch_size,
                max_samples=max_samples,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                truncation=truncation,
            )
        return cache[cache_key]

    def _use_offline_teacher_rollout_data(self) -> bool:
        distill_cfg = self.config.get("distill", {})
        data_mode = str(distill_cfg.get("data_mode", "online_rollout")).lower()
        return data_mode == "offline_teacher_rollout"

    def _normalize_checkpoint_load_contents(self, load_contents: list[str]) -> list[str]:
        normalized = []
        for item in load_contents:
            value = str(item)
            if value not in normalized:
                normalized.append(value)
        if "model" not in normalized:
            raise ValueError(
                "checkpoint load contents must include 'model', "
                f"got {normalized}"
            )
        return normalized

    def _metadata_checkpoint_load_contents(self, local_path: str) -> list[str] | None:
        metadata_path = os.path.join(local_path, self._CHECKPOINT_CONTENTS_METADATA_FILENAME)
        if not os.path.isfile(metadata_path):
            return None
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        contents = metadata.get("load_contents", metadata.get("save_contents"))
        if not isinstance(contents, list):
            raise ValueError(
                f"Malformed checkpoint contents metadata at {metadata_path}: "
                f"expected list-valued load_contents/save_contents, got {contents!r}"
            )
        return self._normalize_checkpoint_load_contents(contents)

    def _maybe_apply_checkpoint_load_contents_from_metadata(self, local_path: str) -> None:
        if (
            local_path is None
            or not hasattr(self, "checkpoint_manager")
            or self.checkpoint_manager is None
            or getattr(self, "_checkpoint_load_contents_override", False)
        ):
            return
        metadata_contents = self._metadata_checkpoint_load_contents(local_path)
        if metadata_contents is None:
            return
        self.checkpoint_manager.checkpoint_load_contents = metadata_contents
        logger.warning("Override checkpoint load_contents from %s to %s", local_path, metadata_contents)

    def _resolve_teacher_forward_backend(self) -> str:
        distill_cfg = self.config.get("distill", {})
        backend = str(distill_cfg.get("teacher_forward_backend", "fsdp_ref")).lower()
        if backend not in {"fsdp_ref", "local_replica"}:
            raise ValueError(
                "distill.teacher_forward_backend must be one of {'fsdp_ref', 'local_replica'}, "
                f"got {backend!r}."
            )
        return backend

    def _resolve_ref_model_path(self) -> str:
        ref_model_path = self.config.model.path
        ref_model = self.config.ref.get("model", None)
        if ref_model is not None:
            ref_model_path = ref_model.get("path", self.config.model.path)
        return ref_model_path

    def _build_local_teacher_replica(self) -> torch.nn.Module:
        """Build one non-FSDP teacher replica per rank for distill forward."""
        ref_model_path = self._resolve_ref_model_path()
        local_path = copy_to_local(ref_model_path, use_shm=self.config.model.get("use_shm", False))
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_liger = self.config.model.get("use_liger", False)

        torch_dtype = self.config.ref.fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        if self.ulysses_sequence_parallel_size > 1 and hasattr(model_config, "vision_config"):
            model_config.vision_config._attn_implementation = "eager"
        if (
            getattr(model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(model_config, "vision_config")
        ):
            model_config.vision_config._attn_implementation = "flash_attention_2"
        if getattr(model_config, "model_type", None) == "kimi_vl":
            model_config.text_config.topk_method = "greedy"

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_architectures = getattr(model_config, "architectures", None) or []
            has_remote_code = hasattr(model_config, "auto_map") and any(
                model_architectures and model_architectures[0] in val for val in model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(k for k, v in model_config.auto_map.items() if model_architectures[0] in v)
                match auto_class:
                    case "AutoModelForVision2Seq":
                        model_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        model_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        model_class = AutoModelForImageTextToText
                    case _:
                        model_class = AutoModel
            else:
                if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    model_class = AutoModelForVision2Seq
                elif type(model_config) in AutoModelForCausalLM._model_mapping.keys():
                    model_class = AutoModelForCausalLM
                elif type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    model_class = AutoModelForImageTextToText
                else:
                    model_class = AutoModel

            teacher_module = model_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

        if use_liger:
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=teacher_module)

        fused_kernel_options = self.config.model.get("fused_kernel_options", None)
        fused_kernels_backend = fused_kernel_options.get("impl_backend", None) if fused_kernel_options else None
        ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
        if ref_tiled_mlp_config is None:
            ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
        ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
        ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)
        apply_monkey_patch(
            model=teacher_module,
            use_remove_padding=self.config.model.get("use_remove_padding", False),
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
            use_fused_kernels=self.config.model.get("use_fused_kernels", False),
            fused_kernels_backend=fused_kernels_backend,
            use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
            use_tiled_mlp=ref_use_tiled_mlp,
            tiled_mlp_shards=ref_tiled_mlp_shards,
        )

        device_name = get_device_name()
        if device_name == "cpu":
            teacher_device = torch.device("cpu")
        else:
            teacher_device = torch.device(device_name, get_device_id())
        teacher_module.to(teacher_device)
        teacher_module.eval()
        teacher_module.requires_grad_(False)
        return teacher_module

    def _resolve_rollout_target_model_path(self) -> str:
        """Resolve the model path used by rollout target model.

        distill.rollout_target:
          - actor: always rollout with actor/student model.
          - local_ref: rollout with ref model when provided, otherwise fallback to actor.
        """
        distill_cfg = self.config.get("distill", {})
        rollout_target = str(distill_cfg.get("rollout_target", "local_ref")).lower()
        require_ref_path = bool(distill_cfg.get("off_policy_require_ref", False))
        model_cfg = self.config.get("model", None)
        actor_model_path = model_cfg.get("path", None) if model_cfg is not None else None

        ref_cfg = self.config.get("ref", None)
        ref_model = ref_cfg.get("model", None) if ref_cfg is not None else None
        ref_model_path = ref_model.get("path", None) if ref_model is not None else None

        if rollout_target not in {"actor", "local_ref"}:
            raise ValueError(
                "distill.rollout_target must be one of {'actor', 'local_ref'}, "
                f"got {rollout_target!r}."
            )
        if not isinstance(actor_model_path, str) or not actor_model_path.strip():
            raise ValueError("actor_rollout_ref.model.path must be a non-empty string.")

        if require_ref_path and (not isinstance(ref_model_path, str) or not ref_model_path.strip()):
            raise ValueError(
                "distill.off_policy_require_ref=true requires actor_rollout_ref.ref.model.path to be set. "
                "Refusing actor-path fallback in teacher off-policy mode."
            )

        if rollout_target == "actor":
            return actor_model_path
        if not isinstance(ref_model_path, str) or not ref_model_path.strip():
            return actor_model_path
        return ref_model_path

    def _build_rollout(self, trust_remote_code=False):
        target_model_path = self._resolve_rollout_target_model_path()
        actor_model_path = self.config.model.path

        if target_model_path == actor_model_path:
            return super()._build_rollout(trust_remote_code=trust_remote_code)

        original_hf_config_path = self.config.model.get("hf_config_path", None)
        original_tokenizer_path = self.config.model.get("tokenizer_path", None)

        logger.warning(
            "waste_sd rollout target model path is overridden from actor model path: "
            "actor=%s, rollout_target=%s",
            actor_model_path,
            target_model_path,
        )

        with open_dict(self.config.model):
            self.config.model.path = target_model_path
            if original_hf_config_path in (None, actor_model_path):
                self.config.model.hf_config_path = target_model_path
            if original_tokenizer_path in (None, actor_model_path):
                self.config.model.tokenizer_path = target_model_path

        try:
            return super()._build_rollout(trust_remote_code=trust_remote_code)
        finally:
            with open_dict(self.config.model):
                self.config.model.path = actor_model_path
                self.config.model.hf_config_path = original_hf_config_path
                self.config.model.tokenizer_path = original_tokenizer_path

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        distill_cfg = self.config.get("distill", {})

        def _as_bool(value) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        block_eval_only = _as_bool(distill_cfg.get("block_eval_only", False))
        skip_ref_init = _as_bool(distill_cfg.get("block_eval_skip_ref_init", block_eval_only))
        if block_eval_only and skip_ref_init and self._is_actor and self._is_rollout and self._is_ref:
            logger.warning("waste_sd block_eval_only enabled: skip ref FSDP initialization to reduce memory usage.")
            self._is_ref = False

        self._waste_sd_rollout_disabled = bool(self._is_rollout and self._use_offline_teacher_rollout_data())
        if self._waste_sd_rollout_disabled:
            logger.warning(
                "waste_sd offline_teacher_rollout enabled: skip rollout engine initialization on actor worker."
            )
            self._is_rollout = False

        super().init_model()
        if block_eval_only:
            logger.warning("waste_sd block_eval_only enabled: skip distill actor initialization.")
            return

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            if not distill_cfg:
                raise ValueError(
                    "Missing `distill` config on WasteSD actor worker. "
                    "This would silently fall back to defaults (gamma=1, debug disabled). "
                    "Ensure top-level `distill` is propagated into `actor_rollout_ref.distill`."
                )
            self.actor = DataParallelWasteSDDistillActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                distill_config=distill_cfg,
            )
            if not hasattr(self, "ref_module_fsdp"):
                raise RuntimeError("waste_sd requires ref_module_fsdp to build teacher forward module.")

            teacher_forward_backend = self._resolve_teacher_forward_backend()
            if teacher_forward_backend == "fsdp_ref":
                self.actor.teacher_module = self.ref_module_fsdp
            else:
                self.actor.teacher_module = self._build_local_teacher_replica()
            self.actor.teacher_forward_backend = teacher_forward_backend
            logger.warning("waste_sd teacher_forward_backend=%s", teacher_forward_backend)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        if getattr(self, "_waste_sd_rollout_disabled", False):
            return True
        return await super().update_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_checkpoint_load_contents(self, load_contents: list[str]):
        """Override checkpoint load contents at runtime.

        This is used by block-eval-only runs to skip optimizer/extra loading and
        reduce resume latency.
        """
        if not hasattr(self, "checkpoint_manager") or self.checkpoint_manager is None:
            return
        normalized = self._normalize_checkpoint_load_contents(load_contents)
        self._checkpoint_load_contents_override = True
        self.checkpoint_manager.checkpoint_load_contents = normalized
        logger.warning("Override checkpoint load_contents to %s", normalized)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        self._maybe_apply_checkpoint_load_contents_from_metadata(local_path)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_distill_update")
    def update_actor_distill(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            with Timer(name="update_actor_distill", logger=None) as timer:
                metrics = self.actor.update_policy_distill(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens,
                delta_time,
                images_seqlens=images_seqlens,
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if hasattr(lr, "item") else lr
            self.actor_lr_scheduler.step()
            output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor_distill", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor_distill", logger=logger)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def evaluate_exact_blocks_offline(self, data: DataProto):
        assert self._is_actor
        if self.actor.teacher_module is None:
            raise RuntimeError("exact_blocks_eval requires actor.teacher_module to be initialized.")

        data_files = data.meta_info.get("data_files", None)
        if not data_files:
            raise ValueError("exact_blocks_eval requires non-empty data_files in meta_info.")
        if isinstance(data_files, str):
            data_files = [data_files]
        data_files = [str(path) for path in data_files]

        batch_size = int(data.meta_info.get("batch_size", 64))
        max_samples = int(data.meta_info.get("max_samples", -1))
        max_prompt_length = int(data.meta_info.get("max_prompt_length", 1024))
        max_response_length = int(data.meta_info.get("max_response_length", 2048))
        truncation = str(data.meta_info.get("truncation", "right"))
        gamma = int(data.meta_info.get("gamma", 8))
        temperature = float(data.meta_info.get("temperature", 1.0))
        disable_tqdm = True

        batches = self._get_exact_blocks_eval_batches(
            data_files=data_files,
            batch_size=batch_size,
            max_samples=max_samples,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            truncation=truncation,
        )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        actor_module = self.actor.actor_module
        teacher_module = self.actor.teacher_module
        actor_was_training = actor_module.training
        teacher_was_training = teacher_module.training
        actor_module.eval()
        teacher_module.eval()

        started = time.time()
        try:
            with self.ulysses_sharding_manager:
                result = evaluate_exact_blocks_with_models(
                    teacher_model=teacher_module,
                    student_model=actor_module,
                    batches=batches,
                    teacher_device=get_torch_device(),
                    student_device=get_torch_device(),
                    gamma=gamma,
                    temperature=temperature,
                    disable_tqdm=disable_tqdm,
                    progress_desc="exact_blocks_eval",
                )
        finally:
            if actor_was_training:
                actor_module.train()
            if teacher_was_training:
                teacher_module.train()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        metrics = {
            f"exact_blocks_eval/{key}": float(value) if isinstance(value, (int, float)) else value
            for key, value in result.items()
        }
        metrics["exact_blocks_eval/runtime_s"] = float(time.time() - started)
        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output
