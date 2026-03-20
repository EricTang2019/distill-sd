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

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.waste_sd.ray_trainer import WasteSDRayTrainer
from verl.single_controller.ray import RayResourcePool, ResourcePoolManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available


def _get_positive_batch_sizes(config) -> list[int]:
    batch_sizes: list[int] = []
    data_cfg = config.get("data", {})
    for key in ("train_batch_size", "val_batch_size"):
        value = data_cfg.get(key, None)
        if value is None:
            continue
        batch_size = int(value)
        if batch_size > 0:
            batch_sizes.append(batch_size)
    return batch_sizes


def _normalize_rollout_agent_num_workers(config) -> int | None:
    rollout_cfg = config.actor_rollout_ref.get("rollout", None)
    if rollout_cfg is None:
        return None
    agent_cfg = rollout_cfg.get("agent", None)
    if agent_cfg is None or agent_cfg.get("num_workers", None) is None:
        return None

    requested = int(agent_cfg.num_workers)
    if requested <= 1:
        return requested

    batch_sizes = _get_positive_batch_sizes(config)
    if not batch_sizes:
        return requested

    max_candidate = min(requested, min(batch_sizes))
    normalized = 1
    for candidate in range(max_candidate, 0, -1):
        if all(batch_size % candidate == 0 for batch_size in batch_sizes):
            normalized = candidate
            break

    if normalized != requested:
        with open_dict(agent_cfg):
            agent_cfg.num_workers = normalized
        print(
            "Normalized actor_rollout_ref.rollout.agent.num_workers from "
            f"{requested} to {normalized} to evenly divide batch sizes {batch_sizes}."
        )
    return normalized


@hydra.main(config_path="config", config_name="waste_sd_trainer", version_base=None)
def main(config):
    run_waste_sd(config)


def run_waste_sd(
    config,
    *,
    task_runner_cls=None,
    enable_patch_env: bool = True,
    patch_env_override: dict[str, str] | None = None,
    force_patch_env: bool = False,
):
    distill_cfg = config.get("distill", {})
    block_eval_only_cfg = config.trainer.get("block_eval_only", False)
    if isinstance(block_eval_only_cfg, str):
        block_eval_only = block_eval_only_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        block_eval_only = bool(block_eval_only_cfg)
    q_source = str(distill_cfg.get("q_source", "local_ref")).lower()
    strict_mode = bool(distill_cfg.get("strict", True))
    teacher_forward_backend = str(distill_cfg.get("teacher_forward_backend", "fsdp_ref")).lower()
    if q_source != "local_ref":
        raise ValueError(f"waste_sd now requires distill.q_source=local_ref, got {q_source}.")
    if teacher_forward_backend not in {"fsdp_ref", "local_replica"}:
        raise ValueError(
            "waste_sd requires distill.teacher_forward_backend in {'fsdp_ref', 'local_replica'}, "
            f"got {teacher_forward_backend!r}."
        )

    # Worker processes are initialized from `config.actor_rollout_ref` only.
    # Propagate top-level distill config into that subtree to avoid falling back
    # to actor-side defaults (gamma=1, debug disabled) when distill is configured.
    with open_dict(config.actor_rollout_ref):
        worker_distill_cfg = OmegaConf.create(OmegaConf.to_container(distill_cfg, resolve=False))
        worker_distill_cfg.block_eval_only = bool(block_eval_only)
        # Eval-only path can skip FSDP ref init to reduce memory pressure.
        # Users may explicitly override this via `distill.block_eval_skip_ref_init=false`.
        if "block_eval_skip_ref_init" not in worker_distill_cfg:
            worker_distill_cfg.block_eval_skip_ref_init = bool(block_eval_only)
        config.actor_rollout_ref.distill = worker_distill_cfg

    with open_dict(config.actor_rollout_ref):
        rollout_cfg = config.actor_rollout_ref.get("rollout", None)
        if rollout_cfg is not None:
            agent_cfg = rollout_cfg.get("agent", None)
            if agent_cfg is None:
                rollout_cfg.agent = OmegaConf.create({})
                agent_cfg = rollout_cfg.agent
            if not agent_cfg.get("agent_loop_manager_class", None):
                agent_cfg.agent_loop_manager_class = (
                    "recipe.waste_sd.agent_loop.WasteSDAgentLoopManager"
                )
    _normalize_rollout_agent_num_workers(config)

    patch_env: dict[str, str] = {}
    if enable_patch_env:
        patch_env = {
            "VERL_SGLANG_WASTE_SD_PATCH": os.getenv("VERL_SGLANG_WASTE_SD_PATCH", "1"),
            "VERL_SGLANG_STANDALONE_SYNC_MODE": os.getenv(
                "VERL_SGLANG_STANDALONE_SYNC_MODE", "draft_only"
            ),
            # Source of truth is config.distill.strict; do not let a stale shell env override it.
            "VERL_SGLANG_WASTE_SD_STRICT": "1" if strict_mode else "0",
        }
    if patch_env_override:
        patch_env.update({str(k): str(v) for k, v in patch_env_override.items()})

    if patch_env:
        # Keep local process and Ray workers consistent for runtime patch behavior.
        for key, value in patch_env.items():
            if force_patch_env or key == "VERL_SGLANG_WASTE_SD_STRICT":
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        runtime_env_env_vars = {str(k): str(v) for k, v in dict(runtime_env.get("env_vars", {})).items()}
        for key, value in patch_env.items():
            if force_patch_env or key == "VERL_SGLANG_WASTE_SD_STRICT":
                runtime_env_env_vars[key] = str(value)
            else:
                runtime_env_env_vars.setdefault(key, str(value))
        # Keep dynamic linker environment consistent in Ray workers/subprocesses.
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
            if key in os.environ:
                runtime_env_env_vars[key] = str(os.environ[key])
        runtime_env["env_vars"] = runtime_env_env_vars
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_cls is None:
        task_runner_cls = WasteSDTaskRunner
    task_runner_class = ray.remote(num_cpus=1)(task_runner_cls)
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class WasteSDTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        self.trainer_cls = WasteSDRayTrainer

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("waste_sd v1 only supports FSDP/FSDP2 actor strategy.")

        from recipe.waste_sd.fsdp_workers import WasteSDAsyncActorRolloutRefWorker

        actor_rollout_cls = WasteSDAsyncActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        # local_ref distillation needs actor and ref in the same worker process.
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRolloutRef] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        class _WasteSDResourcePoolManager(ResourcePoolManager):
            def __init__(self, *args, max_colocate_count: int = 1, **kwargs):
                super().__init__(*args, **kwargs)
                self._max_colocate_count = max_colocate_count

            def create_resource_pool(self):
                for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
                    self.resource_pool_dict[resource_pool_name] = RayResourcePool(
                        process_on_nodes=process_on_nodes,
                        use_gpu=True,
                        max_colocate_count=self._max_colocate_count,
                        name_prefix=resource_pool_name,
                    )
                self._check_resource_available()

        max_colocate_count = int(config.trainer.get("ray_max_colocate_count", 1))
        if max_colocate_count < 1:
            raise ValueError(f"trainer.ray_max_colocate_count must be >= 1, got {max_colocate_count}")
        resource_pool_manager = _WasteSDResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            max_colocate_count=max_colocate_count,
        )
        return resource_pool_manager

    def run(self, config):
        from verl.utils.fs import copy_to_local

        print(f"WasteSDTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        _, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        validate_config(config=config, use_reference_policy=False, use_critic=False)

        tokenizer_path = (
            config.actor_rollout_ref.model.get("tokenizer_path", None) or config.actor_rollout_ref.model.path
        )
        local_path = copy_to_local(tokenizer_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = self.trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
