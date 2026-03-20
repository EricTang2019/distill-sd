# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, furunding
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from recipe.gkd.ray_trainer import OnPolicyDistillTrainer

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


@hydra.main(config_path="config", config_name="on_policy_distill_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_on_policy_distill(config)


def _apply_legacy_config_compat(config) -> None:
    """Map legacy GKD config fields to current verl config schema."""

    actor_cfg = config.actor_rollout_ref.actor
    model_cfg = config.actor_rollout_ref.model
    rollout_cfg = config.actor_rollout_ref.rollout

    with open_dict(config):
        # New verl expects model config to be instantiable dataclass config.
        if OmegaConf.select(config, "actor_rollout_ref.model._target_") is None:
            model_cfg._target_ = "verl.workers.config.model.HFModelConfig"

        # Router replay became a required structured sub-config in newer verl.
        if OmegaConf.select(config, "actor_rollout_ref.actor.router_replay") is None:
            actor_cfg.router_replay = OmegaConf.create(
                {
                    "mode": "disabled",
                    "record_file": None,
                    "replay_file": None,
                }
            )

        # Legacy actor knobs used by recipe/gkd custom actor implementation.
        micro_batch_size = OmegaConf.select(config, "actor_rollout_ref.actor.micro_batch_size")
        max_token_len = OmegaConf.select(config, "actor_rollout_ref.actor.max_token_len")

        # Backfill fields read by the upstream ActorRolloutRefWorker.
        if OmegaConf.select(config, "actor_rollout_ref.actor.ppo_mini_batch_size") is None:
            actor_cfg.ppo_mini_batch_size = int(micro_batch_size) if micro_batch_size is not None else 1
        if (
            OmegaConf.select(config, "actor_rollout_ref.actor.ppo_micro_batch_size") is None
            and OmegaConf.select(config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu") is None
            and micro_batch_size is not None
        ):
            actor_cfg.ppo_micro_batch_size_per_gpu = int(micro_batch_size)
        if OmegaConf.select(config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu") is None:
            actor_cfg.ppo_max_token_len_per_gpu = int(max_token_len) if max_token_len is not None else 16384

        # Rollout n is referenced in upstream worker normalization.
        if OmegaConf.select(config, "actor_rollout_ref.rollout.n") is None:
            rollout_cfg.n = 1
        if OmegaConf.select(config, "actor_rollout_ref.actor.rollout_n") is None:
            actor_cfg.rollout_n = int(rollout_cfg.n)

        # New rollout config no longer accepts sync mode.
        if OmegaConf.select(config, "actor_rollout_ref.rollout.mode") == "sync":
            rollout_cfg.mode = "async"

        # Legacy GKD rollout load_format value is not recognized by recent vLLM.
        # Preserve old semantics by mapping to vLLM-supported "dummy".
        if OmegaConf.select(config, "actor_rollout_ref.rollout.load_format") == "dummy_megatron":
            rollout_cfg.load_format = "dummy"

        # Backfill rollout subtrees that new async agent-loop/server paths access directly.
        rollout_defaults = {
            "agent": {
                "num_workers": 8,
                "custom_async_server": {"path": None, "name": None},
            },
            "trace": {
                "backend": None,
                "token2text": False,
                "max_samples_per_step_per_worker": None,
            },
            "server": {
                "timeout": 60.0,
                "max_attempts": 3,
                "retry_delay": 2.0,
                "max_connections": 1000,
                "max_start_wait_time": 300.0,
            },
            "multi_turn": {
                "enable": False,
                "format": "hermes",
                "max_assistant_turns": None,
                "tool_config_path": None,
                "max_user_turns": None,
                "max_parallel_calls": 1,
                "max_tool_response_length": 256,
                "tool_response_truncate_side": "middle",
                "interaction_config_path": None,
                "use_inference_chat_template": False,
                "tokenization_sanity_check_mode": "strict",
            },
            "engine_kwargs": {
                "vllm": {"disable_mm_preprocessor_cache": False, "swap_space": None},
                "sglang": {"attention_backend": None},
            },
            "val_kwargs": {
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0,
                "n": 1,
                "do_sample": False,
            },
            "layer_name_map": {"qkv_layer_name": "qkv", "gate_proj_layer_name": "gate_up"},
        }
        for key, default_val in rollout_defaults.items():
            if OmegaConf.select(config, f"actor_rollout_ref.rollout.{key}") is None:
                rollout_cfg[key] = OmegaConf.create(default_val)

        # Migrate legacy top-level bucket setting to checkpoint_engine subtree.
        legacy_bucket_mb = OmegaConf.select(config, "actor_rollout_ref.rollout.update_weights_bucket_megabytes")
        if OmegaConf.select(config, "actor_rollout_ref.rollout.checkpoint_engine") is None:
            rollout_cfg.checkpoint_engine = OmegaConf.create({})
        if (
            OmegaConf.select(config, "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes")
            is None
        ):
            rollout_cfg.checkpoint_engine.update_weights_bucket_megabytes = (
                int(legacy_bucket_mb) if legacy_bucket_mb is not None else 2048
            )
        if OmegaConf.select(config, "actor_rollout_ref.rollout.checkpoint_engine.backend") is None:
            rollout_cfg.checkpoint_engine.backend = "naive"
        if OmegaConf.select(config, "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs") is None:
            rollout_cfg.checkpoint_engine.engine_kwargs = OmegaConf.create({})
        if "update_weights_bucket_megabytes" in rollout_cfg:
            del rollout_cfg["update_weights_bucket_megabytes"]

        # New agent-loop path accesses rollout.prometheus directly.
        # Legacy GKD config doesn't define this subtree, so provide safe defaults.
        if OmegaConf.select(config, "actor_rollout_ref.rollout.prometheus") is None:
            rollout_cfg.prometheus = OmegaConf.create(
                {
                    "_target_": "verl.workers.config.rollout.PrometheusConfig",
                    "enable": False,
                    "port": 9090,
                    "file": "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml",
                    "served_model_name": None,
                }
            )

        # Teacher defaults used by teacher client bootstrap.
        teacher_cfg = config.actor_rollout_ref.teacher
        if OmegaConf.select(config, "actor_rollout_ref.teacher.n_server_workers") is None:
            teacher_cfg.n_server_workers = 1
        if OmegaConf.select(config, "actor_rollout_ref.teacher.overlap_rollout") is None:
            teacher_cfg.overlap_rollout = False


# Define a function to run the PPO-like training process
def run_on_policy_distill(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    _apply_legacy_config_compat(config)

    # Check if Ray is not initialized

    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        # PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_DEBUG"] = "INFO"
        ray.init(
            runtime_env=RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Megatron-only workers, split into rollout and actor
        if config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup

            from .megatron_workers import (
                MegatronOnPolicyDistillActorWorker,
                MegatronOnPolicyDistillRolloutWorker,
            )

            rollout_cls = MegatronOnPolicyDistillRolloutWorker
            actor_cls = MegatronOnPolicyDistillActorWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        # Worker mapping and resource pools
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.Rollout: ray.remote(rollout_cls),
            Role.Actor: ray.remote(actor_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "rollout_pool": rollout_pool,
            "actor_pool": actor_pool,
        }
        mapping = {
            Role.Rollout: "rollout_pool",
            Role.Actor: "actor_pool",
        }
        print(f"resource_pool_spec: {resource_pool_spec}")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Create training and validation datasets.
        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = OnPolicyDistillTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
