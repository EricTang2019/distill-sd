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

import hydra
from omegaconf import open_dict

from recipe.waste_sd.main_waste_sd import WasteSDTaskRunner, run_waste_sd
from recipe.waste_sd.ray_trainer_teacher_off_policy import WasteSDTeacherOffPolicyRayTrainer


class WasteSDTeacherOffPolicyTaskRunner(WasteSDTaskRunner):
    def __init__(self):
        super().__init__()
        self.trainer_cls = WasteSDTeacherOffPolicyRayTrainer


def apply_teacher_off_policy_defaults(config) -> None:
    """Apply teacher-rollout off-policy defaults without clobbering explicit user overrides."""
    with open_dict(config):
        if config.get("distill") is None:
            config.distill = {}
        if config.get("data") is None:
            config.data = {}
        if config.distill.get("weighting_mode", None) in {None, "waste"}:
            config.distill.weighting_mode = "uniform_mean"
        if config.distill.get("data_mode", None) is None:
            config.distill.data_mode = "online_rollout"
        if config.distill.get("gamma", None) is None:
            config.distill.gamma = 1
        if config.distill.get("kl_floor_coef", None) is None:
            config.distill.kl_floor_coef = 0.0
        config.distill.off_policy_require_ref = True
        config.distill.rollout_target = "local_ref"

        if str(config.distill.data_mode).lower() == "offline_teacher_rollout":
            if config.data.get("custom_cls", None) is None:
                config.data.custom_cls = {}
            if config.data.custom_cls.get("path", None) is None:
                config.data.custom_cls.path = "pkg://recipe.waste_sd.offline_rollout_dataset"
            if config.data.custom_cls.get("name", None) is None:
                config.data.custom_cls.name = "OfflineTeacherRolloutDataset"

        # Force-disable speculative/MTP rollout in off-policy teacher baseline,
        # even when users launch this entrypoint with speculative overrides.
        mtp_cfg = config.actor_rollout_ref.model.get("mtp", None)
        if mtp_cfg is not None:
            mtp_cfg.enable = False
            mtp_cfg.enable_train = False
            mtp_cfg.enable_rollout = False

        rollout_cfg = config.actor_rollout_ref.get("rollout", None)
        if rollout_cfg is not None:
            rollout_mtp_cfg = rollout_cfg.get("mtp", None)
            if rollout_mtp_cfg is not None:
                rollout_mtp_cfg.enable = False
                rollout_mtp_cfg.enable_train = False
                rollout_mtp_cfg.enable_rollout = False
            if rollout_cfg.get("engine_kwargs") is None:
                rollout_cfg.engine_kwargs = {}
            if rollout_cfg.engine_kwargs.get("sglang") is None:
                rollout_cfg.engine_kwargs.sglang = {}
            sglang_kwargs = rollout_cfg.engine_kwargs.sglang
            for key in (
                "speculative_algorithm",
                "speculative_num_steps",
                "speculative_num_draft_tokens",
                "speculative_eagle_topk",
                "speculative_draft_model_path",
                "speculative_draft_model_revision",
                "speculative_draft_load_format",
            ):
                sglang_kwargs.pop(key, None)


@hydra.main(config_path="config", config_name="waste_sd_trainer", version_base=None)
def main(config):
    # Hard-disable Waste-SD runtime patch for this baseline even when launched
    # without `run_teacher_off_policy.sh`.
    os.environ["VERL_SGLANG_WASTE_SD_PATCH"] = "0"
    os.environ["VERL_SGLANG_STANDALONE_SYNC_MODE"] = "both"

    ref_cfg = config.actor_rollout_ref.get("ref", None)
    ref_model_cfg = ref_cfg.get("model", None) if ref_cfg is not None else None
    ref_model_path = ref_model_cfg.get("path", None) if ref_model_cfg is not None else None
    if not isinstance(ref_model_path, str) or not ref_model_path.strip():
        raise ValueError(
            "Teacher off-policy baseline requires actor_rollout_ref.ref.model.path to be explicitly set. "
            "Refusing to fall back to actor model path."
        )

    apply_teacher_off_policy_defaults(config)

    run_waste_sd(
        config,
        task_runner_cls=WasteSDTeacherOffPolicyTaskRunner,
        enable_patch_env=False,
        patch_env_override={
            "VERL_SGLANG_WASTE_SD_PATCH": "0",
            "VERL_SGLANG_STANDALONE_SYNC_MODE": "both",
            # Waste-SD patch is disabled in this baseline, so spec_accept_lens is not emitted by SGLang.
            # Keep this off to avoid noisy "Missing spec_accept_lens" warnings in rollout servers.
            "VERL_SGLANG_WASTE_SD_STRICT": "0",
        },
        force_patch_env=True,
    )


if __name__ == "__main__":
    main()
