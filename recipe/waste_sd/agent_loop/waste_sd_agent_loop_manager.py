from __future__ import annotations

from omegaconf import DictConfig, OmegaConf, open_dict

from verl.experimental.agent_loop import AgentLoopManager


def build_waste_sd_agent_loop_config(config: DictConfig) -> DictConfig:
    """Build an agent-loop-local config with rollout prompt formatting aligned to rollout target.

    Waste-SD may rollout against `local_ref` while actor workers still train with the actor
    model config. AgentLoopWorker reads tokenizer/template fields from
    `config.actor_rollout_ref.model`, so we rewrite only the config copy passed to the
    agent-loop manager, leaving actor training config untouched.
    """

    rewritten = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    distill_cfg = rewritten.get("distill", {})
    rollout_target = str(distill_cfg.get("rollout_target", "local_ref")).lower()
    require_ref_path = bool(distill_cfg.get("off_policy_require_ref", False))

    actor_model_cfg = rewritten.actor_rollout_ref.model
    actor_model_path = actor_model_cfg.get("path", None)

    ref_cfg = rewritten.actor_rollout_ref.get("ref", None)
    ref_model_cfg = ref_cfg.get("model", None) if ref_cfg is not None else None
    ref_model_path = ref_model_cfg.get("path", None) if ref_model_cfg is not None else None

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
            "Refusing actor-path fallback in Waste-SD agent loop."
        )
    if rollout_target == "actor" or not isinstance(ref_model_path, str) or not ref_model_path.strip():
        return rewritten

    ref_tokenizer_path = ref_model_cfg.get("tokenizer_path", None) if ref_model_cfg is not None else None
    ref_hf_config_path = ref_model_cfg.get("hf_config_path", None) if ref_model_cfg is not None else None
    ref_custom_chat_template = ref_model_cfg.get("custom_chat_template", None) if ref_model_cfg is not None else None

    with open_dict(actor_model_cfg):
        actor_model_cfg.path = ref_model_path
        actor_model_cfg.tokenizer_path = ref_tokenizer_path or ref_model_path
        if actor_model_cfg.get("hf_config_path", None) in (None, actor_model_path):
            actor_model_cfg.hf_config_path = ref_hf_config_path or ref_model_path
        # Avoid forcing an actor-side template onto a ref-side tokenizer.
        actor_model_cfg.custom_chat_template = ref_custom_chat_template

    return rewritten


class WasteSDAgentLoopManager(AgentLoopManager):
    """Waste-SD-local AgentLoopManager wrapper.

    It rewrites only the manager-local config so AgentLoopWorker tokenizer/template selection
    matches the rollout target model identity.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config=build_waste_sd_agent_loop_config(config), *args, **kwargs)
