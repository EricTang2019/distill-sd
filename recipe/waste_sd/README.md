# Waste-Aware SD Distillation (SGLang, FSDP)

This recipe implements strict waste-aware SD distillation without modifying `site-packages`.

## What It Does

- Uses **SGLang speculative decoding rollout** to collect:
  - `spec_accept_lens` (for strict `S_n` mapping).
- Computes teacher/student full-vocab distributions with local FSDP forward pass
  (teacher = ref model, student = actor model).
- Keeps rollout target model aligned with local teacher `q`:
  - if `actor_rollout_ref.ref.model.path` is set, waste_sd worker uses it as rollout target path.
  - controlled by `distill.rollout_target`:
    - `local_ref` (default): rollout from teacher/ref path when available.
    - `actor`: rollout from student/actor path (strict on-policy baseline).
- Trains **draft/student** with weighted divergence loss:
  - `fkl`, `rkl`, `tvd`, `teacher_greedy_nll`, or `exact_block_count_wnll`.
  - `teacher_greedy_nll` means: on each sampled prefix from the rollout/offline trajectory,
    maximize the student log-likelihood of the teacher greedy token.
  - `exact_block_count_wnll` means: run the exact forward/backward block-count DP on
    the sampled teacher rollout, compute detached theorem weight
    `alpha_n = omega_n^B 1[p_n^x<q_n^x] p_n^x/q_n^x`, then optimize alpha-weighted sampled-token NLL.
- Weight per response token:
  - `w_n = max(gamma - S_n, 0)`, where `S_n` is accepted-count before generating token `n`.
- Optimization target:
  - for `fkl/rkl/tvd/teacher_greedy_nll`: weighted token mean
    `sum_n w_n * D_n / sum_n w_n`
  - for `exact_block_count_wnll`: alpha-weighted token mean
    `sum_n alpha_n * NLL_n / sum_n alpha_n`
  - optional for `exact_block_count_wnll`: mix in a plain unweighted auxiliary loss with
    `distill.exact_unweighted_aux_loss_type in {fkl, tvd}` and
    `distill.exact_unweighted_aux_coef in [0,1]`
    so the optimized objective becomes
    `(1-c) * exact_wNLL + c * unweighted_aux`
  - optional for `weighting_mode=remaining_budget_forward` with `loss_type=fkl`:
    mix in plain unweighted forward-KL with
    `distill.rembudget_unweighted_kl_coef in [0,1]`
    so the optimized objective becomes
    `(1-c) * rembudget_FKL + c * unweighted_FKL`
- Enforces strict alignment (`distill.strict=true`):
  - mismatch between SD trace and response tokens fails fast.
- Supports staleness filter:
  - drops samples with `version_gap > distill.staleness_max_version_gap`.

## Entry Point

- `python -m recipe.waste_sd.main_waste_sd`

Config:
- `recipe/waste_sd/config/waste_sd_trainer.yaml`

## Runtime Patch Env Vars

These are required for this recipe:

- `VERL_SGLANG_WASTE_SD_PATCH=1`
- `VERL_SGLANG_STANDALONE_SYNC_MODE=draft_only`
- `VERL_SGLANG_WASTE_SD_STRICT=1` (auto-derived from `distill.strict` by `main_waste_sd.py`)

Notes:
- `draft_only` avoids standalone target/draft shape mismatch when only training draft.
- No `site-packages` file edits are needed.

## Quick Start

Use:

- `recipe/waste_sd/run_waste_sd.sh`
- `recipe/waste_sd/run_teacher_on_policy.sh` (strict on-policy teacher baseline)
- `recipe/waste_sd/run_teacher_off_policy_omega.sh` (portable exact block-count / omega off-policy wrapper)
- `scripts/waste_sd_compare/run_gsm8k_qwen_compare.sh` (external orchestrator: train then post-train block eval in same run + summary)
- `scripts/waste_sd_compare/run_gsm8k_block_eval_only.sh` (external test-only block summary)

Pure block-eval mode (no actor update):

- set `+trainer.block_eval_only=true`
- optional:
  - `+trainer.block_eval_split=val|train` (default `val`)
  - `+trainer.block_eval_max_steps=<int>`

or run manually and set:

- `distill.loss_type=fkl|rkl|tvd|teacher_greedy_nll|exact_block_count_wnll`
- `distill.q_source=local_ref`
- `distill.strict=true`
- `distill.staleness_max_version_gap=1`

Recommended model-path setup (draft training):

- `actor_rollout_ref.model.path=<draft_model>`
- `actor_rollout_ref.ref.model.path=<target_model>`
- `actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path=<draft_model>`

Strict on-policy teacher baseline setup:

- `actor_rollout_ref.model.path=<student_model>`
- `actor_rollout_ref.ref.model.path=<teacher_model>`
- `distill.rollout_target=actor`
- `distill.staleness_max_version_gap=0`
- no speculative rollout overrides

## Reproducibility Notes

For exact reproduction of this fork, do **not** start from the generic project requirements or the legacy install script.

Use the pinned bootstrap instead:

```bash
bash scripts/setup/bootstrap_envs.sh
```

This creates the two officially supported environments for this recipe:

- `distillsd`: training / SGLang
- `verlsd`: strict evaluation / vLLM

After the bootstrap finishes:

- run training commands with `conda run -n distillsd ...`
- run strict vLLM eval commands with `conda run -n verlsd ...`

The repository itself is installed into each environment with:

```bash
python -m pip install --no-deps -e .
```

The pinned snapshots live under `envs/` and are validated by `scripts/setup/validate_env.py`.

## Local Component Test

- `python -m recipe.waste_sd.test_waste_sd_components`

## Debug SD Alignment / Blocks / Loss

Turn on structured debug dump:

- `distill.debug.enable=true`
- `distill.debug.dump_dir=/path/to/debug_dir`
- optional:
  - `distill.debug.max_samples_per_step=2`
  - `distill.debug.max_tokens_per_sample=128`
  - `distill.debug.include_topk=true`
  - `distill.debug.logits_topk=5`

Output format:

- one JSONL per worker+step under `dump_dir`:
  - `step_XXXXXXXX_pidYYYY.jsonl`
- each record contains:
  - `spec_accept_lens`
  - `block_trace` (`block_index`, `within_block_offset`, `alignment_ok`)
  - `weights`
  - per-token divergence (`token_divergence`) and weighted contribution
  - optional student/teacher top-k probs per token

## Common Errors

1. `Strict alignment failed ...`
- `spec_accept_lens` and valid response token count mismatch.
- Usually caused by trace missing/truncation mismatch.

2. `Failed to complete async request to update_weights_from_tensor ...`
- Check `VERL_SGLANG_STANDALONE_SYNC_MODE=draft_only`.
- Ensure draft and target shapes are not both being synced in standalone mode.
