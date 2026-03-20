#!/usr/bin/env python3
"""Minimal FSDP PPO smoke test for verl.

This script creates tiny RLHF train/val json files and launches one training step
through `python -m verl.trainer.main_ppo`.

Default mode uses `rollout=vllm`, disables critic, and
forces eager attention to avoid flash-attn ABI issues in mixed environments.
Use `--with-critic` if you explicitly want to include critic initialization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def build_rows(n_rows: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(n_rows):
        a = i + 2
        b = i + 3
        gt = str(a + b)
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "ability": "math",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"What is {a} + {b}? Think step by step and end with #### {gt}.",
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt,
                },
                "extra_info": {
                    "index": i,
                },
            }
        )
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_smoke(args: argparse.Namespace, train_json: Path, val_json: Path) -> int:
    if args.rollout == "hf":
        raise ValueError(
            "rollout=hf is not available in async rollout registry on this branch. "
            "Use --rollout vllm (recommended) or --rollout sglang."
        )
    if args.rollout == "vllm":
        try:
            import numba  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "rollout=vllm requires `numba` in this vLLM version. "
                "Please install it in your conda env first, then rerun."
            ) from e
        # vLLM imports flash_attn rotary op when flash_attn is installed. If the
        # wheel ABI mismatches current torch, fail early with actionable message.
        if importlib.util.find_spec("flash_attn") is not None:
            try:
                import torch  # noqa: F401
                import flash_attn_2_cuda  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "Detected incompatible flash-attn binary in current env. "
                    "For vLLM smoke test, uninstall flash-attn (or reinstall a wheel "
                    "built for your current torch/cuda) and rerun."
                ) from e

    cache_root = args.cache_root.expanduser().resolve()
    triton_cache = cache_root / "triton"
    hf_home = cache_root / "huggingface"
    xdg_cache_home = cache_root / "xdg"
    tmpdir = cache_root / "tmp"
    for p in [cache_root, triton_cache, hf_home, xdg_cache_home, tmpdir]:
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(triton_cache)
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hf_home / "hub")
    env["XDG_CACHE_HOME"] = str(xdg_cache_home)
    env["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    env["TMPDIR"] = str(tmpdir)

    print(f"Using cache root: {cache_root}")
    print(f"TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}")

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_json}",
        f"data.val_files={val_json}",
        f"actor_rollout_ref.model.path={args.model}",
        f"critic.model.path={args.model}",
        f"actor_rollout_ref.rollout.name={args.rollout}",
        f"trainer.total_training_steps={args.steps}",
        "trainer.val_before_train=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.logger=console",
        "+ray_kwargs.ray_init.include_dashboard=False",
        "data.train_batch_size=8",
        "data.val_batch_size=8",
        "data.dataloader_num_workers=0",
        "data.max_prompt_length=128",
        "data.max_response_length=64",
        "data.filter_overlong_prompts=False",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.actor.strategy=fsdp",
        "critic.strategy=fsdp",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.ref.use_torch_compile=False",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "critic.model.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "critic.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        # Keep rollout in eager/no-sleep mode for smoke stability on single GPU.
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.max_num_seqs=128",
        "actor_rollout_ref.rollout.max_num_batched_tokens=1024",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "critic.optim.lr=1e-5",
        "algorithm.kl_ctrl.kl_coef=0.001",
        f"algorithm.adv_estimator={args.adv_estimator}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={args.attn_implementation}",
        f"+critic.model.override_config.attn_implementation={args.attn_implementation}",
    ]
    if args.spec_draft_model is not None:
        if args.spec_num_tokens <= 0:
            raise ValueError("--spec-num-tokens must be > 0 when --spec-draft-model is set.")
        cmd.extend(
            [
                f"+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.model={args.spec_draft_model}",
                f"+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method={args.spec_method}",
                (
                    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens="
                    f"{args.spec_num_tokens}"
                ),
            ]
        )
    if not args.with_critic:
        cmd.extend(["critic.enable=False", "algorithm.adv_estimator=grpo"])
    print("Running command:\n" + " ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd, env=env)
    try:
        if args.timeout_sec > 0:
            return proc.wait(timeout=args.timeout_sec)
        return proc.wait()
    except subprocess.TimeoutExpired:
        print(f"Smoke test timed out after {args.timeout_sec}s, terminating child process.")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"Check Ray logs under: {cache_root / 'tmp' / 'ray'}")
        return 124
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating child process.")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-step FSDP PPO smoke test.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model path or local model path for actor/ref/critic.",
    )
    parser.add_argument(
        "--rollout",
        type=str,
        default="vllm",
        choices=["hf", "vllm", "sglang", "trtllm"],
        help="Rollout backend to smoke test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Total training steps for smoke test.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="HF attention implementation override.",
    )
    parser.add_argument(
        "--adv-estimator",
        type=str,
        default="grpo",
        choices=["grpo", "gae", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "rloo"],
        help="Advantage estimator. If --with-critic is not set, it will be forced to grpo.",
    )
    parser.add_argument(
        "--with-critic",
        action="store_true",
        help="Enable critic initialization (requires value-head path, often needs trl and compatible flash-attn).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/verl_fsdp_smoke"),
        help="Directory to write temporary train/val json files.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/work5/jingwut/.cache/verl_smoke"),
        help="Cache root for Triton/HF/vLLM/TorchInductor/tmp to avoid filling $HOME.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1200,
        help="Timeout for the spawned trainer process. Set <=0 to disable timeout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved command and exit without launching training.",
    )
    parser.add_argument(
        "--spec-draft-model",
        type=str,
        default=None,
        help="Enable vLLM speculative decoding with a separate draft model path.",
    )
    parser.add_argument(
        "--spec-num-tokens",
        type=int,
        default=4,
        help="num_speculative_tokens for vLLM speculative decoding.",
    )
    parser.add_argument(
        "--spec-method",
        type=str,
        default="draft_model",
        help="vLLM speculative method (for draft-model setup, use draft_model).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_json = args.workdir / "train.json"
    val_json = args.workdir / "val.json"
    rows = build_rows(8)
    dump_jsonl(train_json, rows)
    dump_jsonl(val_json, rows[:4])

    print(f"Wrote train data to: {train_json}")
    print(f"Wrote val data to:   {val_json}")
    rc = run_smoke(args, train_json, val_json)
    if rc == 0:
        print("FSDP smoke test PASSED.")
    else:
        print(f"FSDP smoke test FAILED with exit code {rc}.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
