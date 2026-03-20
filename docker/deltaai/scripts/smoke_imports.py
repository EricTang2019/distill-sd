#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib


ENV_IMPORTS = {
    "distillsd": [
        "torch",
        "flash_attn",
        "flashinfer",
        "sglang",
        "sgl_kernel",
        "ray",
        "transformers",
        "recipe.waste_sd.main_teacher_off_policy",
    ],
    "verlsd": [
        "torch",
        "flash_attn",
        "flashinfer",
        "vllm",
        "ray",
        "transformers",
        "recipe.waste_sd.eval_block_counts_vllm",
        "recipe.waste_sd.vllm_strict_draft_probs_patch",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the core modules for a DeltaAI verl environment.")
    parser.add_argument("env_name", choices=sorted(ENV_IMPORTS))
    args = parser.parse_args()

    for module_name in ENV_IMPORTS[args.env_name]:
        importlib.import_module(module_name)
        print(f"OK {module_name}")


if __name__ == "__main__":
    main()
