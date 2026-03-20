from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from transformers import AutoTokenizer


STRICT_EXT = "recipe.waste_sd.vllm_strict_draft_probs_patch.StrictDraftProbsWorkerExtension"


async def _run(args) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=args.trust_remote_code)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    engine_args = AsyncEngineArgs(
        model=args.target_model_path,
        tokenizer=args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=1,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        distributed_executor_backend="mp",
        disable_log_stats=True,
        speculative_config={
            "method": "draft_model",
            "model": args.draft_model_path,
            "num_speculative_tokens": 1,
        },
        worker_extension_cls=STRICT_EXT,
    )

    engine = AsyncLLM.from_engine_args(engine_args, usage_context=UsageContext.ENGINE_CONTEXT)
    print("enable_patch", await engine.collective_rpc("enable_strict_draft_probs_patch"))
    print(
        "set_debug_path",
        await engine.collective_rpc(
            "set_strict_rejection_debug_jsonl",
            args=(args.debug_jsonl,),
        ),
    )
    print("get_debug_path_before", await engine.collective_rpc("get_strict_rejection_debug_jsonl"))
    print("row_count_before", await engine.collective_rpc("get_strict_rejection_debug_row_count"))

    generator = engine.generate(
        prompt=TokensPrompt(prompt_token_ids=list(prompt_ids)),
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            seed=args.seed,
            detokenize=False,
            skip_special_tokens=False,
        ),
        request_id="strict_debug_smoke_0",
    )
    final_output = None
    async for output in generator:
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM returned no output")

    print("output_token_ids", final_output.outputs[0].token_ids)
    print("get_debug_path_after", await engine.collective_rpc("get_strict_rejection_debug_jsonl"))
    print("row_count_after", await engine.collective_rpc("get_strict_rejection_debug_row_count"))

    debug_paths = [Path(path) for path in await engine.collective_rpc("get_strict_rejection_debug_jsonl") if path]
    for debug_path in debug_paths:
        print("debug_exists", str(debug_path), debug_path.exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--visible-device", default="7")
    parser.add_argument("--tmp-dir", default="/work5/jingwut/tmp")
    parser.add_argument(
        "--debug-jsonl",
        default="/work5/jingwut/On-Policy-Distillation/verl/outputs/manual_strict_debug.jsonl",
    )
    parser.add_argument(
        "--prompt",
        default="Pick one digit from 0 to 9 at random. Output only the digit.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    os.environ["TMPDIR"] = args.tmp_dir
    os.environ["TEMP"] = args.tmp_dir
    os.environ["TMP"] = args.tmp_dir
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
