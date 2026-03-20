from __future__ import annotations

import argparse
import os

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

from verl.workers.rollout.sglang_rollout.waste_sd_patch import apply_waste_sd_patch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch SGLang server with Waste-SD metadata patch enabled")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--sampling-backend", default="flashinfer")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--speculative-algorithm", default="STANDALONE")
    parser.add_argument("--speculative-draft-model-path", required=True)
    parser.add_argument("--speculative-draft-model-revision", default="main")
    parser.add_argument("--speculative-draft-load-format", default="auto")
    parser.add_argument("--speculative-num-steps", type=int, default=3)
    parser.add_argument("--speculative-eagle-topk", type=int, default=1)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=4)
    parser.add_argument("--tokenizer-worker-num", type=int, default=1)
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    os.environ.setdefault("VERL_SGLANG_WASTE_SD_PATCH", "1")
    # Needed when the draft checkpoint advertises a shorter derived context length
    # than the target teacher model. This launcher is only used for direct
    # Waste-SD eval scripts, so prefer robustness over strict default rejection.
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    apply_waste_sd_patch()

    server_args = ServerArgs(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_worker_num=args.tokenizer_worker_num,
        skip_tokenizer_init=args.skip_tokenizer_init,
        trust_remote_code=args.trust_remote_code,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        tp_size=args.tensor_parallel_size,
        log_level=args.log_level,
        attention_backend=args.attention_backend,
        sampling_backend=args.sampling_backend,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        max_total_tokens=args.max_total_tokens,
        disable_cuda_graph=args.disable_cuda_graph,
        api_key=args.api_key or None,
        speculative_algorithm=args.speculative_algorithm,
        speculative_draft_model_path=args.speculative_draft_model_path,
        speculative_draft_model_revision=args.speculative_draft_model_revision,
        speculative_draft_load_format=args.speculative_draft_load_format,
        speculative_num_steps=args.speculative_num_steps,
        speculative_eagle_topk=args.speculative_eagle_topk,
        speculative_num_draft_tokens=args.speculative_num_draft_tokens,
    )
    launch_server(server_args)


if __name__ == "__main__":
    main()
