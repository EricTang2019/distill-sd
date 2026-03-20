from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from recipe.waste_sd.collect_offline_rollout import (
    DEFAULT_ROLLOUT_INSTRUCTION,
    DEFAULT_TMP_DIR,
    _apply_chat_template,
    _build_sampling_params,
    _configure_tmp_env,
    _extract_response_ids,
    _load_records,
    _normalize_prompt_messages,
    _normalize_text_response,
    _parse_visible_devices,
    _prepend_rollout_instruction,
    _prompt_text_for_storage,
    _wait_for_teacher_server,
)
from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_patched_server(
    *,
    server_kwargs: dict[str, Any],
    cuda_visible_devices: str,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VERL_SGLANG_WASTE_SD_PATCH", "1")

    command = [
        sys.executable,
        "-m",
        "recipe.waste_sd.launch_patched_sglang_server",
        "--model-path",
        str(server_kwargs["model_path"]),
        "--speculative-draft-model-path",
        str(server_kwargs["draft_model_path"]),
        "--host",
        str(server_kwargs["host"]),
        "--port",
        str(server_kwargs["port"]),
        "--tensor-parallel-size",
        str(server_kwargs["tensor_parallel_size"]),
        "--dtype",
        str(server_kwargs["dtype"]),
        "--log-level",
        str(server_kwargs["log_level"]),
        "--attention-backend",
        str(server_kwargs["attention_backend"]),
        "--sampling-backend",
        str(server_kwargs["sampling_backend"]),
        "--mem-fraction-static",
        str(server_kwargs["mem_fraction_static"]),
        "--chunked-prefill-size",
        str(server_kwargs["chunked_prefill_size"]),
        "--max-total-tokens",
        str(server_kwargs["max_total_tokens"]),
        "--speculative-algorithm",
        str(server_kwargs["speculative_algorithm"]),
        "--speculative-draft-model-revision",
        str(server_kwargs["speculative_draft_model_revision"]),
        "--speculative-draft-load-format",
        str(server_kwargs["speculative_draft_load_format"]),
        "--speculative-num-steps",
        str(server_kwargs["speculative_num_steps"]),
        "--speculative-eagle-topk",
        str(server_kwargs["speculative_eagle_topk"]),
        "--speculative-num-draft-tokens",
        str(server_kwargs["speculative_num_draft_tokens"]),
        "--tokenizer-worker-num",
        str(server_kwargs["tokenizer_worker_num"]),
    ]
    if server_kwargs.get("disable_cuda_graph"):
        command.append("--disable-cuda-graph")
    if server_kwargs.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if server_kwargs.get("skip_tokenizer_init"):
        command.append("--skip-tokenizer-init")
    if server_kwargs.get("api_key"):
        command.extend(["--api-key", str(server_kwargs["api_key"])])
    if server_kwargs.get("tokenizer_path"):
        command.extend(["--tokenizer-path", str(server_kwargs["tokenizer_path"])])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("COMMAND: " + " ".join(command) + "\n")
    log_file.flush()
    try:
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception:
        log_file.close()
        raise
    process._verl_log_file = log_file
    return process


def _extract_verify_stats(response: dict[str, Any]) -> dict[str, Any]:
    meta_info = response.get("meta_info", {}) if isinstance(response, dict) else {}
    return {
        "spec_verify_ct": int(meta_info.get("spec_verify_ct", 0) or 0),
        "spec_accepted_tokens": int(meta_info.get("spec_accepted_tokens", 0) or 0),
        "spec_accept_lens": meta_info.get("spec_accept_lens"),
        "completion_tokens": int(meta_info.get("completion_tokens", len(_extract_response_ids(response))) or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight direct SD block-count eval using patched SGLang generate")
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--parallel-mode", choices=("dp", "tp"), default="dp")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--sampling-backend", default="flashinfer")
    parser.add_argument("--disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--no-rollout-instruction", action="store_true")
    parser.add_argument("--speculative-num-steps", type=int, default=3)
    parser.add_argument("--speculative-eagle-topk", type=int, default=1)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=4)
    parser.add_argument("--speculative-draft-model-revision", default="main")
    parser.add_argument("--speculative-draft-load-format", default="auto")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_tmp_env(args.tmp_dir)

    tokenizer_path = args.tokenizer_path or args.target_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_records(args.input_data, prompt_key=args.prompt_key, max_prompts=args.max_prompts)
    for record in records:
        messages = _normalize_prompt_messages(record["messages"])
        if not args.no_rollout_instruction:
            messages = _prepend_rollout_instruction(messages, DEFAULT_ROLLOUT_INSTRUCTION)
        record["messages"] = messages
        record["prompt_text"] = _prompt_text_for_storage(messages)
        record["prompt_ids"] = _apply_chat_template(tokenizer, messages, enable_thinking=args.enable_thinking)

    devices = _parse_visible_devices(args.visible_devices)
    server_processes: list[subprocess.Popen] = []
    adapters: list[HttpServerAdapter] = []
    try:
        if args.parallel_mode == "dp":
            for replica_idx, device_id in enumerate(devices):
                port = _find_free_port()
                server_kwargs = {
                    "host": args.host,
                    "port": port,
                    "model_path": args.target_model_path,
                    "draft_model_path": args.draft_model_path,
                    "tensor_parallel_size": 1,
                    "api_key": args.api_key,
                    "dtype": args.dtype,
                    "trust_remote_code": args.trust_remote_code,
                    "log_level": "error",
                    "attention_backend": args.attention_backend,
                    "sampling_backend": args.sampling_backend,
                    "disable_cuda_graph": args.disable_cuda_graph,
                    "mem_fraction_static": args.mem_fraction_static,
                    "chunked_prefill_size": args.chunked_prefill_size,
                    "max_total_tokens": args.max_total_tokens,
                    "tokenizer_path": tokenizer_path,
                    "tokenizer_worker_num": 1,
                    "skip_tokenizer_init": False,
                    "speculative_algorithm": "STANDALONE",
                    "speculative_draft_model_revision": args.speculative_draft_model_revision,
                    "speculative_draft_load_format": args.speculative_draft_load_format,
                    "speculative_num_steps": args.speculative_num_steps,
                    "speculative_eagle_topk": args.speculative_eagle_topk,
                    "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
                }
                log_path = output_path.parent / f"block_eval_sglang_dp_{replica_idx}.log"
                process = _launch_patched_server(server_kwargs=server_kwargs, cuda_visible_devices=str(device_id), log_path=log_path)
                _wait_for_teacher_server(args.host, port, args.api_key, args.timeout_s, server_process=process)
                server_processes.append(process)
                adapters.append(HttpServerAdapter(host=args.host, port=port, node_rank=0, model_path=args.target_model_path, api_key=args.api_key, launch_server=False))
        else:
            port = args.port or _find_free_port()
            tp_size = args.tp_size or len(devices)
            server_kwargs = {
                "host": args.host,
                "port": port,
                "model_path": args.target_model_path,
                "draft_model_path": args.draft_model_path,
                "tensor_parallel_size": tp_size,
                "api_key": args.api_key,
                "dtype": args.dtype,
                "trust_remote_code": args.trust_remote_code,
                "log_level": "error",
                "attention_backend": args.attention_backend,
                "sampling_backend": args.sampling_backend,
                "disable_cuda_graph": args.disable_cuda_graph,
                "mem_fraction_static": args.mem_fraction_static,
                "chunked_prefill_size": args.chunked_prefill_size,
                "max_total_tokens": args.max_total_tokens,
                "tokenizer_path": tokenizer_path,
                "tokenizer_worker_num": 1,
                "skip_tokenizer_init": False,
                "speculative_algorithm": "STANDALONE",
                "speculative_draft_model_revision": args.speculative_draft_model_revision,
                "speculative_draft_load_format": args.speculative_draft_load_format,
                "speculative_num_steps": args.speculative_num_steps,
                "speculative_eagle_topk": args.speculative_eagle_topk,
                "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
            }
            log_path = output_path.parent / 'block_eval_sglang.log'
            process = _launch_patched_server(server_kwargs=server_kwargs, cuda_visible_devices=args.visible_devices, log_path=log_path)
            _wait_for_teacher_server(args.host, port, args.api_key, args.timeout_s, server_process=process)
            server_processes.append(process)
            adapters.append(HttpServerAdapter(host=args.host, port=port, node_rank=0, model_path=args.target_model_path, api_key=args.api_key, launch_server=False))

        metadata = {
            "draft_model_path": args.draft_model_path,
            "target_model_path": args.target_model_path,
            "tokenizer_path": tokenizer_path,
            "input_data": args.input_data,
            "samples_per_prompt": args.samples_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": args.enable_thinking,
            "parallel_mode": args.parallel_mode,
            "num_servers": len(adapters),
            "num_prompts": len(records),
            "num_outputs": len(records) * args.samples_per_prompt,
        }
        with open(output_path.with_suffix(output_path.suffix + '.meta.json'), 'w', encoding='utf-8') as meta_f:
            json.dump(metadata, meta_f, ensure_ascii=False, indent=2)

        jobs: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        for record in records:
            for sample_idx in range(args.samples_per_prompt):
                seed = None
                if args.seed_base is not None:
                    seed = args.seed_base + record["source_index"] * args.samples_per_prompt + sample_idx
                jobs.append((record, sample_idx, _build_sampling_params(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, seed=seed)))

        total_jobs = len(jobs)
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.max_concurrent_requests)) as executor:
            pending: dict[Future, tuple[dict[str, Any], int]] = {}
            job_iter = iter(jobs)

            def _submit_next() -> bool:
                try:
                    record, sample_idx, sampling_params = next(job_iter)
                except StopIteration:
                    return False
                adapter = adapters[(record["source_index"] * args.samples_per_prompt + sample_idx) % len(adapters)]
                future = executor.submit(adapter.generate, input_ids=record["prompt_ids"], sampling_params=sampling_params)
                pending[future] = (record, sample_idx)
                return True

            for _ in range(min(args.max_concurrent_requests, total_jobs)):
                _submit_next()

            with open(output_path, 'w', encoding='utf-8') as out_f:
                with tqdm(total=total_jobs, desc='eval_block_counts', dynamic_ncols=True, disable=args.disable_tqdm) as progress:
                    while pending:
                        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                        for future in done:
                            record, sample_idx = pending.pop(future)
                            response = future.result()
                            response_ids = _extract_response_ids(response)
                            response_text = _normalize_text_response(response.get('text'), tokenizer, response_ids)
                            verify_stats = _extract_verify_stats(response)
                            payload = {
                                "uid": f'{record["uid"]}_sample_{sample_idx:02d}',
                                "prompt": record["prompt_text"],
                                "response": response_text,
                                "prompt_ids": record["prompt_ids"],
                                "response_ids": response_ids,
                                "sample_idx": sample_idx,
                                "source_uid": record["uid"],
                                "source_index": record["source_index"],
                                "data_source": record.get("data_source"),
                                **verify_stats,
                            }
                            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            if completed % 32 == 0:
                                out_f.flush()
                            completed += 1
                            progress.update(1)
                            if args.log_every > 0 and args.disable_tqdm and (completed % args.log_every == 0 or completed == total_jobs):
                                print(f"[eval_block_counts] completed {completed}/{total_jobs}")
                            _submit_next()
    finally:
        for adapter in adapters:
            try:
                requests.get(
                    f"http://{adapter.server_args.host}:{adapter.server_args.port}/shutdown",
                    headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
                    timeout=5.0,
                )
            except Exception:
                pass
        for process in server_processes:
            if process.poll() is None:
                try:
                    from sglang.srt.utils import kill_process_tree
                    kill_process_tree(process.pid)
                except Exception:
                    process.terminate()
                    process.wait(timeout=5.0)
            log_file = getattr(process, "_verl_log_file", None)
            if log_file is not None:
                log_file.close()


if __name__ == "__main__":
    main()
