from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter


DEFAULT_TMP_DIR = "/work5/jingwut/tmp"
DEFAULT_ROLLOUT_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."


def _parse_visible_devices(value: str) -> list[int]:
    devices = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not devices:
        raise ValueError("Expected at least one CUDA device id")
    return devices


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_tmp_env(tmp_dir: str) -> None:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir


def _launch_teacher_server(
    *,
    server_kwargs: dict[str, Any],
    cuda_visible_devices: str,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(server_kwargs["model_path"]),
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
    ]
    if server_kwargs.get("disable_cuda_graph"):
        command.append("--disable-cuda-graph")
    if server_kwargs.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if server_kwargs.get("api_key"):
        command.extend(["--api-key", str(server_kwargs["api_key"])])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("COMMAND: " + " ".join(command) + "\n")
    log_file.flush()
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_file.close()
        raise
    process._verl_log_file = log_file
    return process


def _wait_for_teacher_server(
    host: str,
    port: int,
    api_key: str,
    timeout_s: float,
    server_process: subprocess.Popen | None = None,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base_url = f"http://{host}:{port}"
    start = time.time()
    while time.time() - start < timeout_s:
        if server_process is not None:
            exit_code = server_process.poll()
            if exit_code is not None:
                raise RuntimeError(f"Teacher SGLang server exited early with code {exit_code}")
        try:
            response = requests.get(f"{base_url}/health_generate", headers=headers, timeout=5.0)
            if response.status_code == 200:
                requests.get(f"{base_url}/flush_cache", headers=headers, timeout=10.0)
                return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Teacher SGLang server did not become ready within {timeout_s}s")


def _normalize_prompt_messages(prompt_value: Any) -> list[dict[str, str]]:
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()

    if isinstance(prompt_value, str):
        return [{"role": "user", "content": prompt_value}]

    if isinstance(prompt_value, dict):
        if "messages" in prompt_value:
            return _normalize_prompt_messages(prompt_value["messages"])
        if "role" in prompt_value and "content" in prompt_value:
            return [{"role": str(prompt_value["role"]), "content": str(prompt_value["content"])}]

    if isinstance(prompt_value, (list, tuple)):
        messages: list[dict[str, str]] = []
        for idx, item in enumerate(prompt_value):
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                raise ValueError(
                    f"Prompt message at index {idx} must be a dict with role/content, got {type(item).__name__}"
                )
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
        if messages:
            return messages

    raise ValueError(f"Unsupported prompt format: {type(prompt_value).__name__}")


def _prepend_rollout_instruction(messages: list[dict[str, str]], instruction: str = DEFAULT_ROLLOUT_INSTRUCTION) -> list[dict[str, str]]:
    updated_messages = [dict(message) for message in messages]
    for idx in range(len(updated_messages)):
        if updated_messages[idx]["role"] != "user":
            continue
        content = updated_messages[idx]["content"].rstrip()
        if instruction not in content:
            content = f"{instruction}\n\n{content}" if content else instruction
        updated_messages[idx]["content"] = content
        return updated_messages
    raise ValueError("Expected at least one user message in prompt")


def _prompt_text_for_storage(messages: list[dict[str, str]]) -> str:
    user_contents = [message["content"] for message in messages if message["role"] == "user"]
    if user_contents:
        return "\n\n".join(user_contents)
    return "\n\n".join(f'{message["role"]}: {message["content"]}' for message in messages)


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return list(rendered)


def _extract_response_ids(response: dict[str, Any]) -> list[int]:
    meta_info = response.get("meta_info", {}) if isinstance(response, dict) else {}
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if output_token_logprobs:
        return [int(token_id) for _, token_id, _ in output_token_logprobs]

    output_ids = response.get("output_ids")
    if output_ids is None:
        raise ValueError("Teacher rollout response missing output_ids")
    return [int(x) for x in output_ids]


def _load_records(data_path: str, *, prompt_key: str, max_prompts: int) -> list[dict[str, Any]]:
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse JSONL line {line_no} from {path}: {exc}") from exc
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected top-level list in {path}, got {type(payload).__name__}")
        records = payload
    elif suffix == ".parquet":
        records = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if max_prompts > 0 and len(normalized) >= max_prompts:
            break
        if prompt_key not in record:
            raise ValueError(f"Missing prompt key {prompt_key!r} in record {index}")
        messages = _prepend_rollout_instruction(_normalize_prompt_messages(record[prompt_key]))
        uid = record.get("uid", f"sample_{index:08d}")
        normalized.append(
            {
                "uid": str(uid),
                "source_index": index,
                "messages": messages,
                "prompt_text": _prompt_text_for_storage(messages),
                "data_source": record.get("data_source"),
            }
        )
    return normalized


def _build_sampling_params(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    normalized_top_k = -1 if top_k == 0 else top_k
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": normalized_top_k,
        "max_new_tokens": max_new_tokens,
    }
    if seed is not None:
        params["seed"] = seed
    return params


def _normalize_text_response(response_text: Any, tokenizer, response_ids: list[int]) -> str:
    if isinstance(response_text, str):
        return response_text
    if isinstance(response_text, list):
        return "".join(str(x) for x in response_text)
    return tokenizer.decode(response_ids, skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect teacher rollouts into offline waste_sd dataset format")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--teacher-visible-devices", required=True)
    parser.add_argument("--parallel-mode", choices=("dp", "tp"), default="dp")
    parser.add_argument("--teacher-tp-size", type=int, default=None)
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=0)
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--teacher-timeout-s", type=float, default=300.0)
    parser.add_argument("--teacher-attention-backend", default="fa3")
    parser.add_argument("--teacher-sampling-backend", default="flashinfer")
    parser.add_argument("--teacher-disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--teacher-chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--teacher-max-total-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_tmp_env(args.tmp_dir)

    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_records(args.input_data, prompt_key=args.prompt_key, max_prompts=args.max_prompts)
    for record in records:
        record["prompt_ids"] = _apply_chat_template(tokenizer, record["messages"], enable_thinking=args.enable_thinking)

    teacher_devices = _parse_visible_devices(args.teacher_visible_devices)
    server_processes: list[subprocess.Popen] = []
    adapters: list[HttpServerAdapter] = []
    try:
        if args.parallel_mode == "dp":
            for replica_idx, device_id in enumerate(teacher_devices):
                teacher_port = _find_free_port()
                server_kwargs = {
                    "host": args.teacher_host,
                    "port": teacher_port,
                    "model_path": args.teacher_model_path,
                    "tensor_parallel_size": 1,
                    "api_key": args.teacher_api_key,
                    "dtype": args.dtype,
                    "trust_remote_code": args.trust_remote_code,
                    "log_level": "error",
                    "attention_backend": args.teacher_attention_backend,
                    "sampling_backend": args.teacher_sampling_backend,
                    "disable_cuda_graph": args.teacher_disable_cuda_graph,
                    "mem_fraction_static": args.teacher_mem_fraction_static,
                    "chunked_prefill_size": args.teacher_chunked_prefill_size,
                    "max_total_tokens": args.teacher_max_total_tokens,
                }
                teacher_log_path = output_path.parent / f"teacher_sglang_dp_{replica_idx}.log"
                server_process = _launch_teacher_server(
                    server_kwargs=server_kwargs,
                    cuda_visible_devices=str(device_id),
                    log_path=teacher_log_path,
                )
                _wait_for_teacher_server(
                    args.teacher_host,
                    teacher_port,
                    args.teacher_api_key,
                    args.teacher_timeout_s,
                    server_process=server_process,
                )
                server_processes.append(server_process)
                adapters.append(
                    HttpServerAdapter(
                        host=args.teacher_host,
                        port=teacher_port,
                        node_rank=0,
                        model_path=args.teacher_model_path,
                        api_key=args.teacher_api_key,
                        launch_server=False,
                    )
                )
        else:
            teacher_port = args.teacher_port or _find_free_port()
            teacher_tp_size = args.teacher_tp_size or len(teacher_devices)
            if teacher_tp_size <= 0:
                raise ValueError("teacher_tp_size must be positive")
            server_kwargs = {
                "host": args.teacher_host,
                "port": teacher_port,
                "model_path": args.teacher_model_path,
                "tensor_parallel_size": teacher_tp_size,
                "api_key": args.teacher_api_key,
                "dtype": args.dtype,
                "trust_remote_code": args.trust_remote_code,
                "log_level": "error",
                "attention_backend": args.teacher_attention_backend,
                "sampling_backend": args.teacher_sampling_backend,
                "disable_cuda_graph": args.teacher_disable_cuda_graph,
                "mem_fraction_static": args.teacher_mem_fraction_static,
                "chunked_prefill_size": args.teacher_chunked_prefill_size,
                "max_total_tokens": args.teacher_max_total_tokens,
            }
            teacher_log_path = output_path.parent / "teacher_sglang.log"
            server_process = _launch_teacher_server(
                server_kwargs=server_kwargs,
                cuda_visible_devices=args.teacher_visible_devices,
                log_path=teacher_log_path,
            )
            _wait_for_teacher_server(
                args.teacher_host,
                teacher_port,
                args.teacher_api_key,
                args.teacher_timeout_s,
                server_process=server_process,
            )
            server_processes.append(server_process)
            adapters.append(
                HttpServerAdapter(
                    host=args.teacher_host,
                    port=teacher_port,
                    node_rank=0,
                    model_path=args.teacher_model_path,
                    api_key=args.teacher_api_key,
                    launch_server=False,
                )
            )

        metadata = {
            "teacher_model_path": args.teacher_model_path,
            "tokenizer_path": tokenizer_path,
            "input_data": args.input_data,
            "prompt_key": args.prompt_key,
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
        with open(output_path.with_suffix(output_path.suffix + ".meta.json"), "w", encoding="utf-8") as meta_f:
            json.dump(metadata, meta_f, ensure_ascii=False, indent=2)

        jobs: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        for record in records:
            for sample_idx in range(args.samples_per_prompt):
                seed = None
                if args.seed_base is not None:
                    seed = args.seed_base + record["source_index"] * args.samples_per_prompt + sample_idx
                jobs.append(
                    (
                        record,
                        sample_idx,
                        _build_sampling_params(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_new_tokens=args.max_new_tokens,
                            seed=seed,
                        ),
                    )
                )

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
                future = executor.submit(
                    adapter.generate,
                    input_ids=record["prompt_ids"],
                    sampling_params=sampling_params,
                )
                pending[future] = (record, sample_idx)
                return True

            for _ in range(min(args.max_concurrent_requests, total_jobs)):
                _submit_next()

            with open(output_path, "w", encoding="utf-8") as out_f:
                with tqdm(
                    total=total_jobs,
                    desc="collect_offline_rollout",
                    dynamic_ncols=True,
                    disable=args.disable_tqdm,
                ) as progress:
                    while pending:
                        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                        for future in done:
                            record, sample_idx = pending.pop(future)
                            response = future.result()
                            response_ids = _extract_response_ids(response)
                            response_text = _normalize_text_response(response.get("text"), tokenizer, response_ids)
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
                            }
                            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            completed += 1
                            progress.update(1)
                            if (
                                args.log_every > 0
                                and args.disable_tqdm
                                and (completed % args.log_every == 0 or completed == total_jobs)
                            ):
                                print(f"[collect_offline_rollout] completed {completed}/{total_jobs}")
                            _submit_next()
    finally:
        for adapter in adapters:
            try:
                requests.get(
                    f"http://{adapter.server_args.host}:{adapter.server_args.port}/shutdown",
                    headers={"Authorization": f"Bearer {args.teacher_api_key}"} if args.teacher_api_key else {},
                    timeout=5.0,
                )
            except Exception:
                pass
        for server_process in server_processes:
            if server_process.poll() is None:
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(server_process.pid)
                except Exception:
                    server_process.terminate()
                    server_process.wait(timeout=5.0)
            log_file = getattr(server_process, "_verl_log_file", None)
            if log_file is not None:
                log_file.close()


if __name__ == "__main__":
    main()
