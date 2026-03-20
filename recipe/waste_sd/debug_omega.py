from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.waste_sd.block_count_dp import compute_block_count_omega
from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter


DEFAULT_TMP_DIR = "/work5/jingwut/tmp"


def _parse_visible_devices(value: str) -> list[int]:
    devices = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not devices:
        raise ValueError("Expected at least one CUDA device id")
    return devices


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _configure_tmp_env(tmp_dir: str) -> None:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir


def _wait_for_teacher_server(
    host: str,
    port: int,
    api_key: str,
    timeout_s: float,
    server_process: Optional[subprocess.Popen] = None,
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


def _apply_chat_template(
    tokenizer,
    prompt_text: str,
    *,
    enable_thinking: bool,
) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
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


def _extract_generated_ids_and_logprobs(response: dict[str, Any]) -> tuple[list[int], list[float]]:
    meta_info = response.get("meta_info", {}) if isinstance(response, dict) else {}
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if output_token_logprobs:
        token_ids = [int(token_id) for _, token_id, _ in output_token_logprobs]
        logprobs = [float(log_prob) for log_prob, _, _ in output_token_logprobs]
        return token_ids, logprobs

    output_ids = response.get("output_ids")
    if output_ids is None:
        raise ValueError("Teacher rollout response missing both output_ids and meta_info.output_token_logprobs")
    return [int(x) for x in output_ids], []


def _load_student_model(
    model_path: str,
    *,
    trust_remote_code: bool,
    dtype: torch.dtype,
    visible_devices: Optional[str],
) -> tuple[AutoModelForCausalLM, torch.device]:
    if not torch.cuda.is_available() or not visible_devices:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, device

    device_ids = _parse_visible_devices(visible_devices)
    if len(device_ids) == 1:
        device = torch.device(f"cuda:{device_ids[0]}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        return model, device

    max_memory = {gpu_id: "70GiB" for gpu_id in device_ids}
    max_memory["cpu"] = "256GiB"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
    )
    first_device = next(model.parameters()).device
    model.eval()
    return model, first_device


def _score_teacher_tokens(
    model: AutoModelForCausalLM,
    *,
    device: torch.device,
    prompt_ids: list[int],
    response_ids: list[int],
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    response_len = len(response_ids)
    if response_len <= 0:
        return torch.empty((0,), dtype=torch.float64, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, -response_len - 1 : -1, :].float()
        target_ids = torch.tensor(response_ids, dtype=torch.long, device=logits.device).unsqueeze(0)
        target_logits = logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(logits, dim=-1)
        token_logprobs = target_logits - log_z
    return token_logprobs.squeeze(0).to(dtype=torch.float64)


def _load_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_text is not None:
        return [{"uid": "sample_00000000", "prompt": args.input_text}]

    if not args.input_jsonl:
        raise ValueError("Provide either --input-text or --input-jsonl")

    samples: list[dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get(args.input_key) or record.get("prompt") or record.get("input") or record.get("text")
            if not isinstance(prompt, str):
                raise ValueError(f"Could not find prompt string in line {idx + 1}")
            uid = record.get("uid", f"sample_{idx:08d}")
            samples.append({"uid": str(uid), "prompt": prompt})
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
    return samples


def _dtype_from_name(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-rollout omega debugger")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--teacher-visible-devices", required=True)
    parser.add_argument("--student-visible-devices", default=None)
    parser.add_argument("--teacher-tp-size", type=int, default=None)
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=0)
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--teacher-timeout-s", type=float, default=300.0)
    parser.add_argument("--teacher-attention-backend", default="fa3")
    parser.add_argument("--teacher-sampling-backend", default="pytorch")
    parser.add_argument("--teacher-disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-mem-fraction-static", type=float, default=0.70)
    parser.add_argument("--teacher-chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--teacher-max-total-tokens", type=int, default=8192)
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--input-text", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_tmp_env(args.tmp_dir)

    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_port = args.teacher_port or _find_free_port()
    teacher_devices = _parse_visible_devices(args.teacher_visible_devices)
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

    adapter = None
    try:
        _wait_for_teacher_server(
            args.teacher_host,
            teacher_port,
            args.teacher_api_key,
            args.teacher_timeout_s,
            server_process=server_process,
        )
        adapter = HttpServerAdapter(
            host=args.teacher_host,
            port=teacher_port,
            node_rank=0,
            model_path=args.teacher_model_path,
            api_key=args.teacher_api_key,
            launch_server=False,
        )

        student_model, student_device = _load_student_model(
            args.student_model_path,
            trust_remote_code=args.trust_remote_code,
            dtype=_dtype_from_name(args.dtype),
            visible_devices=args.student_visible_devices,
        )

        records = _load_prompts(args)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for sample_idx, record in enumerate(records):
                prompt_ids = _apply_chat_template(
                    tokenizer,
                    record["prompt"],
                    enable_thinking=args.enable_thinking,
                )

                sampling_params = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                }
                if args.seed is not None:
                    sampling_params["seed"] = args.seed + sample_idx
                teacher_response = adapter.generate(
                    input_ids=prompt_ids,
                    sampling_params=sampling_params,
                    return_logprob=True,
                )
                response_ids, teacher_logprobs = _extract_generated_ids_and_logprobs(teacher_response)
                if not teacher_logprobs:
                    raise RuntimeError(
                        "Teacher rollout did not return output_token_logprobs; cannot reuse teacher logprobs."
                    )

                student_logprobs = _score_teacher_tokens(
                    student_model,
                    device=student_device,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                ).cpu()

                teacher_logprobs_tensor = torch.tensor(teacher_logprobs, dtype=torch.float64)
                dp_result = compute_block_count_omega(
                    teacher_logprobs=teacher_logprobs_tensor,
                    student_logprobs=student_logprobs,
                    gamma=args.gamma,
                )

                response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                token_text = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in response_ids]
                payload = {
                    "uid": record["uid"],
                    "prompt": record["prompt"],
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "response_token_text": token_text,
                    "teacher_logprobs": [float(x) for x in dp_result.teacher_logprobs.tolist()],
                    "student_logprobs": [float(x) for x in dp_result.student_logprobs.tolist()],
                    "log_ratio_p_over_q": [float(x) for x in dp_result.log_ratio_p_over_q.tolist()],
                    "ratio_p_over_q": [float(x) for x in dp_result.ratio_p_over_q.tolist()],
                    "reject_prob": [float(x) for x in dp_result.reject_prob.tolist()],
                    "omega": [float(x) for x in dp_result.omega.tolist()],
                    "alpha": [float(x) for x in dp_result.alpha.tolist()],
                    "occupancy_u": [[float(v) for v in row] for row in dp_result.occupancy_u.tolist()],
                    "future_block_count_U": [[float(v) for v in row] for row in dp_result.future_block_count_U.tolist()],
                    "advantage_A": [[float(v) for v in row] for row in dp_result.advantage_A.tolist()],
                    "gamma": int(args.gamma),
                    "teacher_model_path": args.teacher_model_path,
                    "student_model_path": args.student_model_path,
                    "tokenizer_path": tokenizer_path,
                }
                out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if adapter is not None:
            try:
                requests.get(
                    f"http://{args.teacher_host}:{teacher_port}/shutdown",
                    headers={"Authorization": f"Bearer {args.teacher_api_key}"} if args.teacher_api_key else {},
                    timeout=5.0,
                )
            except Exception:
                pass
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
