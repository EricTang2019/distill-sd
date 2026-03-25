from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from verl.workers.rollout.vllm_rollout.spec_trace import summarize_speculative_chunks


DEFAULT_TMP_DIR = "/work5/jingwut/tmp"
DEFAULT_ROLLOUT_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
STRICT_DRAFT_PROBS_WORKER_EXTENSION = (
    "recipe.waste_sd.vllm_strict_draft_probs_patch.StrictDraftProbsWorkerExtension"
)


def _parse_visible_devices(value: str) -> list[int]:
    devices = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not devices:
        raise ValueError("Expected at least one CUDA device id")
    return devices


def _configure_tmp_env(tmp_dir: str) -> None:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_pythonpath() -> None:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        parts = pythonpath.split(os.pathsep)
        if repo_root not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *parts])
    else:
        os.environ["PYTHONPATH"] = repo_root


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


def _prepend_rollout_instruction(
    messages: list[dict[str, str]],
    instruction: str = DEFAULT_ROLLOUT_INSTRUCTION,
) -> list[dict[str, str]]:
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


def _normalize_text_response(response_text: Any, tokenizer, response_ids: list[int]) -> str:
    if isinstance(response_text, str):
        return response_text
    if isinstance(response_text, list):
        return "".join(str(x) for x in response_text)
    return tokenizer.decode(response_ids, skip_special_tokens=False)


def _load_records_for_eval(
    data_path: str,
    *,
    prompt_key: str,
    max_prompts: int,
    add_rollout_instruction: bool,
) -> list[dict[str, Any]]:
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

        messages = _normalize_prompt_messages(record[prompt_key])
        if add_rollout_instruction:
            messages = _prepend_rollout_instruction(messages, DEFAULT_ROLLOUT_INSTRUCTION)

        uid = record.get("uid", f"sample_{index:08d}")
        normalized.append(
            {
                "uid": str(uid),
                "source_uid": str(uid),
                "source_index": index,
                "messages": messages,
                "prompt_text": _prompt_text_for_storage(messages),
                "data_source": record.get("data_source"),
            }
        )
    return normalized


def _infer_num_speculative_tokens_from_config(model_path: str, *, trust_remote_code: bool) -> int | None:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config_candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
    ]
    for candidate in config_candidates:
        if candidate is None:
            continue
        for attr_name in ("num_nextn_predict_layers", "n_predict", "num_lookahead_tokens"):
            value = getattr(candidate, attr_name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    return None


def _build_speculative_config(
    *,
    target_model_path: str,
    draft_model_path: str | None,
    speculative_method: str,
    num_speculative_tokens: int | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    method = speculative_method.lower()
    if method == "auto":
        method = "draft_model" if draft_model_path else "mtp"

    spec_config: dict[str, Any] = {"method": method}
    if draft_model_path:
        spec_config["model"] = draft_model_path

    resolved_num_spec_tokens = num_speculative_tokens
    if resolved_num_spec_tokens is None and method == "mtp":
        resolved_num_spec_tokens = _infer_num_speculative_tokens_from_config(
            target_model_path,
            trust_remote_code=trust_remote_code,
        )
        if resolved_num_spec_tokens is None:
            raise ValueError(
                "Failed to infer num_speculative_tokens from target model config. "
                "Please pass --num-speculative-tokens explicitly."
            )

    if method == "draft_model" and draft_model_path is None:
        raise ValueError("speculative_method=draft_model requires --draft-model-path.")

    if resolved_num_spec_tokens is not None:
        spec_config["num_speculative_tokens"] = int(resolved_num_spec_tokens)

    return spec_config


def _split_jobs_round_robin(jobs: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, job in enumerate(jobs):
        shards[idx % num_workers].append(job)
    return shards


def _build_worker_engine_config(
    args,
    *,
    visible_devices: str,
    tensor_parallel_size: int,
    speculative_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_model_path": args.target_model_path,
        "tokenizer_path": args.tokenizer_path or args.target_model_path,
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": args.dtype,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(args.max_num_seqs),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "enforce_eager": bool(args.enforce_eager),
        "load_format": args.load_format,
        "stream_interval": 1,
        "visible_devices": visible_devices,
        "tensor_parallel_size": int(tensor_parallel_size),
        "speculative_config": speculative_config,
        "strict_draft_probs": bool(args.strict_draft_probs),
        "worker_extension_cls": (
            STRICT_DRAFT_PROBS_WORKER_EXTENSION if bool(args.strict_draft_probs) else ""
        ),
        "draft_temperature": args.draft_temperature,
        "strict_rejection_debug_jsonl": args.strict_rejection_debug_jsonl,
    }


async def _run_single_job(
    engine,
    *,
    job: dict[str, Any],
    sampling_defaults: dict[str, Any],
    request_prefix: str,
) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    sampling_params = SamplingParams(
        temperature=sampling_defaults["temperature"],
        top_p=sampling_defaults["top_p"],
        top_k=sampling_defaults["top_k"],
        max_tokens=sampling_defaults["max_tokens"],
        seed=job.get("seed"),
        detokenize=False,
        skip_special_tokens=False,
        output_kind=RequestOutputKind.DELTA,
    )
    prompt = TokensPrompt(prompt_token_ids=list(job["prompt_ids"]))
    generator = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=f"{request_prefix}_{job['job_id']}",
    )

    final_output = None
    delta_token_chunks: list[list[int]] = []
    async for output in generator:
        final_output = output
        delta_ids = [int(token_id) for token_id in output.outputs[0].token_ids]
        if delta_ids:
            delta_token_chunks.append(delta_ids)

    if final_output is None:
        raise RuntimeError(f"vLLM returned no output for job_id={job['job_id']}")

    traced = summarize_speculative_chunks(delta_token_chunks)
    response_ids = traced["token_ids"]
    if sum(traced["spec_accept_lens"]) != len(response_ids):
        raise RuntimeError(
            "Speculative trace alignment mismatch for job_id="
            f"{job['job_id']}: sum(spec_accept_lens)={sum(traced['spec_accept_lens'])} "
            f"vs len(response_ids)={len(response_ids)}"
        )

    return {
        "job_id": int(job["job_id"]),
        "response_ids": response_ids,
        "spec_verify_ct": int(traced["spec_verify_ct"]),
        "spec_accepted_tokens": int(traced["spec_accepted_tokens"]),
        "spec_accept_lens": traced["spec_accept_lens"],
        "completion_tokens": int(len(response_ids)),
    }


async def _run_worker_async(
    *,
    worker_idx: int,
    worker_jobs: list[dict[str, Any]],
    event_queue,
    engine_config: dict[str, Any],
    sampling_defaults: dict[str, Any],
) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=engine_config["target_model_path"],
        tokenizer=engine_config["tokenizer_path"],
        trust_remote_code=engine_config["trust_remote_code"],
        tensor_parallel_size=engine_config["tensor_parallel_size"],
        dtype=engine_config["dtype"],
        gpu_memory_utilization=engine_config["gpu_memory_utilization"],
        max_model_len=engine_config["max_model_len"],
        max_num_batched_tokens=engine_config["max_num_batched_tokens"],
        max_num_seqs=engine_config["max_num_seqs"],
        enable_chunked_prefill=engine_config["enable_chunked_prefill"],
        enable_prefix_caching=engine_config["enable_prefix_caching"],
        enforce_eager=engine_config["enforce_eager"],
        load_format=engine_config["load_format"],
        distributed_executor_backend="mp",
        disable_log_stats=True,
        stream_interval=engine_config["stream_interval"],
        speculative_config=engine_config["speculative_config"],
        worker_extension_cls=engine_config["worker_extension_cls"],
    )

    engine = AsyncLLM.from_engine_args(engine_args, usage_context=UsageContext.ENGINE_CONTEXT)
    if engine_config["strict_draft_probs"]:
        patch_results = await engine.collective_rpc("enable_strict_draft_probs_patch")
        if not any(bool(result) for result in patch_results):
            raise RuntimeError("Strict draft_probs patch was requested but was not applied on any vLLM worker.")
        await engine.collective_rpc(
            "set_strict_draft_temperature",
            args=(engine_config["draft_temperature"],),
        )
        if engine_config["strict_rejection_debug_jsonl"]:
            await engine.collective_rpc(
                "set_strict_rejection_debug_jsonl",
                args=(engine_config["strict_rejection_debug_jsonl"],),
            )
    pending = [
        asyncio.create_task(
            _run_single_job(
                engine,
                job=job,
                sampling_defaults=sampling_defaults,
                request_prefix=f"worker_{worker_idx}",
            )
        )
        for job in worker_jobs
    ]

    try:
        for task in asyncio.as_completed(pending):
            result = await task
            event_queue.put(("result", result))
    finally:
        engine.shutdown()


def _worker_main(
    worker_idx: int,
    worker_jobs: list[dict[str, Any]],
    event_queue,
    engine_config: dict[str, Any],
    sampling_defaults: dict[str, Any],
    tmp_dir: str,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = engine_config["visible_devices"]
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _ensure_repo_on_pythonpath()
        _configure_tmp_env(os.path.join(tmp_dir, f"worker_{worker_idx:02d}"))
        asyncio.run(
            _run_worker_async(
                worker_idx=worker_idx,
                worker_jobs=worker_jobs,
                event_queue=event_queue,
                engine_config=engine_config,
                sampling_defaults=sampling_defaults,
            )
        )
        event_queue.put(("done", worker_idx))
    except Exception:
        event_queue.put(("error", worker_idx, traceback.format_exc()))
        raise


def _build_jobs(records: list[dict[str, Any]], *, samples_per_prompt: int, seed_base: int | None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for record in records:
        for sample_idx in range(samples_per_prompt):
            seed = None
            if seed_base is not None:
                seed = seed_base + record["source_index"] * samples_per_prompt + sample_idx
            jobs.append(
                {
                    "job_id": len(jobs),
                    "uid": f'{record["uid"]}_sample_{sample_idx:02d}',
                    "source_uid": record["source_uid"],
                    "source_index": record["source_index"],
                    "sample_idx": sample_idx,
                    "prompt": record["prompt_text"],
                    "prompt_ids": record["prompt_ids"],
                    "data_source": record.get("data_source"),
                    "seed": seed,
                }
            )
    return jobs


def _build_output_payload(*, tokenizer, job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    response_ids = [int(token_id) for token_id in result["response_ids"]]
    response_text = _normalize_text_response(None, tokenizer, response_ids)
    return {
        "uid": job["uid"],
        "prompt": job["prompt"],
        "response": response_text,
        "prompt_ids": job["prompt_ids"],
        "response_ids": response_ids,
        "sample_idx": job["sample_idx"],
        "source_uid": job["source_uid"],
        "source_index": job["source_index"],
        "data_source": job.get("data_source"),
        "spec_verify_ct": int(result["spec_verify_ct"]),
        "spec_accepted_tokens": int(result["spec_accepted_tokens"]),
        "spec_accept_lens": [int(x) for x in result["spec_accept_lens"]],
        "completion_tokens": int(result["completion_tokens"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone vLLM block-count eval for waste_sd")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--parallel-mode", choices=("dp", "tp"), default="dp")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--max-prompts", type=int, default=-1)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draft-temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--no-rollout-instruction", action="store_true")
    parser.add_argument(
        "--speculative-method",
        default="auto",
        choices=("auto", "mtp", "draft_model", "eagle", "eagle3", "medusa", "mlp_speculator", "ngram", "suffix"),
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--strict-draft-probs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-rejection-debug-jsonl", default=None)
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_tmp_env(args.tmp_dir)

    tokenizer_path = args.tokenizer_path or args.target_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_records_for_eval(
        args.input_data,
        prompt_key=args.prompt_key,
        max_prompts=args.max_prompts,
        add_rollout_instruction=not args.no_rollout_instruction,
    )
    for record in records:
        record["prompt_ids"] = _apply_chat_template(tokenizer, record["messages"], enable_thinking=args.enable_thinking)

    jobs = _build_jobs(records, samples_per_prompt=args.samples_per_prompt, seed_base=args.seed_base)
    speculative_config = _build_speculative_config(
        target_model_path=args.target_model_path,
        draft_model_path=args.draft_model_path,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if args.strict_draft_probs and speculative_config.get("method") != "draft_model":
        raise ValueError("--strict-draft-probs currently only supports speculative_method=draft_model.")
    if args.draft_temperature is not None and not args.strict_draft_probs:
        raise ValueError("--draft-temperature currently requires --strict-draft-probs.")
    if args.draft_temperature is not None and args.draft_temperature < 0:
        raise ValueError("--draft-temperature must be non-negative.")
    if args.strict_rejection_debug_jsonl and not args.strict_draft_probs:
        raise ValueError("--strict-rejection-debug-jsonl requires --strict-draft-probs.")
    meta = {
        "backend": "vllm_standalone",
        "target_model_path": args.target_model_path,
        "draft_model_path": args.draft_model_path,
        "tokenizer_path": tokenizer_path,
        "input_data": args.input_data,
        "prompt_key": args.prompt_key,
        "samples_per_prompt": args.samples_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "draft_temperature": args.draft_temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "parallel_mode": args.parallel_mode,
        "num_prompts": len(records),
        "num_outputs": len(jobs),
        "speculative_method": speculative_config.get("method", args.speculative_method),
        "num_speculative_tokens": speculative_config.get("num_speculative_tokens"),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "stream_interval": 1,
        "strict_draft_probs": bool(args.strict_draft_probs),
        "strict_rejection_debug_jsonl": args.strict_rejection_debug_jsonl,
        "speculative_config": speculative_config,
    }
    with open(output_path.with_suffix(output_path.suffix + ".meta.json"), "w", encoding="utf-8") as meta_f:
        json.dump(meta, meta_f, ensure_ascii=False, indent=2)

    if not jobs:
        output_path.write_text("", encoding="utf-8")
        return

    devices = _parse_visible_devices(args.visible_devices)
    if args.parallel_mode == "tp":
        tp_size = args.tp_size or len(devices)
        if tp_size <= 0:
            raise ValueError("tp_size must be positive.")
        if tp_size > len(devices):
            raise ValueError(
                f"tp_size={tp_size} exceeds number of visible devices={len(devices)} for TP mode."
            )
        worker_device_sets = [",".join(str(device) for device in devices[:tp_size])]
        worker_job_shards = [jobs]
        worker_engine_configs = [
            _build_worker_engine_config(
                args,
                visible_devices=worker_device_sets[0],
                tensor_parallel_size=tp_size,
                speculative_config=speculative_config,
            )
        ]
    else:
        num_workers = min(len(devices), len(jobs))
        worker_device_sets = [str(device) for device in devices[:num_workers]]
        worker_job_shards = _split_jobs_round_robin(jobs, num_workers)
        worker_engine_configs = [
            _build_worker_engine_config(
                args,
                visible_devices=device_set,
                tensor_parallel_size=1,
                speculative_config=speculative_config,
            )
            for device_set in worker_device_sets
        ]

    meta["num_workers"] = len(worker_job_shards)

    sampling_defaults = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": -1 if int(args.top_k) == 0 else int(args.top_k),
        "max_tokens": int(args.max_new_tokens),
    }

    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    processes: list[mp.Process] = []
    completed_job_ids: set[int] = set()
    job_by_id = {int(job["job_id"]): job for job in jobs}

    try:
        for worker_idx, (worker_jobs, engine_config) in enumerate(zip(worker_job_shards, worker_engine_configs, strict=False)):
            process = ctx.Process(
                target=_worker_main,
                args=(
                    worker_idx,
                    worker_jobs,
                    event_queue,
                    engine_config,
                    sampling_defaults,
                    args.tmp_dir,
                ),
            )
            process.start()
            processes.append(process)

        finished_workers = 0
        completed_jobs = 0
        with open(output_path, "w", encoding="utf-8", buffering=1) as out_f:
            with tqdm(total=len(jobs), desc="eval_block_counts_vllm", dynamic_ncols=True, disable=args.disable_tqdm) as progress:
                while completed_jobs < len(jobs) or finished_workers < len(processes):
                    try:
                        message = event_queue.get(timeout=1.0)
                    except queue.Empty:
                        failed_processes = [process for process in processes if process.exitcode not in (None, 0)]
                        if failed_processes:
                            failed = failed_processes[0]
                            raise RuntimeError(f"vLLM worker exited with code {failed.exitcode}")
                        continue

                    tag = message[0]
                    if tag == "result":
                        result = message[1]
                        job_id = int(result["job_id"])
                        if job_id in completed_job_ids:
                            raise ValueError(f"Duplicate result for job_id={job_id}")
                        completed_job_ids.add(job_id)
                        payload = _build_output_payload(tokenizer=tokenizer, job=job_by_id[job_id], result=result)
                        out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        out_f.flush()
                        completed_jobs += 1
                        progress.update(1)
                        if (
                            args.log_every > 0
                            and (completed_jobs % args.log_every == 0 or completed_jobs == len(jobs))
                        ):
                            print(f"[eval_block_counts_vllm] completed {completed_jobs}/{len(jobs)}", flush=True)
                    elif tag == "done":
                        finished_workers += 1
                    elif tag == "error":
                        worker_idx = int(message[1])
                        worker_traceback = str(message[2])
                        raise RuntimeError(f"vLLM worker {worker_idx} failed:\n{worker_traceback}")

        if completed_jobs != len(jobs):
            raise ValueError(f"Expected {len(jobs)} completed jobs, got {completed_jobs}.")

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"vLLM worker exited with code {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
        event_queue.close()
        event_queue.join_thread()


if __name__ == "__main__":
    main()
