from __future__ import annotations

import argparse
import json
import subprocess
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
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
    _find_free_port,
    _launch_teacher_server,
    _normalize_text_response,
    _parse_visible_devices,
    _prepend_rollout_instruction,
    _prompt_text_for_storage,
    _wait_for_teacher_server,
)

if TYPE_CHECKING:
    from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter


ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "human": "user",
    "gpt": "assistant",
}


@dataclass
class ConversationState:
    uid: str
    source_index: int
    data_source: str | None
    original_messages: list[dict[str, str]]
    regenerated_messages: list[dict[str, str]] = field(default_factory=list)
    cursor: int = 0
    generated_turns: int = 0


def _load_raw_records(data_files: list[str], *, max_conversations: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for data_file in data_files:
        path = Path(data_file)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
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
            records.extend(payload)
        elif suffix == ".parquet":
            records.extend(pd.read_parquet(path).to_dict(orient="records"))
        else:
            raise ValueError(f"Unsupported dataset format: {path}")
        if max_conversations > 0 and len(records) >= max_conversations:
            return records[:max_conversations]
    return records


def _normalize_conversation_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        if "messages" in value:
            return _normalize_conversation_messages(value["messages"])
        if "conversations" in value:
            return _normalize_conversation_messages(value["conversations"])

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Unsupported conversation format: {type(value).__name__}")

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Conversation turn at index {idx} must be a dict, got {type(item).__name__}")

        if "role" in item and "content" in item:
            raw_role = str(item["role"])
            content = str(item["content"])
        elif "from" in item and "value" in item:
            raw_role = str(item["from"])
            content = str(item["value"])
        else:
            raise ValueError(
                "Conversation turn must contain either role/content or from/value; "
                f"got keys={sorted(item.keys())}"
            )

        role = ROLE_ALIASES.get(raw_role.lower())
        if role is None:
            raise ValueError(f"Unsupported conversation role: {raw_role!r}")
        messages.append({"role": role, "content": content})
    return messages


def _select_conversation_value(record: dict[str, Any], conversation_key: str) -> Any:
    if conversation_key != "auto":
        if conversation_key not in record:
            raise ValueError(f"Missing conversation key {conversation_key!r} in record with keys={sorted(record.keys())}")
        return record[conversation_key]

    for candidate in ("messages", "conversations"):
        if candidate in record:
            return record[candidate]
    raise ValueError(
        "Failed to infer conversation field automatically. Expected one of {'messages', 'conversations'} "
        f"in record keys={sorted(record.keys())}"
    )


def _build_states(
    records: list[dict[str, Any]],
    *,
    conversation_key: str,
    max_turns_per_conversation: int,
) -> list[ConversationState]:
    states: list[ConversationState] = []
    for index, record in enumerate(records):
        messages = _normalize_conversation_messages(_select_conversation_value(record, conversation_key))
        if max_turns_per_conversation > 0:
            user_seen = 0
            trimmed: list[dict[str, str]] = []
            for message in messages:
                if message["role"] == "user":
                    user_seen += 1
                trimmed.append(message)
                if user_seen >= max_turns_per_conversation:
                    break
            messages = trimmed
        states.append(
            ConversationState(
                uid=str(record.get("uid", f"sample_{index:08d}")),
                source_index=index,
                data_source=None if record.get("data_source") is None else str(record.get("data_source")),
                original_messages=messages,
            )
        )
    return states


def _count_pending_turns(messages: list[dict[str, str]]) -> int:
    return sum(1 for message in messages if message["role"] == "user")


def _build_next_regeneration_job(
    state: ConversationState,
    *,
    add_rollout_instruction: bool,
) -> dict[str, Any] | None:
    while state.cursor < len(state.original_messages):
        message = state.original_messages[state.cursor]
        state.cursor += 1

        if message["role"] == "assistant":
            continue

        state.regenerated_messages.append(dict(message))
        if message["role"] != "user":
            continue

        prompt_messages = [dict(item) for item in state.regenerated_messages]
        if add_rollout_instruction:
            prompt_messages = _prepend_rollout_instruction(prompt_messages, DEFAULT_ROLLOUT_INSTRUCTION)

        turn_index = state.generated_turns
        job_uid = f"{state.uid}__turn_{turn_index:03d}"
        return {
            "uid": job_uid,
            "conversation_uid": state.uid,
            "turn_index": turn_index,
            "source_index": state.source_index,
            "data_source": state.data_source,
            "messages": prompt_messages,
            "prompt_text": _prompt_text_for_storage(prompt_messages),
        }
    return None


def _apply_generated_response(state: ConversationState, *, response_text: str) -> None:
    state.regenerated_messages.append({"role": "assistant", "content": response_text})
    state.generated_turns += 1


def _launch_teacher_adapters(args, *, output_dir: Path) -> tuple[list[subprocess.Popen], list[HttpServerAdapter]]:
    from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter

    teacher_devices = _parse_visible_devices(args.teacher_visible_devices)
    server_processes: list[subprocess.Popen] = []
    adapters: list[HttpServerAdapter] = []

    if args.parallel_mode == "dp":
        for replica_idx, device_id in enumerate(teacher_devices):
            teacher_port = _find_free_port()
            server_kwargs = {
                "host": args.teacher_host,
                "port": teacher_port,
                "model_path": args.teacher_model_path,
                "tensor_parallel_size": 1,
                "api_key": args.teacher_api_key,
                "dtype": None if args.use_sglang_defaults else args.dtype,
                "trust_remote_code": args.trust_remote_code,
                "log_level": "error",
                "attention_backend": None if args.use_sglang_defaults else args.teacher_attention_backend,
                "sampling_backend": None if args.use_sglang_defaults else args.teacher_sampling_backend,
                "disable_cuda_graph": args.teacher_disable_cuda_graph,
                "mem_fraction_static": None if args.use_sglang_defaults else args.teacher_mem_fraction_static,
                "chunked_prefill_size": None if args.use_sglang_defaults else args.teacher_chunked_prefill_size,
                "max_total_tokens": None if args.use_sglang_defaults else args.teacher_max_total_tokens,
            }
            teacher_log_path = output_dir / f"teacher_sglang_dp_{replica_idx}.log"
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
                    timeout=args.request_timeout_s,
                    max_attempts=args.request_max_attempts,
                    retry_delay=args.request_retry_delay_s,
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
            "dtype": None if args.use_sglang_defaults else args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "log_level": "error",
            "attention_backend": None if args.use_sglang_defaults else args.teacher_attention_backend,
            "sampling_backend": None if args.use_sglang_defaults else args.teacher_sampling_backend,
            "disable_cuda_graph": args.teacher_disable_cuda_graph,
            "mem_fraction_static": None if args.use_sglang_defaults else args.teacher_mem_fraction_static,
            "chunked_prefill_size": None if args.use_sglang_defaults else args.teacher_chunked_prefill_size,
            "max_total_tokens": None if args.use_sglang_defaults else args.teacher_max_total_tokens,
        }
        teacher_log_path = output_dir / "teacher_sglang.log"
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
                timeout=args.request_timeout_s,
                max_attempts=args.request_max_attempts,
                retry_delay=args.request_retry_delay_s,
                host=args.teacher_host,
                port=teacher_port,
                node_rank=0,
                model_path=args.teacher_model_path,
                api_key=args.teacher_api_key,
                launch_server=False,
            )
        )

    return server_processes, adapters


def _shutdown_teacher_runtime(
    *,
    adapters: list[HttpServerAdapter],
    server_processes: list[subprocess.Popen],
    api_key: str,
) -> None:
    for adapter in adapters:
        try:
            requests.get(
                f"http://{adapter.server_args.host}:{adapter.server_args.port}/shutdown",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
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


def _collect_streaming_jobs(
    states: list[ConversationState],
    *,
    rollout_f,
    adapters: list[HttpServerAdapter],
    tokenizer,
    args,
    total_turns: int,
) -> None:
    ready_states: deque[ConversationState] = deque(states)
    pending: dict[Future, tuple[ConversationState, dict[str, Any]]] = {}
    buffered_payloads: dict[int, dict[str, Any]] = {}
    next_write_index = 0
    next_global_job_index = 0
    completed = 0

    def _prepare_next_job(state: ConversationState) -> dict[str, Any] | None:
        nonlocal next_global_job_index
        job = _build_next_regeneration_job(
            state,
            add_rollout_instruction=bool(args.add_rollout_instruction),
        )
        if job is None:
            return None
        job["wave_index"] = job["turn_index"]
        job["global_job_index"] = next_global_job_index
        job["prompt_ids"] = _apply_chat_template(
            tokenizer,
            job["messages"],
            enable_thinking=bool(args.enable_thinking),
        )
        next_global_job_index += 1
        return job

    def _submit_ready_jobs(executor: ThreadPoolExecutor) -> None:
        max_pending = max(1, args.max_concurrent_requests)
        while ready_states and len(pending) < max_pending:
            state = ready_states.popleft()
            job = _prepare_next_job(state)
            if job is None:
                continue

            seed = None
            if args.seed_base is not None:
                seed = args.seed_base + int(job["global_job_index"])

            sampling_params = _build_sampling_params(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                seed=seed,
            )
            adapter = adapters[int(job["global_job_index"]) % len(adapters)]
            future = executor.submit(
                adapter.generate,
                input_ids=job["prompt_ids"],
                sampling_params=sampling_params,
            )
            pending[future] = (state, job)

    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrent_requests)) as executor:
        _submit_ready_jobs(executor)
        with tqdm(
            total=total_turns,
            desc="collect_specforge_exact_offline_rollout",
            dynamic_ncols=True,
            disable=args.disable_tqdm,
        ) as progress:
            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    state, job = pending.pop(future)
                    response = future.result()
                    response_ids = _extract_response_ids(response)
                    response_text = _normalize_text_response(response.get("text"), tokenizer, response_ids)
                    _apply_generated_response(state, response_text=response_text)

                    buffered_payloads[int(job["global_job_index"])] = {
                        "uid": f'{job["uid"]}_sample_00',
                        "prompt": job["prompt_text"],
                        "response": response_text,
                        "prompt_ids": job["prompt_ids"],
                        "response_ids": response_ids,
                        "sample_idx": 0,
                        "source_uid": job["uid"],
                        "source_index": job["source_index"],
                        "data_source": job.get("data_source"),
                        "conversation_uid": job["conversation_uid"],
                        "turn_index": job["turn_index"],
                        "wave_index": job["wave_index"],
                        "global_job_index": job["global_job_index"],
                    }

                    if state.cursor < len(state.original_messages):
                        ready_states.append(state)

                    completed += 1
                    progress.update(1)
                    if (
                        args.log_every > 0
                        and args.disable_tqdm
                        and (completed % args.log_every == 0 or completed == total_turns)
                    ):
                        print(
                            "[collect_specforge_exact_offline_rollout] "
                            f"completed {completed}/{total_turns} turns"
                        )

                while next_write_index in buffered_payloads:
                    rollout_f.write(json.dumps(buffered_payloads.pop(next_write_index), ensure_ascii=False) + "\n")
                    next_write_index += 1

                _submit_ready_jobs(executor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect SpecForge-exact multiturn teacher rollouts into offline waste_sd dataset format"
    )
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--input-data", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-conversations-jsonl", default=None)
    parser.add_argument("--conversation-key", default="auto")
    parser.add_argument("--teacher-visible-devices", required=True)
    parser.add_argument("--parallel-mode", choices=("dp", "tp"), default="dp")
    parser.add_argument("--teacher-tp-size", type=int, default=None)
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=0)
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--teacher-timeout-s", type=float, default=300.0)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--request-max-attempts", type=int, default=3)
    parser.add_argument("--request-retry-delay-s", type=float, default=2.0)
    parser.add_argument("--use-sglang-defaults", action="store_true")
    parser.add_argument("--teacher-attention-backend", default="fa3")
    parser.add_argument("--teacher-sampling-backend", default="flashinfer")
    parser.add_argument("--teacher-disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--teacher-chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--teacher-max-total-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--max-conversations", type=int, default=-1)
    parser.add_argument("--max-turns-per-conversation", type=int, default=-1)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--add-rollout-instruction", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conversations_output_path = (
        Path(args.output_conversations_jsonl)
        if args.output_conversations_jsonl is not None
        else output_path.with_name(output_path.stem + ".conversations.jsonl")
    )
    conversations_output_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_tmp_env(args.tmp_dir)

    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records = _load_raw_records(args.input_data, max_conversations=args.max_conversations)
    states = _build_states(
        raw_records,
        conversation_key=args.conversation_key,
        max_turns_per_conversation=args.max_turns_per_conversation,
    )
    total_turns = sum(_count_pending_turns(state.original_messages) for state in states)

    metadata = {
        "teacher_model_path": args.teacher_model_path,
        "tokenizer_path": tokenizer_path,
        "input_data": args.input_data,
        "conversation_key": args.conversation_key,
        "max_conversations": args.max_conversations,
        "max_turns_per_conversation": args.max_turns_per_conversation,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "add_rollout_instruction": bool(args.add_rollout_instruction),
        "parallel_mode": args.parallel_mode,
        "num_conversations": len(states),
        "num_outputs": total_turns,
        "output_conversations_jsonl": str(conversations_output_path),
    }
    with open(output_path.with_suffix(output_path.suffix + ".meta.json"), "w", encoding="utf-8") as meta_f:
        json.dump(metadata, meta_f, ensure_ascii=False, indent=2)

    if total_turns == 0:
        output_path.write_text("", encoding="utf-8")
        conversations_output_path.write_text("", encoding="utf-8")
        return

    server_processes, adapters = _launch_teacher_adapters(args, output_dir=output_path.parent)
    try:
        with open(output_path, "w", encoding="utf-8") as rollout_f:
            _collect_streaming_jobs(
                states,
                rollout_f=rollout_f,
                adapters=adapters,
                tokenizer=tokenizer,
                args=args,
                total_turns=total_turns,
            )

        with open(conversations_output_path, "w", encoding="utf-8") as conv_f:
            for state in states:
                payload = {
                    "uid": state.uid,
                    "data_source": state.data_source,
                    "source_index": state.source_index,
                    "messages": state.regenerated_messages,
                    "num_generated_turns": state.generated_turns,
                }
                conv_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        _shutdown_teacher_runtime(
            adapters=adapters,
            server_processes=server_processes,
            api_key=args.teacher_api_key,
        )


if __name__ == "__main__":
    main()
