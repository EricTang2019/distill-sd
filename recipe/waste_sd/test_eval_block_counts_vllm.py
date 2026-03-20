from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import recipe.waste_sd.eval_block_counts_vllm as module


def test_load_records_for_eval_respects_rollout_instruction_flag(tmp_path):
    input_path = tmp_path / "data.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "What is 2 + 2?"}, ensure_ascii=True) + "\n")

    with_instruction = module._load_records_for_eval(
        str(input_path),
        prompt_key="prompt",
        max_prompts=-1,
        add_rollout_instruction=True,
    )
    without_instruction = module._load_records_for_eval(
        str(input_path),
        prompt_key="prompt",
        max_prompts=-1,
        add_rollout_instruction=False,
    )

    assert module.DEFAULT_ROLLOUT_INSTRUCTION in with_instruction[0]["messages"][0]["content"]
    assert module.DEFAULT_ROLLOUT_INSTRUCTION not in without_instruction[0]["messages"][0]["content"]


def test_build_speculative_config_auto_uses_draft_model_when_present():
    spec_config = module._build_speculative_config(
        target_model_path="/tmp/target",
        draft_model_path="/tmp/draft",
        speculative_method="auto",
        num_speculative_tokens=5,
        trust_remote_code=False,
    )

    assert spec_config == {
        "method": "draft_model",
        "model": "/tmp/draft",
        "num_speculative_tokens": 5,
    }


def test_build_speculative_config_auto_infers_mtp_num_tokens(monkeypatch):
    monkeypatch.setattr(module, "_infer_num_speculative_tokens_from_config", lambda *args, **kwargs: 3)

    spec_config = module._build_speculative_config(
        target_model_path="/tmp/target",
        draft_model_path=None,
        speculative_method="auto",
        num_speculative_tokens=None,
        trust_remote_code=False,
    )

    assert spec_config == {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }


def test_split_jobs_round_robin_preserves_worker_balance():
    jobs = [{"job_id": idx} for idx in range(5)]

    shards = module._split_jobs_round_robin(jobs, 2)

    assert [job["job_id"] for job in shards[0]] == [0, 2, 4]
    assert [job["job_id"] for job in shards[1]] == [1, 3]


def test_build_output_payload_decodes_and_preserves_metadata():
    class _DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return "|".join(str(token_id) for token_id in token_ids)

    job = {
        "uid": "sample_00000000_sample_00",
        "prompt": "What is 2 + 2?",
        "prompt_ids": [11, 12, 13],
        "sample_idx": 0,
        "source_uid": "sample_00000000",
        "source_index": 0,
        "data_source": "unit_test",
    }
    result = {
        "response_ids": [21, 22, 23],
        "spec_verify_ct": 2,
        "spec_accepted_tokens": 1,
        "spec_accept_lens": [1, 2],
        "completion_tokens": 3,
    }

    payload = module._build_output_payload(tokenizer=_DummyTokenizer(), job=job, result=result)

    assert payload["uid"] == job["uid"]
    assert payload["prompt"] == job["prompt"]
    assert payload["response"] == "21|22|23"
    assert payload["prompt_ids"] == [11, 12, 13]
    assert payload["response_ids"] == [21, 22, 23]
    assert payload["spec_verify_ct"] == 2
    assert payload["spec_accepted_tokens"] == 1
    assert payload["spec_accept_lens"] == [1, 2]
    assert payload["completion_tokens"] == 3
    assert payload["data_source"] == "unit_test"


def test_build_worker_engine_config_sets_strict_worker_extension():
    args = SimpleNamespace(
        target_model_path="/tmp/target",
        tokenizer_path=None,
        trust_remote_code=False,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=None,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        load_format="auto",
        strict_draft_probs=True,
        draft_temperature=0.5,
        strict_rejection_debug_jsonl="/tmp/reject_debug.jsonl",
    )

    config = module._build_worker_engine_config(
        args,
        visible_devices="0",
        tensor_parallel_size=1,
        speculative_config={"method": "draft_model", "model": "/tmp/draft", "num_speculative_tokens": 3},
    )

    assert config["strict_draft_probs"] is True
    assert config["worker_extension_cls"] == module.STRICT_DRAFT_PROBS_WORKER_EXTENSION
    assert config["draft_temperature"] == 0.5
    assert config["strict_rejection_debug_jsonl"] == "/tmp/reject_debug.jsonl"


def test_main_rejects_draft_temperature_without_strict(monkeypatch, tmp_path):
    input_path = tmp_path / "data.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("", encoding="utf-8")

    class _DummyTokenizer:
        pad_token_id = None
        eos_token = "</s>"
        pad_token = None

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_block_counts_vllm.py",
            "--target-model-path",
            "/tmp/target",
            "--draft-model-path",
            "/tmp/draft",
            "--input-data",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--visible-devices",
            "0",
            "--parallel-mode",
            "dp",
            "--speculative-method",
            "draft_model",
            "--num-speculative-tokens",
            "3",
            "--draft-temperature",
            "0.5",
            "--no-rollout-instruction",
        ],
    )

    try:
        module.main()
    except ValueError as exc:
        assert "--draft-temperature currently requires --strict-draft-probs" in str(exc)
    else:
        raise AssertionError("Expected --draft-temperature without --strict-draft-probs to raise ValueError")


def test_main_empty_job_path_writes_meta_without_touching_vllm(monkeypatch, tmp_path):
    input_path = tmp_path / "data.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("", encoding="utf-8")

    class _DummyTokenizer:
        pad_token_id = None
        eos_token = "</s>"
        pad_token = None

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _DummyTokenizer())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_block_counts_vllm.py",
            "--target-model-path",
            "/tmp/target",
            "--draft-model-path",
            "/tmp/draft",
            "--input-data",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--visible-devices",
            "0",
            "--parallel-mode",
            "dp",
            "--speculative-method",
            "draft_model",
            "--num-speculative-tokens",
            "3",
            "--no-rollout-instruction",
        ],
    )

    module.main()

    assert output_path.read_text(encoding="utf-8") == ""
    meta = json.loads(output_path.with_suffix(output_path.suffix + ".meta.json").read_text(encoding="utf-8"))
    assert meta["speculative_method"] == "draft_model"
    assert meta["num_outputs"] == 0
