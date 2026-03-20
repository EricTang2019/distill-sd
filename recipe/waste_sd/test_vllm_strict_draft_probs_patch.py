from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler

from recipe.waste_sd.vllm_strict_draft_probs_patch import (
    DraftProposalCacheEntry,
    StrictDraftProbsWorkerExtension,
    _clear_strict_draft_cache,
    _flatten_strict_draft_probs,
    _resolve_strict_debug_jsonl_path,
    _sample_draft_tokens_with_probs,
    _single_step_rejection_debug_row,
    apply_strict_draft_probs_patch,
)


def _make_sampling_metadata(
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generators: dict[int, torch.Generator] | None = None,
    all_greedy: bool = False,
    all_random: bool = True,
) -> SamplingMetadata:
    batch_size = temperature.shape[0]
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p,
        top_k=top_k,
        generators=generators or {},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_sample_draft_tokens_with_probs_is_deterministic_with_generators():
    logits = torch.tensor([[0.1, 2.0, 0.3], [1.2, 0.7, -0.4]], dtype=torch.float32)
    metadata_a = _make_sampling_metadata(
        temperature=torch.tensor([1.0, 1.0]),
        generators={
            0: torch.Generator().manual_seed(11),
            1: torch.Generator().manual_seed(29),
        },
    )
    metadata_b = _make_sampling_metadata(
        temperature=torch.tensor([1.0, 1.0]),
        generators={
            0: torch.Generator().manual_seed(11),
            1: torch.Generator().manual_seed(29),
        },
    )

    token_ids_a, probs_a = _sample_draft_tokens_with_probs(logits, metadata_a)
    token_ids_b, probs_b = _sample_draft_tokens_with_probs(logits, metadata_b)

    assert torch.equal(token_ids_a, token_ids_b)
    assert torch.allclose(probs_a, probs_b)
    assert torch.allclose(probs_a.sum(dim=-1), torch.ones(2))


def test_sample_draft_tokens_with_probs_applies_top_k():
    logits = torch.tensor([[0.1, 2.0, 0.3]], dtype=torch.float32)
    metadata = _make_sampling_metadata(
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1], dtype=torch.int32),
    )

    token_ids, probs = _sample_draft_tokens_with_probs(logits, metadata)

    assert int(token_ids.item()) == 1
    assert torch.isclose(probs[0, 1], torch.tensor(1.0))
    assert torch.isclose(probs[0, 0], torch.tensor(0.0))
    assert torch.isclose(probs[0, 2], torch.tensor(0.0))


def test_sample_draft_tokens_with_probs_matches_vllm_sampler_for_supported_path():
    logits = torch.tensor([[0.2, 1.5, -0.4, 0.7], [1.1, -0.8, 0.3, 0.6]], dtype=torch.float32)
    metadata_a = _make_sampling_metadata(
        temperature=torch.tensor([1.0, 0.8]),
        top_k=torch.tensor([3, 2], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
        generators={
            0: torch.Generator().manual_seed(7),
            1: torch.Generator().manual_seed(19),
        },
    )
    metadata_b = _make_sampling_metadata(
        temperature=torch.tensor([1.0, 0.8]),
        top_k=torch.tensor([3, 2], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
        generators={
            0: torch.Generator().manual_seed(7),
            1: torch.Generator().manual_seed(19),
        },
    )

    token_ids, probs = _sample_draft_tokens_with_probs(logits.clone(), metadata_a)

    sampler = Sampler()
    sampled, _ = sampler.sample(logits.clone().to(torch.float32), metadata_b)
    expected_logits = Sampler.apply_temperature(
        logits.clone().to(torch.float32),
        metadata_b.temperature,
        metadata_b.all_random,
    )
    expected_logits = apply_top_k_top_p(expected_logits, metadata_b.top_k, metadata_b.top_p)
    expected_probs = expected_logits.softmax(dim=-1, dtype=torch.float32)

    assert torch.equal(token_ids, sampled)
    assert torch.allclose(probs, expected_probs)


def test_sample_draft_tokens_with_probs_override_uses_draft_temperature():
    logits = torch.tensor([[0.2, 1.5, -0.4, 0.7], [1.1, -0.8, 0.3, 0.6]], dtype=torch.float32)
    metadata = _make_sampling_metadata(
        temperature=torch.tensor([1.0, 1.0]),
        top_k=torch.tensor([3, 2], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
        generators={
            0: torch.Generator().manual_seed(7),
            1: torch.Generator().manual_seed(19),
        },
    )
    expected_metadata = _make_sampling_metadata(
        temperature=torch.tensor([0.5, 0.5]),
        top_k=torch.tensor([3, 2], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
        generators={
            0: torch.Generator().manual_seed(7),
            1: torch.Generator().manual_seed(19),
        },
    )

    token_ids, probs = _sample_draft_tokens_with_probs(
        logits.clone(),
        metadata,
        draft_temperature_override=0.5,
    )

    sampler = Sampler()
    sampled, _ = sampler.sample(logits.clone().to(torch.float32), expected_metadata)
    expected_logits = Sampler.apply_temperature(
        logits.clone().to(torch.float32),
        expected_metadata.temperature,
        expected_metadata.all_random,
    )
    expected_logits = apply_top_k_top_p(expected_logits, expected_metadata.top_k, expected_metadata.top_p)
    expected_probs = expected_logits.softmax(dim=-1, dtype=torch.float32)

    assert torch.equal(token_ids, sampled)
    assert torch.allclose(probs, expected_probs)


def test_sample_draft_tokens_with_probs_greedy_override_returns_one_hot_q():
    logits = torch.tensor([[0.1, 2.0, 0.3]], dtype=torch.float32)
    metadata = _make_sampling_metadata(
        temperature=torch.tensor([1.0]),
        generators={0: torch.Generator().manual_seed(11)},
    )

    token_ids, probs = _sample_draft_tokens_with_probs(
        logits,
        metadata,
        draft_temperature_override=0.0,
    )

    assert int(token_ids.item()) == 1
    assert torch.equal(probs, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32))


def test_flatten_strict_draft_probs_validates_token_alignment():
    cache = {
        "req_a": DraftProposalCacheEntry(
            token_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
            probs=torch.arange(12, dtype=torch.float32).view(3, 4),
        ),
        "req_b": DraftProposalCacheEntry(
            token_ids=torch.tensor([20, 21, 22], dtype=torch.int32),
            probs=torch.arange(12, 24, dtype=torch.float32).view(3, 4),
        ),
    }

    flat_probs, consumed = _flatten_strict_draft_probs(
        req_ids=["req_a", "req_b"],
        num_draft_tokens=[2, 1],
        cache=cache,
        expected_draft_token_ids=torch.tensor([10, 11, 20], dtype=torch.int32),
    )

    assert consumed == ["req_a", "req_b"]
    assert flat_probs.shape == (3, 4)
    assert torch.equal(flat_probs[0], cache["req_a"].probs[0])
    assert torch.equal(flat_probs[2], cache["req_b"].probs[0])


def test_flatten_strict_draft_probs_skips_zero_length_requests():
    cache = {
        "req_a": DraftProposalCacheEntry(
            token_ids=torch.tensor([10], dtype=torch.int32),
            probs=torch.arange(4, dtype=torch.float32).view(1, 4),
        ),
        "req_b": DraftProposalCacheEntry(
            token_ids=torch.tensor([20], dtype=torch.int32),
            probs=torch.arange(4, 8, dtype=torch.float32).view(1, 4),
        ),
    }

    flat_probs, consumed = _flatten_strict_draft_probs(
        req_ids=["req_a", "req_b"],
        num_draft_tokens=[0, 1],
        cache=cache,
        expected_draft_token_ids=torch.tensor([20], dtype=torch.int32),
    )

    assert consumed == ["req_b"]
    assert flat_probs.shape == (1, 4)
    assert torch.equal(flat_probs[0], cache["req_b"].probs[0])


def test_flatten_strict_draft_probs_raises_on_mismatch():
    cache = {
        "req_a": DraftProposalCacheEntry(
            token_ids=torch.tensor([10, 11], dtype=torch.int32),
            probs=torch.ones(2, 3),
        )
    }

    try:
        _flatten_strict_draft_probs(
            req_ids=["req_a"],
            num_draft_tokens=[2],
            cache=cache,
            expected_draft_token_ids=torch.tensor([10, 99], dtype=torch.int32),
        )
    except RuntimeError as exc:
        assert "alignment mismatch" in str(exc)
    else:
        raise AssertionError("Expected token alignment mismatch to raise RuntimeError")


def test_single_step_rejection_debug_row_matches_theory():
    row = _single_step_rejection_debug_row(
        req_id="req_1",
        draft_token_id=1,
        q_row=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32),
        p_row=torch.tensor([0.4, 0.25, 0.35], dtype=torch.float32),
        output_token_ids=[2],
    )

    assert row["req_id"] == "req_1"
    assert row["accepted"] is False
    assert row["q_draft"] == 0.5
    assert row["p_draft"] == 0.25
    assert row["accept_prob_for_sampled_token"] == 0.5
    assert torch.isclose(torch.tensor(row["exact_accept_rate"]), torch.tensor(0.75))


def test_resolve_strict_debug_jsonl_path_adds_pid_suffix():
    path = _resolve_strict_debug_jsonl_path("/tmp/strict_debug.jsonl")
    assert path is not None
    assert path.parent.as_posix() == "/tmp"
    assert path.name.startswith("strict_debug.pid")
    assert path.name.endswith(".jsonl")


def test_clear_strict_draft_cache_removes_requested_entries():
    cache = {
        "req_a": DraftProposalCacheEntry(token_ids=torch.tensor([1], dtype=torch.int32), probs=torch.ones(1, 2)),
        "req_b": DraftProposalCacheEntry(token_ids=torch.tensor([2], dtype=torch.int32), probs=torch.ones(1, 2)),
    }

    _clear_strict_draft_cache(cache, {"req_a"})

    assert "req_a" not in cache
    assert "req_b" in cache


def test_apply_strict_draft_probs_patch_is_idempotent():
    class _FakeDrafter:
        def propose(self, *args, **kwargs):
            return torch.zeros((1, 1), dtype=torch.int64)

    class _FakeRunner:
        def __init__(self):
            self.speculative_config = SimpleNamespace(method="draft_model", disable_padded_drafter_batch=False)
            self.drafter = _FakeDrafter()
            self.use_async_scheduling = False

        def _sample(self, logits, spec_decode_metadata):
            return "sample"

        def propose_draft_token_ids(self, *args, **kwargs):
            return torch.zeros((1, 1), dtype=torch.int64)

        def _update_states(self, scheduler_output):
            return scheduler_output

    runner = _FakeRunner()

    assert apply_strict_draft_probs_patch(runner) is True
    assert apply_strict_draft_probs_patch(runner) is False


def test_worker_extension_sets_strict_draft_temperature():
    worker = StrictDraftProbsWorkerExtension()
    worker.model_runner = SimpleNamespace()

    assert worker.set_strict_draft_temperature(0.5) == 0.5
    assert worker.model_runner._waste_sd_strict_draft_temperature == 0.5
    assert worker.set_strict_draft_temperature(None) is None
    assert worker.model_runner._waste_sd_strict_draft_temperature is None

    try:
        worker.set_strict_draft_temperature(-0.1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected invalid draft temperature to raise ValueError")


def test_apply_strict_draft_probs_patch_rejects_unsupported_runner_modes():
    class _FakeDrafter:
        def propose(self, *args, **kwargs):
            return torch.zeros((1, 1), dtype=torch.int64)

    class _FakeRunner:
        def __init__(self, *, disable_padded_drafter_batch: bool, use_async_scheduling: bool):
            self.speculative_config = SimpleNamespace(
                method="draft_model",
                disable_padded_drafter_batch=disable_padded_drafter_batch,
            )
            self.drafter = _FakeDrafter()
            self.use_async_scheduling = use_async_scheduling

        def _sample(self, logits, spec_decode_metadata):
            return "sample"

        def propose_draft_token_ids(self, *args, **kwargs):
            return torch.zeros((1, 1), dtype=torch.int64)

        def _update_states(self, scheduler_output):
            return scheduler_output

    try:
        apply_strict_draft_probs_patch(
            _FakeRunner(disable_padded_drafter_batch=True, use_async_scheduling=False)
        )
    except RuntimeError as exc:
        assert "padded drafter batches" in str(exc)
    else:
        raise AssertionError("Expected padded-drafter guard to raise RuntimeError")

    try:
        apply_strict_draft_probs_patch(
            _FakeRunner(disable_padded_drafter_batch=False, use_async_scheduling=True)
        )
    except RuntimeError as exc:
        assert "synchronous scheduling" in str(exc)
    else:
        raise AssertionError("Expected async guard to raise RuntimeError")
