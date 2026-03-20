from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from verl.workers.rollout.vllm_rollout.spec_trace import (
    concat_optional_arrays,
    extract_token_log_probs,
    summarize_speculative_chunks,
)


def test_summarize_speculative_chunks_matches_waste_sd_block_semantics():
    traced = summarize_speculative_chunks([[11, 12, 13], [], [21], [31, 32]])

    assert traced["token_ids"] == [11, 12, 13, 21, 31, 32]
    assert traced["spec_accept_lens"] == [3, 1, 2]
    assert traced["spec_verify_ct"] == 3
    assert traced["spec_accepted_tokens"] == 3


def test_extract_token_log_probs_reads_sampled_token_entries():
    token_ids = [7, 8]
    token_logprobs = [
        {7: SimpleNamespace(logprob=-0.1)},
        {8: SimpleNamespace(logprob=-0.2)},
    ]

    assert extract_token_log_probs(token_ids, token_logprobs) == [-0.1, -0.2]


def test_concat_optional_arrays_concatenates_non_empty_chunks():
    merged = concat_optional_arrays(
        [
            np.asarray([[1], [2]]),
            None,
            np.asarray([]),
            np.asarray([[3]]),
        ]
    )

    assert merged is not None
    assert merged.tolist() == [[1], [2], [3]]
