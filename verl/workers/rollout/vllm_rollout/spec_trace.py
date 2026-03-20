from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def summarize_speculative_chunks(token_chunks: Iterable[Sequence[int]]) -> dict[str, Any]:
    token_ids: list[int] = []
    spec_accept_lens: list[int] = []

    for chunk in token_chunks:
        chunk_ids = [int(token_id) for token_id in chunk]
        if not chunk_ids:
            continue
        token_ids.extend(chunk_ids)
        spec_accept_lens.append(len(chunk_ids))

    return {
        "token_ids": token_ids,
        "spec_accept_lens": spec_accept_lens,
        "spec_verify_ct": len(spec_accept_lens),
        "spec_accepted_tokens": sum(max(length - 1, 0) for length in spec_accept_lens),
    }


def extract_token_log_probs(token_ids: Sequence[int], token_logprobs: Any) -> list[float] | None:
    if token_logprobs is None:
        return None
    return [token_logprobs[i][token_id].logprob for i, token_id in enumerate(token_ids)]


def concat_optional_arrays(chunks: Iterable[Any]) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for chunk in chunks:
        if chunk is None:
            continue
        array = np.asarray(chunk)
        if array.size == 0:
            continue
        arrays.append(array)

    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]

    try:
        return np.concatenate(arrays, axis=0)
    except ValueError:
        return arrays[-1]
