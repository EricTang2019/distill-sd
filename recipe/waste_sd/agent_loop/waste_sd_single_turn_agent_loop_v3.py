# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_USER_ONLY_PROMPT_RE = re.compile(r"^user\n(?P<user>.*)\nassistant\s*$", re.S)
_SYSTEM_USER_PROMPT_RE = re.compile(
    r"^system\n(?P<system>.*)\nuser\n(?P<user>.*)\nassistant\s*$",
    re.S,
)


def _parse_legacy_prompt_string(raw_prompt: str) -> list[dict[str, Any]] | None:
    """Parse legacy 'user\\n...\\nassistant\\n' text back to chat messages."""
    text = raw_prompt.strip()

    match = _SYSTEM_USER_PROMPT_RE.match(text)
    if match is not None:
        return [
            {"role": "system", "content": match.group("system")},
            {"role": "user", "content": match.group("user")},
        ]

    match = _USER_ONLY_PROMPT_RE.match(text)
    if match is not None:
        return [{"role": "user", "content": match.group("user")}]

    return None


def _normalize_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    """Normalize dataset raw_prompt into chat message list."""
    if isinstance(raw_prompt, dict):
        return [raw_prompt]

    if isinstance(raw_prompt, str):
        parsed = _parse_legacy_prompt_string(raw_prompt)
        if parsed is not None:
            return parsed
        return [{"role": "user", "content": raw_prompt}]

    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)

    if not isinstance(raw_prompt, list):
        raise TypeError(f"Unsupported raw_prompt type: {type(raw_prompt)}")

    if not raw_prompt:
        return []

    # Common case: already a list of chat messages.
    if isinstance(raw_prompt[0], dict):
        return raw_prompt

    # Fallback: treat as plain text sequence.
    return [{"role": "user", "content": str(x)} for x in raw_prompt]


def _trim_accept_lens(accept_lens: list[int], response_len: int) -> list[int]:
    """Trim speculative accept lengths so token coverage never exceeds response_len."""
    if response_len <= 0:
        return []

    trimmed = []
    remaining = response_len
    for value in accept_lens:
        if remaining <= 0:
            break
        step = int(value)
        if step <= 0:
            continue
        step = min(step, remaining)
        trimmed.append(step)
        remaining -= step
    return trimmed


class WasteSDSingleTurnAgentLoopV3(SingleTurnAgentLoop):
    """Alternative single-turn loop for rollout A/B testing."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = _normalize_messages(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        request_sampling_params = dict(sampling_params)
        request_sampling_params["max_new_tokens"] = int(self.response_length)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            rollout_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=request_sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = rollout_output.num_preempted if rollout_output.num_preempted is not None else -1

        response_len = min(len(rollout_output.token_ids), self.response_length)
        response_ids = rollout_output.token_ids[:response_len]
        response_mask = [1] * response_len

        extra_fields: dict[str, Any] = {}
        if rollout_output.spec_accept_lens is not None:
            accept_lens = _trim_accept_lens(list(rollout_output.spec_accept_lens), response_len)
            extra_fields["spec_accept_lens"] = accept_lens
        if rollout_output.weight_version is not None:
            extra_fields["weight_version"] = rollout_output.weight_version
        for key in (
            "spec_verify_ct",
            "spec_accepted_tokens",
            "target_forward_total_calls",
            "target_forward_verify_true_calls",
            "target_forward_verify_false_calls",
            "target_forward_verify_true_called",
            "verify_hook_calls",
        ):
            value = getattr(rollout_output, key, None)
            if value is not None:
                extra_fields[key] = value

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=(
                rollout_output.log_probs[:response_len]
                if rollout_output.log_probs is not None
                else None
            ),
            routed_experts=(
                rollout_output.routed_experts[: len(prompt_ids) + response_len]
                if rollout_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
