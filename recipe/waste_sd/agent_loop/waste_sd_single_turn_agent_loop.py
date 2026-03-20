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
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WasteSDSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop that preserves SD metadata in extra_fields."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        metrics = {}
        # Keep generation length aligned with trainer response_length.
        # The server-side fallback may allow longer generation based on prompt_length,
        # which can cause drift and later truncation artifacts.
        request_sampling_params = dict(sampling_params)
        request_sampling_params["max_new_tokens"] = int(self.response_length)
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

        response_mask = [1] * len(rollout_output.token_ids)

        # actual_response_len may be less than self.response_length if the model generated fewer tokens.
        # spec_accept_lens tracks all tokens emitted by SGLang (up to max_new_tokens which can exceed
        # response_length when prompt is shorter than prompt_length).  Trim it to match the tokens
        # that are actually stored in response_ids so that build_strict_weights alignment holds.
        actual_response_len = min(len(rollout_output.token_ids), self.response_length)

        extra_fields: dict[str, Any] = {}
        if rollout_output.spec_accept_lens is not None:
            accept_lens = rollout_output.spec_accept_lens
            if sum(accept_lens) != actual_response_len:
                trimmed, remaining = [], actual_response_len
                for al in accept_lens:
                    if remaining <= 0:
                        break
                    trimmed.append(min(al, remaining))
                    remaining -= al
                accept_lens = trimmed
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
            response_ids=rollout_output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=rollout_output.log_probs[: self.response_length] if rollout_output.log_probs else None,
            routed_experts=(
                rollout_output.routed_experts[: len(prompt_ids) + self.response_length]
                if rollout_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
