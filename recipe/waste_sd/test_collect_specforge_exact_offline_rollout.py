from __future__ import annotations

import recipe.waste_sd.collect_specforge_exact_offline_rollout as module


def test_normalize_conversation_messages_accepts_sharegpt_format():
    messages = module._normalize_conversation_messages(
        [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]
    )

    assert messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_normalize_conversation_messages_accepts_role_content_format():
    messages = module._normalize_conversation_messages(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_build_next_regeneration_job_uses_regenerated_assistant_history():
    state = module.ConversationState(
        uid="conv-1",
        source_index=0,
        data_source="unit_test",
        original_messages=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1-original"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2-original"},
        ],
    )

    job1 = module._build_next_regeneration_job(state, add_rollout_instruction=False)
    assert job1 is not None
    assert job1["turn_index"] == 0
    assert job1["messages"] == [{"role": "user", "content": "u1"}]

    module._apply_generated_response(state, response_text="a1-regenerated")

    job2 = module._build_next_regeneration_job(state, add_rollout_instruction=False)
    assert job2 is not None
    assert job2["turn_index"] == 1
    assert job2["messages"] == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1-regenerated"},
        {"role": "user", "content": "u2"},
    ]


def test_build_next_regeneration_job_skips_original_assistant_and_preserves_system():
    state = module.ConversationState(
        uid="conv-2",
        source_index=1,
        data_source="unit_test",
        original_messages=[
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "leading-assistant"},
            {"role": "user", "content": "u1"},
        ],
    )

    job = module._build_next_regeneration_job(state, add_rollout_instruction=False)

    assert job is not None
    assert job["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
    ]
