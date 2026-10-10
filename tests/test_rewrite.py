import copy

from pith.rewrite import PROFILES, SHAPE_TEXT, RouteState, append_shape, apply_effort, apply_profile, is_system_role_rejection

ANTH = {"model": "claude-opus-5-5", "max_tokens": 1024, "system": "S", "tools": [{"name": "t", "input_schema": {}}],
        "messages": [{"role": "user", "content": "q"}]}
CHAT = {"model": "gpt-5", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
RESP = {"model": "gpt-5", "instructions": "S", "input": "q"}
NEVER = ("system", "tools", "model", "thinking", "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature")


def untouched(before, after):
    return all(before.get(k) == after.get(k) for k in NEVER)


def test_profiles_constant():
    assert PROFILES == ("P0", "P1", "P1b", "P2", "P3", "P4")
    assert SHAPE_TEXT == ("Answer directly. No preamble, restatement, or closing summary. "
                          "Target at most {n} words unless the task genuinely needs more.")


def test_p0_is_identity_and_does_not_mutate_input():
    body = copy.deepcopy(ANTH)
    out = apply_profile("anthropic", body, RouteState("P0"))
    assert out == ANTH and body == ANTH and out is not body


def test_anthropic_effort_down_from_model_default_and_explicit():
    out = apply_profile("anthropic", ANTH, RouteState("P1"))
    assert out["output_config"]["effort"] == "low"  # opus-5-5 default medium -> low
    body = dict(ANTH, model="claude-sonnet-5-5")
    assert apply_profile("anthropic", body, RouteState("P1"))["output_config"]["effort"] == "medium"  # default high
    body = dict(ANTH, output_config={"effort": "low"})
    assert apply_profile("anthropic", body, RouteState("P1")) == body  # already lowest: no-op
    body = dict(ANTH, model="claude-haiku-4-5")
    assert apply_profile("anthropic", body, RouteState("P1")) == body  # no effort on haiku: no-op
    assert untouched(ANTH, out)


def test_anthropic_shape_appends_mid_conversation_system_message():
    out = apply_profile("anthropic", ANTH, RouteState("P2", target_words=30))
    assert out["messages"][-1] == {"role": "system", "content": SHAPE_TEXT.format(n=30)}
    assert out["messages"][:-1] == ANTH["messages"] and untouched(ANTH, out)


def test_anthropic_shape_user_text_fallback_keeps_cache_control_block_first():
    body = dict(ANTH, messages=[{"role": "user", "content": [
        {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]}])
    out = apply_profile("anthropic", body, RouteState("P2", injection_form="user_text", target_words=20))
    blocks = out["messages"][-1]["content"]
    assert blocks[0] == body["messages"][0]["content"][0]
    assert blocks[1] == {"type": "text", "text": SHAPE_TEXT.format(n=20)}
    out = apply_profile("anthropic", ANTH, RouteState("P2", injection_form="user_text", target_words=20))
    assert out["messages"][-1]["content"] == [{"type": "text", "text": "q"}, {"type": "text", "text": SHAPE_TEXT.format(n=20)}]


def test_anthropic_shape_falls_back_to_user_text_when_last_message_is_assistant():
    body = dict(ANTH, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    out = apply_profile("anthropic", body, RouteState("P2"))
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][0]["content"][-1]["text"] == SHAPE_TEXT.format(n=20)


def test_p3_adds_exemplar_and_p4_combines():
    out = apply_profile("anthropic", ANTH, RouteState("P3", target_words=20, exemplar="Yes."))
    assert out["messages"][-1]["content"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    out = apply_profile("anthropic", ANTH, RouteState("P4", target_words=25))
    assert out["output_config"]["effort"] == "low" and out["messages"][-1]["role"] == "system"


def test_openai_chat_profiles():
    assert apply_profile("openai", CHAT, RouteState("P1"))["reasoning_effort"] == "low"
    body = dict(CHAT, reasoning_effort="high")
    assert apply_profile("openai", body, RouteState("P1"))["reasoning_effort"] == "medium"
    assert apply_profile("openai", CHAT, RouteState("P1b"))["verbosity"] == "low"
    out = apply_profile("openai", CHAT, RouteState("P2", target_words=40))
    assert out["messages"][-1] == {"role": "developer", "content": SHAPE_TEXT.format(n=40)}


def test_openai_responses_profiles():
    assert apply_profile("openai", RESP, RouteState("P1"), responses_api=True)["reasoning"] == {"effort": "low"}
    assert apply_profile("openai", RESP, RouteState("P1b"), responses_api=True)["text"] == {"verbosity": "low"}
    out = apply_profile("openai", RESP, RouteState("P2", target_words=20), responses_api=True)
    assert out["input"] == [{"role": "user", "content": "q"}, {"role": "developer", "content": SHAPE_TEXT.format(n=20)}]
    body = dict(RESP, input=[{"role": "user", "content": "q"}])
    out = apply_profile("openai", body, RouteState("P2", target_words=20), responses_api=True)
    assert out["input"][-1]["role"] == "developer" and len(out["input"]) == 2


def test_rejection_detector():
    assert is_system_role_rejection(400, '{"error":{"message":"role \'system\' is not supported on this model"}}')
    assert not is_system_role_rejection(400, "other")
    assert not is_system_role_rejection(500, "role 'system' is not supported")


def test_off_ladder_effort_values_pass_through():
    # OpenAI chat with reasoning_effort: "minimal" under P1 → unchanged
    body = dict(CHAT, reasoning_effort="minimal")
    out = apply_profile("openai", body, RouteState("P1"))
    assert out["reasoning_effort"] == "minimal"

    # OpenAI chat with reasoning_effort: "minimal" under P4 → unchanged effort, developer shape message appended
    out = apply_profile("openai", body, RouteState("P4", target_words=20))
    assert out["reasoning_effort"] == "minimal"
    assert out["messages"][-1] == {"role": "developer", "content": SHAPE_TEXT.format(n=20)}

    # OpenAI responses with reasoning: {"effort": "none"} under P1 → unchanged
    resp_body = dict(RESP, reasoning={"effort": "none"})
    out = apply_profile("openai", resp_body, RouteState("P1"), responses_api=True)
    assert out["reasoning"] == {"effort": "none"}

    # OpenAI chat reasoning_effort: "low" under P1 → stays "low"
    body = dict(CHAT, reasoning_effort="low")
    out = apply_profile("openai", body, RouteState("P1"))
    assert out["reasoning_effort"] == "low"


def test_p3_developer_message_with_exemplar():
    # OpenAI chat P3 with exemplar → developer message content == SHAPE_TEXT + exemplar
    out = apply_profile("openai", CHAT, RouteState("P3", target_words=20, exemplar="Yes."))
    expected = SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    assert out["messages"][-1] == {"role": "developer", "content": expected}


def test_unknown_profile_returns_deep_copy():
    # Unknown profile name returns deep copy, not identity
    body = copy.deepcopy(ANTH)
    out = apply_profile("anthropic", body, RouteState("P9"))
    assert out == ANTH and out is not body


def test_litellm_profiles_are_user_text_shape_only():
    out = apply_profile("litellm", CHAT, RouteState("P2", target_words=30))
    assert out["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": "q"},
                                                               {"type": "text", "text": SHAPE_TEXT.format(n=30)}]}
    assert out["messages"][:-1] == CHAT["messages"][:-1] and untouched(CHAT, out)
    out = apply_profile("litellm", CHAT, RouteState("P2", injection_form="system", target_words=30))
    assert out["messages"][-1]["role"] == "user" and len(out["messages"]) == len(CHAT["messages"])  # form column ignored
    out = apply_profile("litellm", CHAT, RouteState("P3", target_words=20, exemplar="Yes."))
    assert out["messages"][-1]["content"][-1]["text"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    for p in ("P1", "P1b", "P4"):
        assert apply_profile("litellm", CHAT, RouteState(p)) == CHAT


def test_apply_effort_is_the_parameter_half():
    assert apply_effort("anthropic", ANTH, "P1")["output_config"]["effort"] == "low"
    assert apply_effort("anthropic", ANTH, "P2") == ANTH and apply_effort("anthropic", ANTH, "P2") is not ANTH
    out = apply_effort("openai", CHAT, "P4")
    assert out["reasoning_effort"] == "low" and out["messages"] == CHAT["messages"]  # no shape in the parameter half
    assert apply_effort("openai", CHAT, "P1b")["verbosity"] == "low"
    out = apply_effort("openai", RESP, "P1", responses_api=True)
    assert out["reasoning"] == {"effort": "low"} and untouched(RESP, out)
    assert apply_effort("openai", CHAT, "P0") == CHAT


def test_append_shape_targets_the_last_user_message_in_place():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    append_shape(msgs, RouteState("P2", target_words=30))
    assert msgs[0]["content"] == [{"type": "text", "text": "q"}, {"type": "text", "text": SHAPE_TEXT.format(n=30)}]
    assert msgs[1] == {"role": "assistant", "content": "a"}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]}]
    append_shape(msgs, RouteState("P3", target_words=20, exemplar="Yes."))
    assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[0]["content"][1]["text"] == SHAPE_TEXT.format(n=20) + "\n\nExample of the expected length:\nYes."
    empty = []
    append_shape(empty, RouteState("P2"))
    assert empty == []
