from optimizer.fingerprint import fingerprint, normalize, system_text, tool_signature


def test_normalize_collapses_whitespace_only():
    assert normalize("  a \n\n b\t c ") == "a b c"
    assert normalize("A b") != normalize("a b")


def test_anthropic_system_string_and_blocks_are_equal():
    s1 = {"model": "claude-opus-5-5", "system": "You are   terse.", "messages": []}
    s2 = {"model": "claude-opus-5-5", "system": [{"type": "text", "text": "You are"}, {"type": "text", "text": "terse."}],
          "messages": []}
    assert fingerprint("anthropic", s1) == fingerprint("anthropic", s2)
    assert fingerprint("anthropic", s1).key.startswith("anthropic:claude-opus-5-5:")


def test_openai_chat_uses_leading_system_and_developer_messages_only():
    body = {"model": "gpt-5", "messages": [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                                          {"role": "user", "content": "hi"}, {"role": "system", "content": "late"}]}
    assert system_text("openai", body) == "A\nB"


def test_openai_responses_uses_instructions_and_leading_developer_items():
    body = {"model": "gpt-5", "instructions": "I", "input": [{"role": "developer", "content": "D"},
                                                              {"role": "user", "content": "u"}]}
    assert system_text("openai", body) == "I\nD"
    assert system_text("openai", {"model": "gpt-5", "input": "plain string"}) == ""


def test_tools_sorted_by_name_and_schema_included():
    a = {"model": "m", "messages": [], "tools": [{"name": "b", "input_schema": {"x": 1}}, {"name": "a", "input_schema": {}}]}
    b = {"model": "m", "messages": [], "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {"x": 1}}]}
    c = {"model": "m", "messages": [], "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {"x": 2}}]}
    assert fingerprint("anthropic", a) == fingerprint("anthropic", b)
    assert fingerprint("anthropic", a) != fingerprint("anthropic", c)
    assert tool_signature("openai", {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}).startswith("f:")


def test_override_and_model_change_routes():
    body = {"model": "claude-opus-5-5", "system": "s", "messages": []}
    assert fingerprint("anthropic", body, override="billing").key == "billing"
    other = dict(body, model="claude-sonnet-5-5")
    assert fingerprint("anthropic", body).key != fingerprint("anthropic", other).key
    assert fingerprint("anthropic", body).system_hash == fingerprint("anthropic", other).system_hash
