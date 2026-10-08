from optimizer.usage import StreamUsage, Usage, estimate_tokens, usage_from_body


def test_anthropic_body():
    u = usage_from_body("anthropic", {"stop_reason": "end_turn", "usage": {
        "input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 1}})
    assert u == Usage(10, 20, 5, 1, "end_turn", False)


def test_openai_chat_and_responses_bodies():
    u = usage_from_body("openai", {"choices": [{"finish_reason": "length"}], "usage": {
        "prompt_tokens": 7, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}}})
    assert u == Usage(7, 3, 2, None, "length", False)
    u = usage_from_body("openai", {"object": "response", "status": "incomplete",
                                   "incomplete_details": {"reason": "max_output_tokens"},
                                   "usage": {"input_tokens": 4, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 0}}})
    assert u == Usage(4, 9, 0, None, "max_output_tokens", False)
    assert usage_from_body("openai", {"object": "response", "status": "completed", "usage": {}}).stop_reason == "completed"


def test_missing_usage_is_none_not_error():
    assert usage_from_body("anthropic", {}) == Usage(None, None, None, None, None, False)


def test_anthropic_stream():
    s = StreamUsage("anthropic")
    s.feed(b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":3,"cache_creation_input_tokens":0}}}\n\n')
    s.feed(b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n')
    s.feed(b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n')
    assert s.result() == Usage(11, 6, 3, 0, "end_turn", False)


def test_openai_chat_stream_with_and_without_usage():
    s = StreamUsage("openai")
    s.feed(b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n')
    s.feed(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n')
    assert s.result() == Usage(5, 2, None, None, "stop", False)
    s = StreamUsage("openai")
    s.feed(b'data: {"choices":[{"delta":{"content":"hello world"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    u = s.result()
    assert u.estimated and u.output_tokens == estimate_tokens("hello world") and u.stop_reason == "stop"


def test_openai_responses_stream():
    s = StreamUsage("openai")
    s.feed(b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":2}}}\n\n')
    assert s.result() == Usage(1, 2, None, None, "completed", False)


def test_split_chunks_are_reassembled():
    s = StreamUsage("anthropic")
    line = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}\n\n'
    s.feed(line[:20]); s.feed(line[20:])
    assert s.result().output_tokens == 6


def test_estimate_is_positive():
    assert estimate_tokens("The quick brown fox") >= 3


def test_feed_never_raises_on_garbage():
    garbage_inputs = [
        b'garbage\n',
        b'data: {not json}\n',
        b'data: [1,2]\n',
        b'data: null\n',
        b'data: 42\n',
        b'data: {"a":"\xff\xfe"}\n',
    ]
    for garbage in garbage_inputs:
        s_anthropic = StreamUsage("anthropic")
        s_openai = StreamUsage("openai")
        s_anthropic.feed(garbage)  # must not raise
        s_openai.feed(garbage)  # must not raise

    # Test that good events parse after garbage (with CRLF framing)
    s = StreamUsage("anthropic")
    for garbage in garbage_inputs:
        s.feed(garbage)
    s.feed(b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\r\n\r\n')
    assert s.result().output_tokens == 3 and s.result().stop_reason == "end_turn"


def test_estimate_tokens_caches_load_failure(monkeypatch):
    import sys
    import optimizer.usage as usage
    monkeypatch.setattr(usage, "_enc", None)
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # makes `import tiktoken` raise
    assert estimate_tokens("abcd") == max(1, len("abcd") // 4)
    assert usage._enc is False
    assert estimate_tokens("abcd") == max(1, len("abcd") // 4)
