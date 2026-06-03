"""Tests for the OpenAI-compatible backend adapter (HTTP mocked).

The adapter presents the Anthropic-style `.messages.create(...)` surface the
agent loop expects, translating to/from the OpenAI chat-completions tool-calling
format used by OpenRouter / DigitalOcean serverless inference.
"""

import json

from biomarker_agent import providers


def test_translate_messages_and_tools():
    system = [{"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "ctx"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "tc1", "name": "drug_context", "input": {"compound_id": "X"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tc1", "content": "{\"moa\": \"y\"}"},
            {"type": "text", "text": "now finish"},
        ]},
    ]
    oai = providers._to_openai_messages(system, messages)
    # system carries a cache_control breakpoint -> content-part array form
    assert oai[0] == {"role": "system", "content": [
        {"type": "text", "text": "be helpful", "cache_control": {"type": "ephemeral"}}]}
    assert oai[1] == {"role": "user", "content": "ctx"}
    # assistant turn carries text + a tool_call with JSON-string arguments
    asst = oai[2]
    assert asst["role"] == "assistant"
    assert asst["content"] == "thinking"
    assert asst["tool_calls"][0]["function"]["name"] == "drug_context"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"compound_id": "X"}
    # tool_result becomes a `tool` message; trailing text becomes a user message
    assert oai[3] == {"role": "tool", "tool_call_id": "tc1", "content": "{\"moa\": \"y\"}"}
    assert oai[4] == {"role": "user", "content": "now finish"}

    tools = [{"name": "drug_context", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    otools = providers._to_openai_tools(tools)
    assert otools[0]["type"] == "function"
    assert otools[0]["function"]["name"] == "drug_context"
    assert otools[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_cache_control_preserved_on_seed_context():
    system = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "SEED", "cache_control": {"type": "ephemeral"}}]}]
    oai = providers._to_openai_messages(system, messages)
    assert oai[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    seed = oai[1]
    assert seed["role"] == "user"
    assert seed["content"][0]["text"] == "SEED"
    assert seed["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_control_stays_string():
    oai = providers._to_openai_messages("plain system", [{"role": "user", "content": "hi"}])
    assert oai[0] == {"role": "system", "content": "plain system"}
    assert oai[1] == {"role": "user", "content": "hi"}


def test_create_returns_anthropic_style_blocks(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["auth"] = headers.get("Authorization")

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "let me check",
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "drug_context",
                                         "arguments": "{\"compound_id\": \"BRD:1\"}"},
                        }],
                    },
                }]}

        return R()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    client = providers.OpenAICompatClient(api_key="sk-test", base_url="https://example/v1")
    resp = client.messages.create(
        model="anthropic/claude-sonnet-4.6", max_tokens=512,
        system="sys", tools=[{"name": "drug_context", "description": "d",
                              "input_schema": {"type": "object", "properties": {}}}],
        messages=[{"role": "user", "content": "go"}],
    )
    # request shaped correctly
    assert captured["url"] == "https://example/v1/chat/completions"
    assert captured["body"]["model"] == "anthropic/claude-sonnet-4.6"
    assert captured["auth"] == "Bearer sk-test"
    # response normalized to Anthropic-style blocks the agent loop understands
    assert resp.stop_reason == "tool_use"
    types = [b.type for b in resp.content]
    assert "text" in types and "tool_use" in types
    tu = next(b for b in resp.content if b.type == "tool_use")
    assert tu.id == "call_1"
    assert tu.name == "drug_context"
    assert tu.input == {"compound_id": "BRD:1"}


def test_tool_choice_translated_to_openai(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"finish_reason": "tool_calls", "message": {
                    "content": None,
                    "tool_calls": [{"id": "c", "type": "function",
                                    "function": {"name": "submit_report", "arguments": "{}"}}]}}]}

        return R()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    client = providers.OpenAICompatClient(api_key="k", base_url="https://x/v1")
    client.messages.create(model="m", max_tokens=10, system="s", tools=[],
                           messages=[{"role": "user", "content": "go"}],
                           tool_choice={"type": "tool", "name": "submit_report"})
    assert captured["body"]["tool_choice"] == {
        "type": "function", "function": {"name": "submit_report"}}


def test_usage_accumulates(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"usage": {"cost": 0.0021, "prompt_tokens": 500, "completion_tokens": 60},
                        "choices": [{"finish_reason": "stop",
                                     "message": {"content": "ok", "tool_calls": None}}]}

        return R()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    client = providers.OpenAICompatClient(api_key="k", base_url="https://x/v1")
    for _ in range(3):
        client.messages.create(model="m", max_tokens=10, system="s", tools=[],
                               messages=[{"role": "user", "content": "hi"}])
    totals = client.usage_totals()
    assert totals["n_calls"] == 3
    assert abs(totals["cost_usd"] - 0.0063) < 1e-9
    assert totals["prompt_tokens"] == 1500
    assert totals["completion_tokens"] == 180


def test_create_handles_plain_text_stop(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"finish_reason": "stop",
                                     "message": {"content": "done", "tool_calls": None}}]}

        return R()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    client = providers.OpenAICompatClient(api_key="k", base_url="https://x/v1")
    resp = client.messages.create(model="m", max_tokens=10, system="s", tools=[],
                                  messages=[{"role": "user", "content": "hi"}])
    assert resp.stop_reason == "end_turn"
    assert [b.type for b in resp.content] == ["text"]
    assert resp.content[0].text == "done"
