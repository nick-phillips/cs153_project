"""Tests for the agent loop using a fake Anthropic client."""

from biomarker_agent.agent import run_agent


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    """Turn 1: call a tool. Turn 2: call submit_report."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _Msg(
                [
                    _Block(type="text", text="checking"),
                    _Block(type="tool_use", id="t1", name="drug_context",
                           input={"compound_id": "BRD:TEST-1"}),
                ],
                "tool_use",
            )
        return _Msg(
            [_Block(type="tool_use", id="t2", name="submit_report",
                    input={"summary": "done", "hypotheses": [
                        {"rank": 1, "title": "H", "features": ["GE_AAA"],
                         "mechanism": "m", "novelty": "off-MOA", "confidence": 0.5}]})],
            "tool_use",
        )


class FakeRegistry:
    def anthropic_schemas(self):
        return [{"name": "drug_context", "description": "d",
                 "input_schema": {"type": "object", "properties": {}}}]

    def dispatch(self, name, arguments):
        return {"drug_name": "DRUG"}


def test_run_agent_returns_report():
    client = FakeClient()
    payload, transcript = run_agent(
        client=client, registry=FakeRegistry(), system_prompt="sys",
        seed_context="ctx", model="fake", max_tool_calls=10,
    )
    assert payload["summary"] == "done"
    assert payload["hypotheses"][0]["features"] == ["GE_AAA"]
    assert client.calls == 2
