"""Anthropic tool-use loop driving one compound's investigation."""

import json

from .prompts import REPORT_TOOL

DEFAULT_MODEL = "claude-sonnet-4-6"


def _content_blocks(message):
    """Normalize message.content into a list of plain dicts for the next turn."""
    out = []
    for b in message.content:
        btype = getattr(b, "type", None)
        if btype == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif btype == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def run_agent(client, registry, system_prompt: str, seed_context: str,
              model: str = DEFAULT_MODEL, max_tool_calls: int = 40):
    """Run the loop until submit_report is called or the call budget is hit.

    Returns (report_payload, transcript). `client` must expose
    `client.messages.create(...)` like the anthropic SDK.
    """
    tools = registry.anthropic_schemas() + [REPORT_TOOL]
    system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": seed_context}]
    transcript = []
    used = 0

    while used < max_tool_calls:
        resp = client.messages.create(
            model=model, max_tokens=4096, system=system, tools=tools, messages=messages,
        )
        blocks = _content_blocks(resp)
        messages.append({"role": "assistant", "content": blocks})
        for b in blocks:
            if b["type"] == "text" and b["text"].strip():
                transcript.append({"event": "assistant_text", "text": b["text"]})
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        if not tool_uses:
            # model stopped without a tool call; nudge once then stop
            transcript.append({"event": "no_tool_use", "stop_reason": resp.stop_reason})
            break

        results = []
        for tu in tool_uses:
            if tu["name"] == "submit_report":
                return tu["input"], transcript
            out = registry.dispatch(tu["name"], tu["input"])
            used += 1
            transcript.append({"tool": tu["name"], "input": tu["input"], "output": out})
            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": json.dumps(out)[:6000],
            })
        messages.append({"role": "user", "content": results})

    # Budget exhausted (or model stopped) without a report: ask once for it.
    # The API requires alternating roles, so if the last turn is already a user
    # message (the final tool_result batch), fold the nudge into it as a text
    # block rather than appending a second consecutive user message.
    nudge = "Tool budget reached. Call submit_report now with your best hypotheses."
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        if isinstance(content, str):
            messages[-1]["content"] = content + "\n\n" + nudge
        else:
            content.append({"type": "text", "text": nudge})
    else:
        messages.append({"role": "user", "content": nudge})
    # Force the model to emit the report tool on this final turn, so a budget
    # exhaustion always yields a real report rather than an empty fallback.
    resp = client.messages.create(
        model=model, max_tokens=4096, system=system, tools=tools, messages=messages,
        tool_choice={"type": "tool", "name": "submit_report"},
    )
    for b in _content_blocks(resp):
        if b["type"] == "tool_use" and b["name"] == "submit_report":
            return b["input"], transcript
    return {"summary": "No report produced.", "hypotheses": [],
            "caveats": ["agent did not call submit_report"]}, transcript
