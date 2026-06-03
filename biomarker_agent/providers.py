"""OpenAI-compatible chat backend (OpenRouter / DigitalOcean serverless).

Presents the same Anthropic-style ``client.messages.create(...)`` surface the
agent loop uses, translating to/from the OpenAI chat-completions tool-calling
format. This lets `run_agent` stay provider-agnostic: the only Anthropic-only
feature it sets — ``cache_control`` on the system block — is simply dropped here
(OpenAI-style endpoints don't support it).
"""

import json

import requests

CHAT_TIMEOUT = 120
_FINISH_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "stop",
}


class _Block:
    """Minimal stand-in for an Anthropic content block (attribute access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Response:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def _system_text(system) -> str:
    if isinstance(system, str):
        return system
    parts = [b.get("text", "") for b in (system or []) if isinstance(b, dict)]
    return "\n".join(p for p in parts if p)


def _to_openai_messages(system, messages: list) -> list:
    """Translate the agent's Anthropic-style message history into OpenAI form."""
    out: list = []
    sys_text = _system_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})

    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            texts = []
            for blk in content:
                if blk.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": blk["tool_use_id"],
                                "content": blk.get("content", "")})
                elif blk.get("type") == "text":
                    texts.append(blk.get("text", ""))
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            texts, tool_calls = [], []
            for blk in (content if isinstance(content, list) else []):
                if blk.get("type") == "text":
                    texts.append(blk.get("text", ""))
                elif blk.get("type") == "tool_use":
                    tool_calls.append({
                        "id": blk["id"], "type": "function",
                        "function": {"name": blk["name"],
                                     "arguments": json.dumps(blk.get("input", {}))},
                    })
            msg = {"role": "assistant", "content": ("\n".join(texts) or None)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
    return out


def _to_openai_tools(tools: list) -> list:
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t.get("description", ""),
                      "parameters": t["input_schema"]}}
        for t in tools
    ]


def _to_blocks(message: dict) -> list:
    blocks = []
    text = message.get("content")
    if text:
        blocks.append(_Block(type="text", text=text))
    for tc in (message.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append(_Block(type="tool_use", id=tc.get("id"), name=fn.get("name"), input=args))
    return blocks


class _Messages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, model, max_tokens, system, tools, messages, tool_choice=None):
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _to_openai_messages(system, messages),
            "tools": _to_openai_tools(tools),
        }
        # Translate Anthropic-style tool_choice {"type":"tool","name":X} to the
        # OpenAI form so a forced report works across both backends.
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
            body["tool_choice"] = {"type": "function",
                                   "function": {"name": tool_choice["name"]}}
        data = self._parent._post(body)
        choice = data["choices"][0]
        stop = _FINISH_MAP.get(choice.get("finish_reason"), "end_turn")
        return _Response(_to_blocks(choice["message"]), stop)


class OpenAICompatClient:
    """Drop-in replacement for `anthropic.Anthropic()` over an OpenAI-style API."""

    def __init__(self, api_key: str, base_url: str, extra_headers: dict | None = None,
                 timeout: int = CHAT_TIMEOUT):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.messages = _Messages(self)

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json", **self.extra_headers}
        resp = requests.post(f"{self.base_url}/chat/completions", json=body,
                             headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
