"""Tool abstraction: JSON-schema declaration + graceful-failure handler."""

from dataclasses import dataclass
from typing import Callable

import requests

DEFAULT_TIMEOUT = 20


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]

    def run(self, arguments: dict) -> dict:
        try:
            return self.handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - tools must degrade gracefully
            return {"error": f"{type(exc).__name__}: {exc}"}

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def http_get_json(url: str, params: dict | None = None, headers: dict | None = None,
                  timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def http_post_json(url: str, json_body: dict, headers: dict | None = None,
                   timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def http_get_text(url: str, params: dict | None = None, headers: dict | None = None,
                  timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET returning the raw response body as text (e.g. for XML endpoints)."""
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text
