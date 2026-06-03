"""Simple content-addressed JSON disk cache for external API calls."""

import hashlib
import json
from pathlib import Path
from typing import Callable


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / f"{h}.json"

    def get_or_set(self, key: str, produce: Callable[[], dict]) -> dict:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())
        value = produce()
        p.write_text(json.dumps(value))
        return value
