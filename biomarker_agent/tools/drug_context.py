"""Tool: look up a compound's known MOA / target from treatment info."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "compound_id": {"type": "string", "description": "BRD id, e.g. 'BRD:BRD-K25244359-066-03-4'"}
    },
    "required": ["compound_id"],
}


@lru_cache(maxsize=4)
def _load(info_path: str) -> pd.DataFrame:
    return pd.read_csv(info_path)


def make_handler(info_path: Path):
    info_path = str(info_path)

    def handler(compound_id: str) -> dict:
        df = _load(info_path)
        hit = df[df["IDs"] == compound_id]
        if hit.empty:
            return {"error": f"{compound_id} not found in treatment info"}
        row = hit.iloc[0]
        return {
            "compound_id": compound_id,
            "drug_name": str(row.get("Drug.Name", "")),
            "moa": str(row.get("MOA", "") or ""),
            "targets": str(row.get("repurposing_target", "") or ""),
        }

    return handler


def make_tool(info_path: Path) -> Tool:
    return Tool(
        name="drug_context",
        description=(
            "Look up the known mechanism of action (MOA) and protein target(s) for a "
            "compound by its BRD id. Use this first to decide whether a selected feature "
            "is on-MOA (expected) or off-MOA (potentially novel)."
        ),
        input_schema=INPUT_SCHEMA,
        handler=make_handler(info_path),
    )
