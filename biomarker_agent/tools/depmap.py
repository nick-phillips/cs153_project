"""Tool: DepMap-style dependency analysis computed from the local CRISPR features."""

from ..datactx import DataContext
from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol, e.g. 'SMARCD1' (no class prefix)"},
        "top_codeps": {"type": "integer", "description": "How many co-dependencies to return", "default": 8},
    },
    "required": ["gene"],
}


def make_tool(ctx: DataContext) -> Tool:
    def handler(gene: str, top_codeps: int = 8) -> dict:
        profile = ctx.dependency_profile(gene)
        if "error" in profile:
            return profile
        return {"profile": profile, "codependencies": ctx.codependencies(gene, top=top_codeps)}

    return Tool(
        name="depmap_dependency",
        description=(
            "Analyze a gene's CRISPR knockout dependency profile across cancer cell lines "
            "(computed from the modeling data): how selectively cells depend on it, and its "
            "top co-dependencies (genes with correlated dependency, hinting at shared pathway). "
            "Use to judge whether a CRISPR/shRNA feature reflects a real, selective vulnerability."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
