"""Tool: Reactome pathway membership for a gene."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://reactome.org/ContentService/search/query"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"}},
    "required": ["gene"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str) -> dict:
        data = cache.get_or_set(
            f"reactome:{gene}",
            lambda: base.http_get_json(
                API,
                params={"query": gene, "species": "Homo sapiens", "types": "Pathway"},
            ),
        )
        pathways_out = []
        for group in data.get("results", []):
            if group.get("typeName") != "Pathway":
                continue
            for e in group.get("entries", []):
                if e.get("species") in (None, "Homo sapiens"):
                    pathways_out.append({"name": e.get("name"), "id": e.get("id")})
        return {"gene": gene, "n_pathways": len(pathways_out), "pathways": pathways_out[:20]}

    return Tool(
        name="reactome_pathways",
        description=(
            "List the Reactome pathways a gene participates in. Use to test whether several "
            "selected genes converge on a common pathway, suggesting a coherent mechanism."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
