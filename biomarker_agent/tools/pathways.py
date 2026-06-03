"""Tool: Reactome pathway membership for a gene.

The Reactome search endpoint maps a gene symbol to its entities (it does not
return pathways directly). We resolve the gene's human UniProt accession from
search, then query the mapping endpoint for the pathways that accession
participates in.
"""

import re

from ..cache import DiskCache
from . import base
from .base import Tool

SEARCH = "https://reactome.org/ContentService/search/query"
MAPPING = "https://reactome.org/ContentService/data/mapping/UniProt/{acc}/pathways"
HUMAN = "Homo sapiens"
_TAGS = re.compile(r"<[^>]+>")

INPUT_SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"}},
    "required": ["gene"],
}


def _human_uniprot(search_data: dict) -> str | None:
    """Find the human UniProt accession in a Reactome search response."""
    for group in search_data.get("results", []):
        for e in group.get("entries", []):
            species = e.get("species") or []
            if isinstance(species, str):
                species = [species]
            if (
                e.get("databaseName") == "UniProt"
                and HUMAN in species
                and e.get("referenceIdentifier")
            ):
                return e["referenceIdentifier"]
    return None


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str) -> dict:
        search = cache.get_or_set(
            f"reactome:search:{gene}",
            lambda: base.http_get_json(SEARCH, params={"query": gene, "species": HUMAN}),
        )
        acc = _human_uniprot(search)
        if not acc:
            return {"gene": gene, "n_pathways": 0, "pathways": [],
                    "note": "no human UniProt accession found in Reactome"}
        pathways = cache.get_or_set(
            f"reactome:pathways:{acc}",
            lambda: base.http_get_json(MAPPING.format(acc=acc), params={"species": "9606"}),
        )
        out = [
            {"name": _TAGS.sub("", p.get("displayName", "")), "id": p.get("stId")}
            for p in (pathways if isinstance(pathways, list) else [])
        ]
        return {"gene": gene, "uniprot": acc, "n_pathways": len(out), "pathways": out[:20]}

    return Tool(
        name="reactome_pathways",
        description=(
            "List the Reactome pathways a gene participates in. Use to test whether several "
            "selected genes converge on a common pathway, suggesting a coherent mechanism."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
