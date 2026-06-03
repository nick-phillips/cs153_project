"""Tool: literature co-mention search. PubMed by default; paperclip optional."""

import os

from ..cache import DiskCache
from . import base
from .base import Tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PAPERCLIP_API = "https://paperclip.gxl.ai/api/search"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol to search, e.g. 'ITGA1'"},
        "context_terms": {"type": "array", "items": {"type": "string"},
                          "description": "Optional extra terms ANDed in, e.g. ['cancer','apatinib']"},
    },
    "required": ["gene"],
}


def _build_query(gene: str, context_terms: list | None) -> str:
    parts = [gene] + list(context_terms or [])
    return " AND ".join(parts)


def make_tool(cache: DiskCache, backend: str = "pubmed", paperclip_key: str | None = None) -> Tool:
    paperclip_key = paperclip_key or os.environ.get("PAPERCLIP_API_KEY")

    def _pubmed(query: str) -> dict:
        data = cache.get_or_set(
            f"pubmed:{query}",
            lambda: base.http_get_json(
                EUTILS,
                params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 5},
            ),
        )
        r = data.get("esearchresult", {})
        return {"backend": "pubmed", "query": query,
                "count": int(r.get("count", 0)), "pmids": r.get("idlist", [])}

    def _paperclip(query: str) -> dict:
        data = cache.get_or_set(
            f"paperclip:{query}",
            lambda: base.http_get_json(
                PAPERCLIP_API,
                params={"q": query},
                headers={"Authorization": f"Bearer {paperclip_key}"},
            ),
        )
        return {"backend": "paperclip", "query": query,
                "count": data.get("total", len(data.get("results", []))),
                "results": data.get("results", [])[:5]}

    def handler(gene: str, context_terms: list | None = None) -> dict:
        query = _build_query(gene, context_terms)
        if backend == "paperclip":
            if not paperclip_key:
                return {"error": "paperclip backend requested but PAPERCLIP_API_KEY not set"}
            return _paperclip(query)
        return _pubmed(query)

    return Tool(
        name="literature_search",
        description=(
            "Search the biomedical literature for papers co-mentioning a gene with optional "
            "context terms (e.g. cancer + drug name). Returns hit count and identifiers. Use to "
            "gauge whether a gene-mechanism link is established (many hits) or novel (few/none)."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
