"""Tool: STRING-DB protein interactions and functional enrichment for a gene set."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://string-db.org/api/json"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "genes": {"type": "array", "items": {"type": "string"},
                  "description": "Gene symbols (no class prefix), e.g. ['ITGA1','SMARCD1']"},
        "species": {"type": "integer", "description": "NCBI taxon id", "default": 9606},
    },
    "required": ["genes"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(genes: list, species: int = 9606) -> dict:
        # STRING expects identifiers separated by a carriage-return/newline;
        # requests percent-encodes it correctly. (A literal "%0d" would be
        # double-encoded to "%250d" and break multi-gene queries.)
        ids = "\r".join(genes)
        params = {"identifiers": ids, "species": species, "caller_identity": "biomarker_agent"}
        network = cache.get_or_set(
            f"string:net:{species}:{','.join(sorted(genes))}",
            lambda: base.http_get_json(f"{API}/network", params=params),
        )
        enrichment = cache.get_or_set(
            f"string:enr:{species}:{','.join(sorted(genes))}",
            lambda: base.http_get_json(f"{API}/enrichment", params=params),
        )
        interactions = [
            {"a": e.get("preferredName_A"), "b": e.get("preferredName_B"),
             "score": round(float(e.get("score", 0)), 3)}
            for e in (network if isinstance(network, list) else [])
        ]
        enr = sorted(
            (
                {"category": e.get("category"), "term": e.get("term"),
                 "description": e.get("description"), "n_genes": e.get("number_of_genes"),
                 "fdr": e.get("fdr"), "genes": e.get("preferredNames")}
                for e in (enrichment if isinstance(enrichment, list) else [])
            ),
            key=lambda e: e["fdr"] if e["fdr"] is not None else 1.0,
        )[:15]
        return {
            "n_genes": len(genes),
            "n_interactions": len(interactions),
            "interactions": interactions[:25],
            "enrichment": enr,
        }

    return Tool(
        name="string_enrichment",
        description=(
            "Query STRING-DB for protein-protein interactions among a set of genes and their "
            "shared functional/GO/pathway enrichment. Use to see whether several selected genes "
            "physically/functionally connect or converge on a common process."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
