"""Tool: STRING-DB protein interactions and functional enrichment for a gene set.

The network is auto-expanded with each gene's top interaction partners (STRING
``add_nodes``) so the tool is informative even for a single gene, and enrichment
is computed over the resulting neighborhood (submitted genes + partners) rather
than only the submitted genes — which surfaces shared biology that a sparse,
heterogeneous input set would otherwise miss.
"""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://string-db.org/api/json"
TARGET_NODES = 12  # expand small inputs up to ~this many nodes
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "genes": {"type": "array", "items": {"type": "string"},
                  "description": "Gene symbols (no class prefix). One gene is fine — the "
                                 "network expands to its top interaction partners."},
        "species": {"type": "integer", "description": "NCBI taxon id", "default": 9606},
    },
    "required": ["genes"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(genes: list, species: int = 9606) -> dict:
        if not genes:
            return {"error": "no genes provided"}
        # STRING expects identifiers separated by a carriage-return; requests
        # percent-encodes it (a literal "%0d" would double-encode to "%250d").
        ids = "\r".join(genes)
        add_nodes = max(0, TARGET_NODES - len(genes))
        key = f"{species}:{','.join(sorted(genes))}:n{add_nodes}"
        net_params = {"identifiers": ids, "species": species,
                      "caller_identity": "biomarker_agent", "add_nodes": add_nodes}
        network = cache.get_or_set(
            f"string:net:{key}",
            lambda: base.http_get_json(f"{API}/network", params=net_params),
        )
        net = network if isinstance(network, list) else []
        interactions = [
            {"a": e.get("preferredName_A"), "b": e.get("preferredName_B"),
             "score": round(float(e.get("score", 0)), 3)}
            for e in net
        ]
        # Neighborhood = submitted genes + everything the expanded network surfaced,
        # dropping unmapped raw Ensembl protein ids (e.g. 'ENSP00000491596').
        def _named(x):
            return bool(x) and not x.startswith("ENSP")
        neighborhood = sorted({g for g in genes}
                              | {e["a"] for e in interactions if _named(e["a"])}
                              | {e["b"] for e in interactions if _named(e["b"])})
        enr_params = {"identifiers": "\r".join(neighborhood), "species": species,
                      "caller_identity": "biomarker_agent"}
        enrichment = cache.get_or_set(
            f"string:enr:{species}:{','.join(neighborhood)}",
            lambda: base.http_get_json(f"{API}/enrichment", params=enr_params),
        )
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
            "submitted_genes": genes,
            "neighborhood": neighborhood,
            "n_interactions": len(interactions),
            "interactions": interactions[:25],
            "enrichment": enr,
        }

    return Tool(
        name="string_enrichment",
        description=(
            "Query STRING-DB for protein-protein interactions and shared functional/GO/pathway "
            "enrichment. Works for a single gene (expands to its interaction neighborhood) or a "
            "set. Use to find whether a gene's partners or a selected set converge on a common "
            "process/complex."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
