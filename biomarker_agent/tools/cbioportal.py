"""Tool: cBioPortal somatic mutation frequency for a gene in a tumor study.

v1 scope: mutation frequency in a single configurable pan-cancer study. Copy-number
and survival association are deferred to a future iteration.
"""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://www.cbioportal.org/api"
DEFAULT_STUDY = "msk_impact_2017"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"},
        "study_id": {"type": "string", "description": "cBioPortal study id", "default": DEFAULT_STUDY},
    },
    "required": ["gene"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str, study_id: str = DEFAULT_STUDY) -> dict:
        g = cache.get_or_set(
            f"cbio:gene:{gene}",
            lambda: base.http_get_json(f"{API}/genes/{gene}"),
        )
        entrez = g.get("entrezGeneId")
        if not entrez:
            return {"error": f"no Entrez id for {gene}"}
        profile = f"{study_id}_mutations"
        muts = cache.get_or_set(
            f"cbio:mut:{study_id}:{entrez}",
            lambda: base.http_get_json(
                f"{API}/molecular-profiles/{profile}/mutations",
                params={"sampleListId": f"{study_id}_all", "entrezGeneId": entrez,
                        "projection": "SUMMARY"},
            ),
        )
        samples = {m.get("sampleId") for m in (muts if isinstance(muts, list) else [])}
        return {
            "gene": gene,
            "entrez_id": entrez,
            "study_id": study_id,
            "n_mutated_samples": len(samples),
        }

    return Tool(
        name="cbioportal_mutations",
        description=(
            "Query cBioPortal for how often a gene is somatically mutated in patient tumors "
            "(a configurable pan-cancer study). Use as tumor-level evidence that a gene is "
            "cancer-relevant in patients, not just cell lines."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
