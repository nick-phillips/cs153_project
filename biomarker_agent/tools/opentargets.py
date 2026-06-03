"""Tool: Open Targets cancer association, tractability, and known drugs for a gene."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://api.platform.opentargets.org/api/v4/graphql"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"}},
    "required": ["gene"],
}

_SEARCH = """query ($q:String!){ search(queryString:$q, entityNames:["target"]){
  hits{ id name } } }"""

_TARGET = """query ($id:String!){ target(ensemblId:$id){
  approvedSymbol
  tractability{ modality label value }
  associatedDiseases(page:{index:0,size:25}){
    rows{ score disease{ name therapeuticAreas{ name } } } } } }"""


def _is_cancer(row: dict) -> bool:
    areas = " ".join(a.get("name", "") for a in row.get("disease", {}).get("therapeuticAreas", []))
    name = row.get("disease", {}).get("name", "")
    blob = f"{areas} {name}".lower()
    return any(k in blob for k in ("neoplasm", "cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma"))


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str) -> dict:
        sr = cache.get_or_set(
            f"ot:search:{gene}",
            lambda: base.http_post_json(API, {"query": _SEARCH, "variables": {"q": gene}}),
        )
        hits = sr.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return {"error": f"no Open Targets hit for {gene}"}
        ens = hits[0]["id"]
        tr = cache.get_or_set(
            f"ot:target:{ens}",
            lambda: base.http_post_json(API, {"query": _TARGET, "variables": {"id": ens}}),
        )
        tgt = tr.get("data", {}).get("target") or {}
        rows = tgt.get("associatedDiseases", {}).get("rows", [])
        cancer = [
            {"disease": r["disease"]["name"], "score": round(float(r["score"]), 3)}
            for r in rows if _is_cancer(r)
        ][:10]
        return {
            "gene": gene,
            "ensembl_id": ens,
            "tractability": [
                {"modality": t.get("modality"), "label": t.get("label")}
                for t in (tgt.get("tractability") or []) if t.get("value")
            ],
            "cancer_associations": cancer,
            "max_cancer_score": max((c["score"] for c in cancer), default=0.0),
        }

    return Tool(
        name="opentargets_target",
        description=(
            "Query Open Targets for a gene's association with cancers, its druggability "
            "(tractability modalities), and whether it is an established drug target. Use to "
            "judge plausibility and novelty of a feature as an anti-cancer target."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
