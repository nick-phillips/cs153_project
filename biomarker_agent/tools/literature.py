"""Tool: literature co-mention search. PubMed by default; paperclip optional."""

import os
import xml.etree.ElementTree as ET

from ..cache import DiskCache
from . import base
from .base import Tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PAPERCLIP_API = "https://paperclip.gxl.ai/api/search"
ABSTRACT_MAXLEN = 1500  # cap each abstract to bound the agent's token budget
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
    ncbi_key = os.environ.get("NCBI_API_KEY")  # optional; raises NCBI rate limit

    def _eutils_params(extra: dict) -> dict:
        params = {"db": "pubmed", "retmode": "json", **extra}
        if ncbi_key:
            params["api_key"] = ncbi_key
        return params

    def _titles(pmids: list) -> list:
        """Fetch titles + year for up to a few PMIDs so the agent can cite findings."""
        if not pmids:
            return []
        ids = ",".join(pmids)
        data = cache.get_or_set(
            f"pubmed:summary:{ids}",
            lambda: base.http_get_json(ESUMMARY, params=_eutils_params({"id": ids})),
        )
        res = data.get("result", {})
        papers = []
        for pid in res.get("uids", []):
            rec = res.get(pid, {})
            year = (rec.get("pubdate") or "").split(" ")[0]
            papers.append({"pmid": pid, "title": rec.get("title", ""), "year": year})
        return papers

    def _parse_abstracts(xml_text: str) -> dict:
        """Map PMID -> abstract text from an EFETCH PubmedArticleSet XML payload."""
        out: dict = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return out
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            # AbstractText may be split into labeled sections (Background/Methods/…)
            # and contain nested markup; itertext() flattens each section.
            sections = ["".join(node.itertext()).strip()
                        for node in art.findall(".//Abstract/AbstractText")]
            abstract = " ".join(s for s in sections if s).strip()
            if len(abstract) > ABSTRACT_MAXLEN:
                abstract = abstract[:ABSTRACT_MAXLEN].rstrip() + "…"
            out[pmid_el.text] = abstract
        return out

    def _abstracts(pmids: list) -> dict:
        """Fetch abstracts for PMIDs via EFETCH (cached). PMID -> abstract text."""
        if not pmids:
            return {}
        ids = ",".join(pmids)
        params = {"db": "pubmed", "id": ids, "rettype": "abstract", "retmode": "xml"}
        if ncbi_key:
            params["api_key"] = ncbi_key
        return cache.get_or_set(
            f"pubmed:abstracts:{ids}",
            lambda: _parse_abstracts(base.http_get_text(EFETCH, params=params)),
        )

    def _pubmed(query: str) -> dict:
        data = cache.get_or_set(
            f"pubmed:{query}",
            lambda: base.http_get_json(
                EUTILS, params=_eutils_params({"term": query, "retmax": 5})),
        )
        r = data.get("esearchresult", {})
        pmids = r.get("idlist", [])
        papers = _titles(pmids)
        abstracts = _abstracts(pmids)
        for p in papers:
            p["abstract"] = abstracts.get(p["pmid"], "")
        return {"backend": "pubmed", "query": query,
                "count": int(r.get("count", 0)), "pmids": pmids,
                "top_papers": papers}

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
            "context terms (e.g. cancer + drug name). Returns hit count, PMIDs, and the top "
            "papers' titles, years, and abstracts. Use the count to gauge whether a link is "
            "established (many hits) vs novel (few/none), and read the abstracts to cite what "
            "specific papers actually report."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
