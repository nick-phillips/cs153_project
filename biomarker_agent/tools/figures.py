"""Figure-generating Tool wrappers.

Each tool computes a deterministic filename, calls a pure function in
`biomarker_agent.plots`, and returns {"figure": "<rel_prefix>/<slug>.png",
"caption": ...} or {"error": ...}. Figures are written under `figures_dir`.
"""

import re
from pathlib import Path

from .. import plots
from ..cache import DiskCache
from ..datactx import DataContext
from . import cbioportal, pathways, stringdb
from .base import Tool


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def make_figure_tools(figures_dir, rel_prefix: str, data_ctx: DataContext,
                      compound_result, cache: DiskCache) -> list:
    figures_dir = Path(figures_dir)
    cid = compound_result.compound_id

    def _abs(name: str):
        return figures_dir / name, f"{rel_prefix}/{name}"

    def _feature_response(feature: str) -> dict:
        if feature not in data_ctx.features.columns:
            return {"error": f"feature {feature!r} not in matrix"}
        if cid not in data_ctx.responses.columns:
            return {"error": f"compound {cid!r} not in responses"}
        ap, rel = _abs(f"feature_response__{_slug(feature)}.png")
        plots.feature_response(data_ctx.features[feature], data_ctx.responses[cid],
                               feature, cid, ap)
        return {"figure": rel, "caption": f"{feature} vs {cid} response across cell lines"}

    def _feature_panel(features: list) -> dict:
        cols = [f for f in features if f in data_ctx.features.columns]
        if len(cols) < 2:
            return {"error": "need >=2 known features"}
        sub = data_ctx.features[cols].join(
            data_ctx.responses[cid].rename("response"), how="inner").dropna()
        if len(sub) < 3:
            return {"error": "too few shared samples"}
        ap, rel = _abs(f"feature_panel__{_slug('_'.join(cols))}.png")
        plots.feature_panel(sub.corr(), cid, ap)
        return {"figure": rel, "caption": f"Correlations among {len(cols)} features and response"}

    def _dependency(gene: str) -> dict:
        col = f"CRISPR_{gene}"
        if col not in data_ctx.features.columns:
            return {"error": f"{col} not in matrix"}
        ap, rel = _abs(f"dependency__{_slug(gene)}.png")
        plots.dependency_distribution(data_ctx.features[col], gene, ap)
        return {"figure": rel, "caption": f"{gene} CRISPR dependency distribution"}

    def _codependency(gene: str) -> dict:
        codeps = data_ctx.codependencies(gene)
        if not codeps:
            return {"error": f"no co-dependencies for {gene}"}
        ap, rel = _abs(f"codependency__{_slug(gene)}.png")
        plots.codependency_bar(codeps, gene, ap)
        return {"figure": rel, "caption": f"{gene} top CRISPR co-dependencies"}

    def _passing_importance() -> dict:
        feats = [{"name": f.name, "mean_real_importance": f.mean_real_importance,
                  "mean_null_importance": f.mean_null_importance}
                 for f in compound_result.passing_features]
        if not feats:
            return {"error": "no passing features"}
        ap, rel = _abs("passing_importance.png")
        plots.passing_importance(feats, ap)
        return {"figure": rel, "caption": "Passing features: real vs null importance"}

    def _string_network(genes: list) -> dict:
        data = stringdb.make_tool(cache).run({"genes": genes})
        if "error" in data:
            return data
        ap, rel = _abs(f"string__{_slug('_'.join(genes))}.png")
        plots.string_network(genes, data.get("interactions", []), ap)
        return {"figure": rel, "caption": f"STRING interactions among {len(genes)} genes"}

    def _mutation_frequency(genes: list) -> dict:
        tool = cbioportal.make_tool(cache)
        counts = []
        for g in genes:
            out = tool.run({"gene": g})
            counts.append((g, int(out.get("n_mutated_samples", 0)) if "error" not in out else 0))
        ap, rel = _abs(f"mutations__{_slug('_'.join(genes))}.png")
        plots.mutation_frequency(counts, ap)
        return {"figure": rel, "caption": "cBioPortal mutation frequency per gene"}

    def _pathway_membership(genes: list) -> dict:
        tool = pathways.make_tool(cache)
        mapping = {}
        for g in genes:
            out = tool.run({"gene": g})
            if "error" not in out:
                mapping[g] = [p["name"] for p in out.get("pathways", [])]
        if not any(mapping.values()):
            return {"error": "no pathways found for given genes"}
        ap, rel = _abs(f"pathways__{_slug('_'.join(genes))}.png")
        plots.pathway_membership(mapping, ap)
        return {"figure": rel, "caption": "Reactome pathway membership across genes"}

    feat_schema = {"type": "object", "properties": {
        "feature": {"type": "string", "description": "Full feature name, e.g. 'GE_ITGA1'"}},
        "required": ["feature"]}
    gene_schema = {"type": "object", "properties": {
        "gene": {"type": "string", "description": "Gene symbol (no class prefix)"}},
        "required": ["gene"]}
    genes_schema = {"type": "object", "properties": {
        "genes": {"type": "array", "items": {"type": "string"},
                  "description": "Gene symbols (no class prefix)"}},
        "required": ["genes"]}
    features_schema = {"type": "object", "properties": {
        "features": {"type": "array", "items": {"type": "string"},
                     "description": "Full feature names, e.g. ['GE_ITGA1','PROT_PDLIM5']"}},
        "required": ["features"]}
    empty_schema = {"type": "object", "properties": {}}

    return [
        Tool("plot_feature_response",
             "Scatter of a feature vs this compound's response with regression line and r/p/n. "
             "Use to visualize and adjudicate the direction/strength of a feature's association.",
             feat_schema, _feature_response),
        Tool("plot_feature_panel",
             "Correlation heatmap among several features and the response together. Use to show "
             "how a combination of features relates to response and to each other (collinearity).",
             features_schema, _feature_panel),
        Tool("plot_dependency_distribution",
             "Histogram of a gene's CRISPR knockout effect across cell lines with the dependency "
             "threshold marked. Use to visualize selective dependency.",
             gene_schema, _dependency),
        Tool("plot_codependency_bar",
             "Bar chart of a gene's top CRISPR co-dependencies. Use to show shared-complex/pathway "
             "structure behind a dependency.",
             gene_schema, _codependency),
        Tool("plot_passing_importance",
             "Bar chart of the passing features' real vs null importance for this compound. Use as "
             "an overview of what the resampled model selected.",
             empty_schema, lambda: _passing_importance()),
        Tool("plot_string_network",
             "STRING protein-interaction network among a set of genes. Use to visualize whether "
             "selected genes physically/functionally connect.",
             genes_schema, _string_network),
        Tool("plot_mutation_frequency",
             "Bar of cBioPortal somatic mutation counts per gene. Use to visualize tumor-level "
             "alteration frequency across selected genes.",
             genes_schema, _mutation_frequency),
        Tool("plot_pathway_membership",
             "Heatmap of which Reactome pathways each gene belongs to. Use to visualize pathway "
             "convergence across selected genes.",
             genes_schema, _pathway_membership),
    ]
