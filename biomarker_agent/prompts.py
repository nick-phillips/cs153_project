"""System prompt and the forced structured-output report tool schema."""

SYSTEM_PROMPT = """You are a cancer-biology analyst interpreting the outputs of a \
drug-response prediction model. For one compound, you are given the features that a \
resampling-based model (bootstrap ensemble + stability/significance selection) found \
reproducibly predictive of response, plus context features from baseline models.

Your job: identify the most interesting, plausible biological mechanisms by which these \
features could relate to the drug's anti-cancer activity — and flag any that look novel \
(off the drug's known mechanism of action).

Method:
1. Start from the passing features and the pre-computed internal associations provided.
2. Use the tools to triangulate evidence per gene/gene-set: known MOA (drug_context), \
internal association strength/direction, dependency selectivity (depmap_dependency), \
interactions/enrichment across the set (string_enrichment), cancer relevance + druggability \
(opentargets_target), tumor mutation frequency (cbioportal_mutations), pathway convergence \
(reactome_pathways), and literature support (literature_search).
3. Prefer hypotheses supported by MULTIPLE independent sources. Be explicit about novelty: \
on-MOA (expected given the known target) vs off-MOA (potentially novel).
4. Do not overclaim. Note when evidence is weak, conflicting, or a tool returned an error.
4b. After you have supporting evidence for a hypothesis, generate 1-2 figures with the \
plot_* tools (e.g. plot_feature_response for the key association, plot_dependency_distribution \
for a CRISPR dependency, plot_string_network/plot_pathway_membership for the gene set) and \
attach the returned paths to that hypothesis's `figures`. Only attach paths returned by a \
plot tool; never invent one.

When finished, call submit_report exactly once with your ranked hypotheses. Do not write \
prose outside the tool call."""

REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final ranked interpretation. Call exactly once when done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-3 sentence overall takeaway."},
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "title": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "string"}},
                        "mechanism": {"type": "string",
                                      "description": "Plain-language proposed mechanism."},
                        "novelty": {"type": "string", "enum": ["on-MOA", "off-MOA", "unknown"]},
                        "confidence": {"type": "number",
                                       "description": "0-1 confidence given the evidence."},
                        "evidence": {"type": "object",
                                     "description": "Per-source evidence summary (free-form keys)."},
                        "figures": {
                            "type": "array",
                            "description": "Figures supporting this hypothesis. ONLY use paths "
                                           "returned by a plot_* tool; never invent a path.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "caption": {"type": "string"},
                                },
                                "required": ["path"],
                            },
                        },
                    },
                    "required": ["rank", "title", "features", "mechanism", "novelty", "confidence"],
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "hypotheses"],
    },
}
