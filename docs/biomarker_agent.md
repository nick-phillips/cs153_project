# Biomarker Interpretation Agent

Points an LLM agent at a biomarker-discovery output directory and produces ranked,
evidence-backed biological-mechanism hypotheses per compound.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...           # required
# optional: export PAPERCLIP_API_KEY=...  # enables --literature paperclip

# whole batch (dir containing MANIFEST.csv)
uv run biomarker-analyze data/small_test_sample --out results/interpretation

# one compound (writes to <dir>/interpretation by default)
uv run biomarker-analyze data/small_test_sample/BRD_BRD-K25244359-066-03-4
```

Run from the repo root so the default `data/` paths resolve.

Flags: `--model`, `--literature {pubmed,paperclip}`, `--max-tool-calls`,
`--feature-file`, `--response-file`, `--treatment-info`, `--cache-dir`.

## Tools the agent can call

| Tool | Source | What it answers |
|------|--------|-----------------|
| `drug_context` | `primary_screen_treatment_info.csv` | Known MOA / target — is a feature on- or off-mechanism? |
| `internal_association` | `x-all_v4.pkl` + responses | Does the feature track response across cell lines, and which direction? |
| `depmap_dependency` | local `CRISPR_*` columns | Is the gene a selective dependency? What are its co-dependencies? |
| `string_enrichment` | STRING-DB | Do the selected genes interact / share enriched functions? |
| `opentargets_target` | Open Targets | Cancer association, druggability, known drugs. |
| `cbioportal_mutations` | cBioPortal | How often is the gene mutated in patient tumors? |
| `reactome_pathways` | Reactome | Pathway membership / convergence. |
| `literature_search` | PubMed (default) / paperclip | Established vs novel gene–mechanism links. |

All external calls are cached under `--cache-dir` (default `.biomarker_agent_cache`),
so re-runs are cheap and resilient to transient API failures. Each tool degrades
gracefully — a failing API returns an `{"error": ...}` the agent works around.

## Output

Per compound: `report.md` (human) + `report.json` (machine). Batch runs with `--out`
also write `interpretation_index.md`.

## Architecture

A deterministic data/tool layer (`loader`, `datactx`, `tools/`) feeds a single-agent
Anthropic tool-use loop (`agent.py`). The agent gets a grounded seed context
(`context.py`: drug MOA + passing features + pre-computed internal associations),
investigates via tools, then emits a forced structured report (`prompts.REPORT_TOOL`)
rendered by `report.py`. The Anthropic client is dependency-injected, so the whole
pipeline is testable offline with no live API or network.
