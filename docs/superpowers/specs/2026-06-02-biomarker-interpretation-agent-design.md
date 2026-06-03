# Biomarker Interpretation Agent — Design

**Date:** 2026-06-02
**Status:** Approved (design); pending implementation plan

## Problem

The `biomarker-discovery` pipeline produces, per compound, a set of multi-omic
features that a resampled (bootstrap-ensemble + stability/significance) model
selected as predictive of drug response, alongside baseline-model feature
importances and prediction metrics. These outputs are not biologically
interpreted. We want a system that can be pointed at an output directory, run
with a single command, and use an LLM agent + a registry of analysis tools to
surface **"interesting" connections between model outputs and plausible,
possibly novel, biological mechanisms of anti-cancer activity** present in the
resampled model.

## Goal

`analyze <output_dir>` → for each compound, a ranked set of candidate
mechanistic hypotheses, each backed by triangulated evidence (internal data
associations + external biological databases + literature) and flagged for
novelty (on-MOA vs off-MOA). Output as human-readable markdown + machine-readable
JSON.

## Inputs

### Model outputs (per compound dir `BRD_<id>/`)
- `summary.json` — `passing_features`, `passing_by_class`, performance metrics
  (`bootstrap_pred_pearson`, `baseline_pred_pearson`, `selected_refit_oob_pearson`).
- `refract/significant/significant_features.csv` — passing features (significant
  + stable) with `feature_class`, `n_replicates`, `reproducibility`,
  `mean_real_importance`, `mean_null_importance`, `p_value`, `q_value`.
- `refract/significant/feature_statistics.csv` — all tested features, ranked.
- `refract/significant/feature_importance.csv` — SHAP on the refit model.
- `baselines/<model>/feature_importance.csv` — top SHAP per baseline model
  (random_forest, elasticnet, lightgbm, xgboost, catboost).
- `MANIFEST.csv` — present at the directory root for a batch of compounds.

### Supporting data (repo `data/`)
- `primary_screen_treatment_info.csv` — BRD id → `Drug.Name`, `MOA`,
  `repurposing_target`.
- `x-all_v4.pkl` — feature matrix, 2132 cell lines (`ModelID`, e.g. `ACH-000001`)
  × 112005 features.
- `responses_primary_v4.pkl` — response matrix, 919 cell lines × 6790 drugs
  (columns are BRD ids).

### Feature naming convention
Every feature is `<CLASS>_<GENE>`, where CLASS ∈ {GE (expression), CRISPR (KO
dependency), PROT (proteomics), shRNA (knockdown), CNA, FUS, MUTDAM, MUTHS, LIN,
GFEAT}. The gene symbol is directly usable for external lookups (STRING, Open
Targets, etc.).

## Architecture

Standalone Python package `biomarker_agent/` with a CLI entry point `analyze`,
driving its own Anthropic API tool-use loop. No Claude Code session required;
runs headless and reproducibly.

```
biomarker_agent/
  cli.py            # arg parsing; batch (MANIFEST.csv) vs single-compound detection; orchestration
  loader.py         # output dir → CompoundResult (passing features+stats, baseline top-SHAP, metrics)
  context.py        # builds grounded opening context (drug MOA + pre-computed internal assoc)
  agent.py          # Anthropic tool-use loop; prompt caching on system block + context
  prompts.py        # system prompt + report schema instructions
  report.py         # structured output → report.md + report.json
  cache.py          # on-disk cache for all external API calls (keyed by tool name + args)
  tools/
    base.py           # Tool protocol: name, json-schema, handler, graceful-error wrapper
    drug_context.py   # MOA/target lookup from treatment_info
    internal_assoc.py # feature↔response correlation + differential activity from pkls
    stringdb.py       # interactions, enrichment, coexpression within passing set
    opentargets.py    # target↔cancer association, tractability, known drugs
    depmap.py         # selective dependency, co-dependencies, lineage selectivity
    cbioportal.py     # tumor alteration frequency + survival association
    pathways.py       # Reactome/KEGG membership (convergence check)
    literature.py     # PubMed E-utilities (default) + paperclip (optional, same interface)
```

### Orchestration strategy
Single-agent tool-use loop per compound (one Claude conversation). Borrows one
idea from a deterministic gather→synthesize approach: the cheap **internal**
tools (`drug_context`, `internal_assoc`) are pre-computed for all passing
features and fed into the opening context, so the agent starts grounded and
spends its tool budget on external triangulation that needs reasoning.

## Data flow

1. `cli.py` resolves the target. If it contains `MANIFEST.csv` → batch over each
   listed compound dir; if it is a single `BRD_*` dir → just that one.
2. `loader.py` builds a `CompoundResult`: BRD id, passing features (name, class,
   reproducibility, p/q, real-vs-null importance), baseline top-N SHAP features
   (for contrast), and metrics (refit vs baseline Pearson).
3. `context.py` enriches: drug MOA/target (`drug_context`) and pre-computed
   `internal_assoc` for each passing feature.
4. `agent.py` seeds the conversation with this context + system prompt, then runs
   the tool-use loop. The agent investigates features/gene-sets via tools.
5. The agent must finish by calling `submit_report` (forced structured schema).
6. `report.py` renders `interpretation/report.md` + `report.json` into the
   compound dir (or a `--out` dir).

## Tool registry

Each tool exposes a JSON schema to the LLM, has a deterministic Python handler,
caches external calls to disk, and returns **compact summarized JSON** (never raw
dumps — protects the token budget). A failing external call returns
`{"error": ...}` so one dead API never aborts a run.

- **drug_context** — `Drug.Name`, `MOA`, `repurposing_target` for the BRD id.
  Enables on-MOA (expected) vs off-MOA (candidate-novel) labeling.
- **internal_assoc** — for a feature + this drug: Pearson/Spearman of feature vs
  response across shared cell lines; differential activity (top vs bottom feature
  tertile → response mean diff, Mann-Whitney U); n, direction. Matrices loaded
  once per run and held in memory.
- **stringdb** — protein interactions, functional/GO/pathway enrichment, and
  coexpression among the passing gene set (free API, no key).
- **opentargets** — target↔cancer association score, tractability/druggability,
  known drugs per target (GraphQL).
- **depmap** — whether the gene is a selective dependency, its co-dependencies,
  lineage selectivity.
- **cbioportal** — tumor-level alteration frequency (mutation/CNA/expression) and
  survival association.
- **pathways** — Reactome/KEGG pathway membership; convergence of disparate
  features onto a shared pathway.
- **literature** — gene / gene+cancer / gene+MOA co-mention search. **Default:
  PubMed E-utilities (no key).** Optional paperclip backend behind the same
  interface, enabled when `PAPERCLIP_API_KEY` is set or `--literature paperclip`
  is passed.

## Output

Per compound, a ranked list of candidate mechanisms. Each hypothesis:
- plain-language mechanism statement,
- feature(s) involved (with class + reproducibility/significance),
- evidence per source (internal assoc, STRING, Open Targets, DepMap, cBioPortal,
  pathways, literature),
- novelty flag (on-MOA / off-MOA / unknown),
- confidence score with brief rationale.

`report.md` (human) and `report.json` (machine) written side by side. A batch run
also writes a top-level `interpretation_index.md` linking each compound's report.

## Error handling

- Per-tool graceful failure (`{"error": ...}`); the agent is instructed it may
  proceed with partial evidence and must note gaps.
- Missing `ANTHROPIC_API_KEY` → clear startup error.
- Missing optional keys (paperclip) → that backend silently unavailable; PubMed
  used.
- Disk cache makes reruns cheap and resilient to transient API failures.

## Testing & reproducibility

- Unit tests: `loader` against the real `data/small_test_sample`;
  `internal_assoc` numeric correctness on synthetic matrices; `report` rendering
  from a fixed structured-output fixture.
- Tool clients tested against recorded/mocked HTTP responses — **no live network
  in CI**.
- `--offline` / `--dry-run`: exercises the full tool layer with a mocked LLM on
  one compound (CI smoke test).
- All external calls cached to disk → reruns reproducible; only the LLM is
  nondeterministic. `--model` selectable (default a Claude Sonnet tier for cost);
  prompt caching applied to the system block and seed context.

## Configuration / dependencies

- New deps: `anthropic` (API), `requests` (HTTP). `scipy`/`pandas`/`numpy`
  already present for internal stats.
- Env: `ANTHROPIC_API_KEY` (required), `PAPERCLIP_API_KEY` (optional).
- CLI flags: `--compound <id>`, `--out <dir>`, `--model <id>`,
  `--literature {pubmed,paperclip}`, `--offline`, `--max-tool-calls <n>`.

## Out of scope (v1)

- No re-running or modifying the ML pipeline.
- No new wet-lab/experimental design output.
- No web UI; CLI + files only.
- Multi-agent / per-feature sub-agents (single-agent loop is sufficient at ~6
  passing features/compound).
