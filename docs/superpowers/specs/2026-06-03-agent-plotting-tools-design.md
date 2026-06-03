# Agent Plotting Tools — Design

**Date:** 2026-06-03
**Status:** Approved (design); pending implementation plan

## Problem

The interpretation agent (`biomarker_agent`) produces text-only reports. We want
it to generate **publication-quality figures** that support its hypotheses and
embed them inline in the final report, so a reader can *see* the evidence
(directionality of a feature–response association, dependency selectivity,
pathway convergence, etc.), not just read it.

## Goal

Add a set of plotting tools the agent can call. Each saves a high-quality PNG to
`<out>/figures/` and returns a relative path + caption. The agent attaches chosen
figures to each hypothesis; `report.md` embeds them inline.

## Quality bar (publication-grade)

All figures share one styling module so they look consistent and professional:

- Headless `Agg` backend; export at **300 DPI**, `bbox_inches="tight"`.
- Clean sans-serif typography; title ~13pt bold, axis labels ~11pt, ticks ~9pt.
- Top/right spines removed; light y-grid (`alpha≈0.3`) behind data.
- Colorblind-safe palette (Okabe–Ito); a single consistent accent for primary
  series, neutral grey for context/secondary.
- `constrained_layout` (no clipped labels); sensible default figure sizes
  (~5×4 in single panel).
- Informative titles + axis labels + in-plot stat annotations (e.g. r, p, n).
- Every figure returns a one-line caption string for the report.

## Architecture

Mirror the existing `tools/base.py` + `Tool` pattern.

```
biomarker_agent/
  plotstyle.py        # apply_style(), PALETTE, save_figure(fig, path) helper
  plots.py            # pure plotting functions: take data -> write PNG -> return path
  tools/
    figures.py        # Tool wrappers binding figures_dir + data sources; return {figure, caption}
```

- **`plotstyle.py`** — `apply_style()` sets matplotlib rcParams (called once at
  import of `plots`); `PALETTE` dict of named colors; `finalize(fig, path)` does
  tight 300-DPI save and `plt.close(fig)`.
- **`plots.py`** — one function per figure. Each takes already-prepared
  data/handles, builds the figure with the shared style, saves to an absolute
  path, returns that path. No knowledge of tools/agent. Pure + unit-testable.
- **`tools/figures.py`** — `make_figure_tools(figures_dir, rel_prefix, data_ctx,
  compound_result, cache)` returns a list of `Tool`s. Each wrapper computes a
  deterministic slug filename, calls the matching `plots.*` function, and returns
  `{"figure": "<rel_prefix>/<slug>.png", "caption": "..."}` or `{"error": ...}`.

## The tools

**From the modeling data (DataContext + CompoundResult):**

1. `plot_feature_response(feature, compound)` — scatter of feature vs the
   compound's response across cell lines, OLS regression line + 95% band, Pearson
   & Spearman r and n annotated. Single-feature directionality/association.
2. `plot_feature_panel(features[], compound)` — correlation heatmap among the
   feature set, plus a dedicated column of each feature's correlation with
   response. Adjudicate combinations, collinearity, and direction together.
3. `plot_dependency_distribution(gene)` — histogram of `CRISPR_<gene>` gene-effect
   across lines, dependency threshold (−0.5) marked, % dependent annotated.
4. `plot_codependency_bar(gene)` — horizontal bar of the gene's top CRISPR
   co-dependencies by |r|, signed coloring.
5. `plot_passing_importance(compound)` — grouped horizontal bar of passing
   features' mean real vs null importance (sorted), per-compound model summary.

**Visualizing external-tool outputs (reuse on-disk cache; no extra API cost):**

6. `plot_string_network(genes[])` — STRING interaction graph among the genes;
   circular layout, edge width ∝ score (matplotlib-only, no networkx dep).
7. `plot_mutation_frequency(genes[])` — bar of cBioPortal mutated-sample counts
   per gene (default study), value labels.
8. `plot_pathway_membership(genes[])` — genes × Reactome-pathways binary
   membership heatmap; reveals pathway convergence across features.

## Data flow

- `build_registry(...)` gains optional `figures_dir`, `figures_rel_prefix`
  (default `"figures"`), and `compound_result`. When `figures_dir` is set it
  appends the figure tools via `make_figure_tools(...)`; when unset (e.g. unit
  tests of data tools) no figure tools are added.
- `run_one(...)` creates `out_dir/figures`, passes it + the loaded
  `CompoundResult` into `build_registry`.
- Figure files are written under `out_dir/figures/<slug>.png`; tools return the
  path **relative to `out_dir`** (`figures/<slug>.png`) so the embedded
  `![caption](figures/<slug>.png)` in `report.md` resolves.

## Report integration

- `prompts.REPORT_TOOL` schema: each hypothesis gains an optional
  `figures: [{"path": str, "caption": str}]` array.
- `report.render_markdown` renders, under each hypothesis (after evidence), each
  attached figure as:
  ```
  ![caption](path)
  *Figure: caption*
  ```
- `report.json` already serializes the full payload, so figure references persist
  there too. `trace.json` continues to record the plot tool calls.
- System prompt gains guidance: after supporting a hypothesis with evidence,
  generate 1–2 figures and attach them; don't fabricate figure paths — only
  attach paths returned by a plot tool.

## Error handling

- Every figure tool wrapped by `Tool.run` → `{"error": ...}` on any exception, so
  a plotting failure never aborts the run; the agent proceeds and may note it.
- Tool-output plots fetch via the shared `DiskCache`; if the external data is
  empty/unavailable the tool returns `{"error": ...}` rather than an empty plot.
- `plt.close(fig)` always (in `finalize`) to avoid figure leaks across many calls.

## Testing

- `plotstyle`: `apply_style()` runs; `finalize` writes a non-empty PNG and closes.
- `plots.py`: each pure function, given the synthetic fixture (or small inline
  data), writes a non-empty PNG file with the `.png` signature. Assert file
  exists and size > a few hundred bytes; assert returned path.
- `tools/figures.py`: each wrapper returns `{"figure": "figures/...png", caption}`
  with the file present; data-driven ones use the synthetic fixture; external ones
  monkeypatch the underlying client/cache fetch (no live network).
- `report`: rendering embeds `![...](figures/...png)` when a hypothesis has
  figures; absent `figures` still renders cleanly (back-compat).
- `prompts`: schema includes the `figures` property on hypotheses.
- Full suite stays green; ruff clean.

## Out of scope (v1)

- Interactive/HTML figures; only static PNG.
- Re-styling or embedding the pre-existing pipeline PNGs (separate concern).
- Multi-compound composite figures.
- networkx dependency (circular layout done with matplotlib directly).
