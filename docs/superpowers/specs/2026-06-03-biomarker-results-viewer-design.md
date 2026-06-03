# Biomarker Results Viewer — Design

**Date:** 2026-06-03
**Status:** Approved

## Goal

A web viewer that presents the `biomarker_agent`'s per-compound interpretation
outputs as a searchable, nicely-rendered site. Users can search compounds by
DepMap/BRD compound id and by compound (drug) name, and search by top features
of both the refit and baseline models. Each compound has a detail page showing
the full scientific report plus an optional, well-rendered view of the agent's
run trace.

## Context

The agent writes results to `data/interpretation_results/` (gitignored). Layout
per compound directory (e.g. `BRD_BRD-A01907367-001-01-7/`):

- `report.json` — the structured payload:
  - `compound_id` (e.g. `"BRD:BRD-A01907367-001-01-7"`)
  - `meta`: `drug_name`, `moa`, `targets`, `n_samples`,
    `performance` (`bootstrap_pred_pearson`, `baseline_pred_pearson`,
    `selected_refit_oob_pearson`), `header_figures` (SHAP panels with
    `path` + `caption`), `feature_comparison` (`baseline_model`, `n_refit`,
    `n_baseline_top`, `shared`, `refit_only`, `baseline_only`, `divergence`)
  - `summary` (str), `clear_hypothesis` (bool)
  - `hypotheses[]`: `rank`, `title`, `features[]`, `mechanism`, `novelty`,
    `confidence`, `kind`, `evidence` (object of label→text), `figures[]`
    (`path` + `caption`)
  - `proposed_mechanisms[]`, `proposed_biomarkers[]`, `caveats[]`
- `figures/*.png` — figures referenced by relative path in the JSON.
- `trace.json` — `compound_id`, `model`, `usage`
  (`cost_usd`, `prompt_tokens`, `completion_tokens`, `n_calls`, optionally
  `cached_tokens`), `seed_context` (markdown str), `transcript[]` where each
  entry is either `{event: "assistant_text", text}` or
  `{tool, input, output}`.
- Top-level `interpretation_index.md` lists the compounds.

Some compounds legitimately have **no hypothesis** (`clear_hypothesis: false`,
empty `hypotheses`), e.g. Posaconazole. This is a first-class state to render,
not an error.

The compound id IS the DepMap/Broad compound id (`BRD:BRD-…`); there is no
separate per-cell-line id in the outputs.

## Architecture

A **Vite + React + TypeScript single-page app** in `viewer/`, plus a **Python
build step** that converts the on-disk agent outputs into a static JSON bundle
the SPA fetches at runtime.

```
viewer/
  scripts/build_data.py    # reads data/interpretation_results/ -> public/data/
  index.html
  package.json  vite.config.ts  tsconfig.json  tsconfig.node.json
  src/
    main.tsx               # HashRouter (works on file:// and any static host)
    App.tsx
    routes/IndexPage.tsx
    routes/CompoundPage.tsx
    components/ReportView.tsx
    components/TraceView.tsx
    components/Figure.tsx
    components/SearchBar.tsx
    components/PerfBadges.tsx
    components/RefitBaselineTable.tsx
    lib/search.ts
    lib/types.ts
    styles.css
    test/search.test.ts
    test/CompoundPage.test.tsx
    test/TraceView.test.tsx
  public/data/             # GENERATED (gitignored)
    index.json
    <compound_dir>.json
    <compound_dir>/figures/*.png
```

**Routing:** HashRouter so the built site works from `file://` and from any
static host without server-side rewrite config. `#/` = index, `#/c/<id>` =
compound detail.

## Data flow

1. `python viewer/scripts/build_data.py [--results DIR] [--out DIR]`
   - Default `--results data/interpretation_results`,
     `--out viewer/public/data`.
   - For each subdir containing `report.json`: load report + trace, copy
     `figures/` into `public/data/<dir>/figures/`, write
     `public/data/<dir>.json` (full report + trace), and append a compact entry
     to `index.json`.
   - Skip dirs without `report.json` (log a warning).
   - `index.json` entry per compound:
     `{ id (dir name), compound_id, drug_name, moa, targets,
        has_hypothesis, performance {refit, bootstrap, baseline},
        divergence, top_hypothesis_title,
        refit_features[], baseline_features[], hypothesis_genes[],
        search_genes[] }`
     where `search_genes` is the union of bare gene symbols parsed from refit +
     baseline + hypothesis features (token partitioned on first `_`), and the
     `*_features` arrays keep the full tokens. Each gene/feature is stored once,
     de-duplicated.
2. IndexPage fetches `data/index.json` once; runs client-side fuzzy search.
3. CompoundPage fetches `data/<id>.json`; renders report + trace tabs.

## Search (`lib/search.ts`)

Fuzzy, typo-tolerant search via **Fuse.js** over the `index.json` entries.
Indexed keys (weighted): `drug_name`, `compound_id`, `moa`, `targets`,
`refit_features`, `baseline_features`, `hypothesis_genes`, `search_genes`,
`top_hypothesis_title`. Feature arrays index both the full token (`shRNA_MDM4`)
and the bare gene (`MDM4`).

- Empty query → all compounds (default sort: has-hypothesis first, then by
  refit performance descending).
- A result row shows a small "matched" chip naming the field(s) that matched
  (e.g. "refit feature: MDM4") using Fuse match metadata.
- A "has hypothesis only" toggle filters the list.
- Feature-scope chips (refit / baseline / discussed) optionally restrict which
  feature fields participate; default all on.

## Index page

Search bar at top; below, one card per matching compound:
- Drug name (or compound id if no name) as title; BRD id as monospace subtitle.
- MOA · target(s).
- Performance badges (refit / bootstrap / baseline Pearson r).
- Divergence badge (refit-vs-baseline).
- One-line top hypothesis title, or a muted "No hypothesis proposed" tag.
- Matched-field chip(s) when a query is active.
Empty results dir → friendly empty state.

## Compound page — tabs

Header (always visible): drug name, BRD id, MOA, target(s), performance badges,
divergence badge.

**Report tab:**
- SHAP panels (`meta.header_figures`) side by side.
- Refit-vs-baseline table (shared / refit-only / baseline-only, with divergence
  prose).
- Summary.
- Proposed mechanisms and proposed biomarkers (lists).
- Ranked hypotheses (collapsible, expanded by default for rank 1): title with
  novelty/kind/confidence badges, features, mechanism prose, evidence as a
  definition list (label → text), and figures (click to enlarge via `Figure`).
- Caveats.
- No-hypothesis compounds: render summary + a clear "No hypothesis — see
  summary/caveats for why" panel instead of empty hypothesis sections.

**Trace tab (optional view):**
- Collapsible `seed_context` rendered as markdown.
- Transcript as an ordered timeline:
  - `assistant_text` → markdown block (entries are written as `## Step N …`).
  - tool call → collapsible card: tool name header, pretty-printed `input`
    (JSON), and pretty-printed `output` (JSON, collapsible when large).
- Footer: model, prompt/completion (and cached, if present) tokens, `$cost`.
- Missing/absent trace → "No trace recorded for this compound."

Markdown rendering uses `react-markdown`.

## Styling

Implementation uses the **frontend-design** skill. Clean professional scientific
aesthetic: restrained palette, strong typography hierarchy, generous whitespace,
responsive layout (cards on index, single-column readable detail). Badges for
performance (r values), novelty, confidence, and divergence. Figures rendered at
readable width with click-to-enlarge (lightbox/overlay).

## Error handling

- Build step: skip dirs lacking `report.json` (warn, continue); tolerate missing
  `trace.json` (omit trace from that compound's JSON); skip figure files that
  don't exist on disk (warn).
- SPA: fetch failure for a compound JSON → error panel with a back link;
  missing figure → broken-image-safe `Figure` (hide on error); missing trace →
  trace tab shows the absent-trace message; empty index → empty state.

## Testing

**Python (`tests/test_viewer_build.py`, pytest):**
- A synthetic results fixture (two compounds: one with a hypothesis incl.
  features + figures + trace, one no-hypothesis without trace) →
  `build_data.py` produces `index.json` with both entries; the hypothesis
  compound's `search_genes` includes bare genes parsed from `shRNA_MDM4`-style
  tokens; figures are copied to `public/data/<dir>/figures/`; the no-hypothesis
  compound has `has_hypothesis: false`; a dir without `report.json` is skipped.

**Frontend (Vitest + React Testing Library):**
- `search.ts`: a gene query (`"MDM4"`) returns the compound whose refit features
  include `shRNA_MDM4`; a drug-name query returns the right compound; a typo
  (`"MDM5"`) still fuzzy-matches `MDM4`.
- `CompoundPage`: renders hypothesis title + features for a hypothesis compound;
  renders the no-hypothesis panel for the other.
- `TraceView`: renders an `assistant_text` markdown block and a tool-call card
  with the tool name and input.

## Out of scope (YAGNI)

- No live server / API; the SPA is static and rebuilt from outputs.
- No editing of results from the UI.
- No authentication.
- No cross-compound aggregate dashboards beyond the searchable index.
