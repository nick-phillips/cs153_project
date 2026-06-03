# Biomarker Results Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a searchable React + Vite web viewer for the `biomarker_agent`'s per-compound interpretation outputs, fed by a Python build step that bundles the on-disk JSON + figures into a static data directory.

**Architecture:** A Python module (`biomarker_agent/viewer_build.py`, with a thin `viewer/scripts/build_data.py` CLI) scans `data/interpretation_results/`, emits a compact searchable `index.json` plus one full `<id>.json` per compound, and copies figures into `viewer/public/data/`. A Vite + React + TypeScript SPA (HashRouter) fetches those at runtime: an index page with Fuse.js fuzzy search and a compound page with a Report tab and an Agent-trace tab.

**Tech Stack:** Python 3.11 stdlib (json/shutil/pathlib), pytest; Vite 5, React 18, TypeScript 5, react-router-dom 6, fuse.js 7, react-markdown 9, Vitest 2 + @testing-library/react.

**Spec:** `docs/superpowers/specs/2026-06-03-biomarker-results-viewer-design.md`

**Prerequisites:** Node ≥18.12 and npm present (verified: node v18.16, npm 9.5). Python build uses only the stdlib, so any Python ≥3.9 works; run pytest via the repo's environment (`uv run pytest` or the `.venv`).

---

## File Structure

**Python build step (lives with the agent package, since it shapes the agent's output):**
- Create `biomarker_agent/viewer_build.py` — pure functions: `parse_gene`, `index_entry`, `build`.
- Create `viewer/scripts/build_data.py` — argparse CLI wrapper calling `build`.
- Create `tests/test_viewer_build.py` — pytest coverage of the build.

**SPA (`viewer/`):**
- `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html` — scaffold/config.
- `src/main.tsx` — entry + HashRouter routes.
- `src/lib/types.ts` — shared TypeScript types.
- `src/lib/data.ts` — fetch + URL helpers.
- `src/lib/search.ts` — Fuse.js search.
- `src/components/SearchBar.tsx`, `PerfBadges.tsx`, `Figure.tsx`, `RefitBaselineTable.tsx`, `ReportView.tsx`, `TraceView.tsx`.
- `src/routes/IndexPage.tsx`, `src/routes/CompoundPage.tsx`.
- `src/styles.css` — stylesheet (refined with frontend-design skill in Task 12).
- `src/test/setup.ts`, `src/test/search.test.ts`, `src/test/ReportView.test.tsx`, `src/test/TraceView.test.tsx`.
- `viewer/README.md` — how to build data + run.

**Repo config:**
- Modify `.gitignore` — add `viewer/node_modules/`, `viewer/dist/`, `viewer/public/data/`.

---

## Task 1: Python build — gene parsing & index entry extraction

**Files:**
- Create: `biomarker_agent/viewer_build.py`
- Test: `tests/test_viewer_build.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer_build.py`:

```python
"""Tests for the viewer data build step."""

import json

from biomarker_agent.viewer_build import build, index_entry, parse_gene


def test_parse_gene_strips_class_prefix():
    assert parse_gene("shRNA_MDM4") == "MDM4"
    assert parse_gene("GE_KRT20") == "KRT20"
    assert parse_gene("CRISPR_TP53") == "TP53"
    assert parse_gene("PLAIN") == "PLAIN"


def test_index_entry_extracts_genes_and_flags():
    report = {
        "compound_id": "BRD:1",
        "clear_hypothesis": True,
        "meta": {
            "drug_name": "DrugA",
            "moa": "KINASE INHIBITOR",
            "targets": "ABL1",
            "performance": {
                "selected_refit_oob_pearson": 0.33,
                "bootstrap_pred_pearson": 0.12,
                "baseline_pred_pearson": 0.08,
            },
            "feature_comparison": {
                "baseline_model": "random_forest",
                "n_refit": 2, "n_baseline_top": 10,
                "shared": ["shRNA_MDM4"],
                "refit_only": ["GE_KRT20"],
                "baseline_only": ["CRISPR_TP53"],
                "divergence": "moderate",
            },
        },
        "hypotheses": [{"rank": 1, "title": "MDM4 axis", "features": ["shRNA_MDM4", "GE_EMSY"]}],
    }
    e = index_entry(report, "C1")
    assert e["id"] == "C1"
    assert e["compound_id"] == "BRD:1"
    assert e["drug_name"] == "DrugA"
    assert e["has_hypothesis"] is True
    assert e["top_hypothesis_title"] == "MDM4 axis"
    assert e["divergence"] == "moderate"
    assert e["performance"]["refit"] == 0.33
    # refit features = shared + refit_only; baseline = shared + baseline_only
    assert set(e["refit_features"]) == {"shRNA_MDM4", "GE_KRT20"}
    assert set(e["baseline_features"]) == {"shRNA_MDM4", "CRISPR_TP53"}
    # bare genes parsed from all feature sources, de-duplicated
    assert {"MDM4", "KRT20", "TP53", "EMSY"} <= set(e["search_genes"])
    assert "EMSY" in e["hypothesis_genes"]


def test_index_entry_no_hypothesis():
    report = {"compound_id": "BRD:2", "clear_hypothesis": False,
              "meta": {"drug_name": "DrugB"}, "hypotheses": []}
    e = index_entry(report, "C2")
    assert e["has_hypothesis"] is False
    assert e["top_hypothesis_title"] is None
    assert e["refit_features"] == []
    assert e["search_genes"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viewer_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biomarker_agent.viewer_build'`

- [ ] **Step 3: Write the implementation**

Create `biomarker_agent/viewer_build.py`:

```python
"""Build a static JSON bundle for the results viewer from the agent's outputs.

Reads each compound directory under a results dir (each containing report.json,
optional trace.json, and a figures/ dir) and emits, into an output dir:
  - index.json: a compact, searchable list (one entry per compound)
  - <dir>.json: the full report payload plus the trace, per compound
  - <dir>/figures/*.png: the compound's figures, copied verbatim
The viewer SPA fetches these at runtime.
"""

import json
import shutil
from pathlib import Path


def parse_gene(token: str) -> str:
    """'shRNA_MDM4' -> 'MDM4'. No class prefix -> token unchanged."""
    return token.split("_", 1)[1] if "_" in token else token


def _genes(tokens: list) -> list:
    """Bare gene symbols for a list of feature tokens, order-preserving + de-duped."""
    out: list = []
    for t in tokens:
        g = parse_gene(t)
        if g and g not in out:
            out.append(g)
    return out


def index_entry(report: dict, dir_name: str) -> dict:
    """Compact, searchable index record for one compound's report payload."""
    meta = report.get("meta", {}) or {}
    perf = meta.get("performance", {}) or {}
    cmp = meta.get("feature_comparison") or {}
    shared = cmp.get("shared") or []
    refit_features = sorted(set(shared) | set(cmp.get("refit_only") or []))
    baseline_features = sorted(set(shared) | set(cmp.get("baseline_only") or []))
    hyps = report.get("hypotheses") or []
    hyp_features: list = []
    for h in hyps:
        for f in h.get("features", []) or []:
            if f not in hyp_features:
                hyp_features.append(f)
    return {
        "id": dir_name,
        "compound_id": report.get("compound_id", dir_name),
        "drug_name": meta.get("drug_name") or "",
        "moa": meta.get("moa") or "",
        "targets": meta.get("targets") or "",
        "has_hypothesis": bool(report.get("clear_hypothesis") and hyps),
        "performance": {
            "refit": perf.get("selected_refit_oob_pearson"),
            "bootstrap": perf.get("bootstrap_pred_pearson"),
            "baseline": perf.get("baseline_pred_pearson"),
        },
        "divergence": cmp.get("divergence"),
        "top_hypothesis_title": hyps[0].get("title") if hyps else None,
        "refit_features": refit_features,
        "baseline_features": baseline_features,
        "hypothesis_genes": _genes(hyp_features),
        "search_genes": _genes(refit_features + baseline_features + hyp_features),
    }


def build(results_dir, out_dir) -> dict:
    """Scan results_dir, write the viewer bundle into out_dir. Returns a summary."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list = []
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        report_path = sub / "report.json"
        if not report_path.exists():
            print(f"skip {sub.name}: no report.json")
            continue
        report = json.loads(report_path.read_text())
        index.append(index_entry(report, sub.name))

        trace = None
        trace_path = sub / "trace.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
        compound = {**report, "id": sub.name, "trace": trace}
        (out_dir / f"{sub.name}.json").write_text(json.dumps(compound, indent=2))

        fig_src = sub / "figures"
        if fig_src.is_dir():
            fig_dest = out_dir / sub.name / "figures"
            fig_dest.mkdir(parents=True, exist_ok=True)
            for png in sorted(fig_src.glob("*.png")):
                shutil.copy2(png, fig_dest / png.name)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return {"n_compounds": len(index), "out_dir": str(out_dir)}
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_viewer_build.py -v`
Expected: PASS for `test_parse_gene_strips_class_prefix`, `test_index_entry_extracts_genes_and_flags`, `test_index_entry_no_hypothesis` (the `build` import resolves; the build test is added in Task 2).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/viewer_build.py tests/test_viewer_build.py
git commit -m "feat(viewer): index-entry extraction for results bundle"
```

---

## Task 2: Python build — full directory build & CLI

**Files:**
- Modify: `tests/test_viewer_build.py` (add build test)
- Create: `viewer/scripts/build_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viewer_build.py`:

```python
def _write_compound(d, report, trace=None, figures=()):
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(report))
    if trace is not None:
        (d / "trace.json").write_text(json.dumps(trace))
    if figures:
        fd = d / "figures"
        fd.mkdir()
        for name in figures:
            (fd / name).write_bytes(b"\x89PNG\r\n")


def test_build_writes_bundle_and_copies_figures(tmp_path):
    results = tmp_path / "results"
    _write_compound(
        results / "C1",
        {
            "compound_id": "BRD:1", "clear_hypothesis": True,
            "meta": {"drug_name": "DrugA", "feature_comparison": {
                "baseline_model": "random_forest", "n_refit": 1, "n_baseline_top": 10,
                "shared": ["shRNA_MDM4"], "refit_only": [], "baseline_only": [],
                "divergence": "low"}},
            "hypotheses": [{"rank": 1, "title": "T", "features": ["shRNA_MDM4"],
                            "figures": [{"path": "figures/a.png", "caption": "c"}]}],
        },
        trace={"compound_id": "BRD:1", "model": "m", "usage": {"cost_usd": 0.1},
               "seed_context": "s", "transcript": [{"event": "assistant_text", "text": "hi"}]},
        figures=["a.png"],
    )
    _write_compound(
        results / "C2",
        {"compound_id": "BRD:2", "clear_hypothesis": False,
         "meta": {"drug_name": "DrugB"}, "hypotheses": []},
    )
    (results / "junk").mkdir()  # no report.json -> skipped

    out = tmp_path / "out"
    summary = build(results, out)
    assert summary["n_compounds"] == 2

    index = json.loads((out / "index.json").read_text())
    assert {e["id"] for e in index} == {"C1", "C2"}

    c1 = json.loads((out / "C1.json").read_text())
    assert c1["trace"]["model"] == "m"
    assert c1["id"] == "C1"
    assert (out / "C1" / "figures" / "a.png").exists()

    c2 = json.loads((out / "C2.json").read_text())
    assert c2["trace"] is None
    entry2 = next(e for e in index if e["id"] == "C2")
    assert entry2["has_hypothesis"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viewer_build.py::test_build_writes_bundle_and_copies_figures -v`
Expected: FAIL (the test fixture is new). It should actually PASS against the Task 1 implementation of `build` — if it does, that's fine; the purpose here is to lock the behavior. If it fails, fix `build` to match.

- [ ] **Step 3: Write the CLI wrapper**

Create `viewer/scripts/build_data.py`:

```python
#!/usr/bin/env python3
"""CLI: build the viewer data bundle from the agent's interpretation results.

    python viewer/scripts/build_data.py \
        --results data/interpretation_results --out viewer/public/data
"""

import argparse
import sys
from pathlib import Path

# Allow running directly (python viewer/scripts/build_data.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biomarker_agent.viewer_build import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="data/interpretation_results",
                    help="directory of per-compound result dirs")
    ap.add_argument("--out", default="viewer/public/data",
                    help="output directory for the viewer bundle")
    args = ap.parse_args()
    summary = build(args.results, args.out)
    print(f"Wrote {summary['n_compounds']} compounds to {summary['out_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + the CLI against real data**

Run: `uv run pytest tests/test_viewer_build.py -v`
Expected: PASS (all four tests).

Run: `python viewer/scripts/build_data.py`
Expected: prints `Wrote 10 compounds to viewer/public/data` and creates `viewer/public/data/index.json`.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/viewer_build.py tests/test_viewer_build.py viewer/scripts/build_data.py
git commit -m "feat(viewer): full results bundle build + CLI"
```

---

## Task 3: SPA scaffold & gitignore

**Files:**
- Create: `viewer/package.json`, `viewer/vite.config.ts`, `viewer/tsconfig.json`, `viewer/index.html`, `viewer/src/test/setup.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Add gitignore entries**

Append to `.gitignore`:

```
# results viewer
viewer/node_modules/
viewer/dist/
viewer/public/data/
```

- [ ] **Step 2: Create `viewer/package.json`**

```json
{
  "name": "biomarker-results-viewer",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit -p tsconfig.json && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "data": "python scripts/build_data.py --results ../data/interpretation_results --out public/data"
  },
  "dependencies": {
    "fuse.js": "^7.0.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.26.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.6",
    "@testing-library/react": "^16.0.0",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "jsdom": "^24.1.0",
    "typescript": "^5.5.3",
    "vite": "^5.3.4",
    "vitest": "^2.0.4"
  }
}
```

- [ ] **Step 3: Create `viewer/vite.config.ts`**

```ts
/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' keeps asset + data URLs relative so the built site works from
// file://, a subpath, or `python -m http.server` without rewrites.
export default defineConfig({
  base: './',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
});
```

- [ ] **Step 4: Create `viewer/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 5: Create `viewer/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Biomarker Results Viewer</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Create `viewer/src/test/setup.ts`**

```ts
import '@testing-library/jest-dom';
```

- [ ] **Step 7: Install dependencies**

Run: `cd viewer && npm install`
Expected: completes, creates `viewer/node_modules` and `viewer/package-lock.json`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore viewer/package.json viewer/package-lock.json viewer/vite.config.ts viewer/tsconfig.json viewer/index.html viewer/src/test/setup.ts
git commit -m "chore(viewer): scaffold Vite + React + TS app"
```

---

## Task 4: Shared types & data helpers

**Files:**
- Create: `viewer/src/lib/types.ts`, `viewer/src/lib/data.ts`

- [ ] **Step 1: Create `viewer/src/lib/types.ts`**

```ts
export interface Performance {
  refit: number | null;
  bootstrap: number | null;
  baseline: number | null;
}

export interface IndexEntry {
  id: string;
  compound_id: string;
  drug_name: string;
  moa: string;
  targets: string;
  has_hypothesis: boolean;
  performance: Performance;
  divergence: string | null;
  top_hypothesis_title: string | null;
  refit_features: string[];
  baseline_features: string[];
  hypothesis_genes: string[];
  search_genes: string[];
}

export interface FeatureComparison {
  baseline_model: string;
  n_refit: number;
  n_baseline_top: number;
  shared: string[];
  refit_only: string[];
  baseline_only: string[];
  divergence: string;
}

export interface Figure {
  path: string;
  caption?: string;
}

export interface Hypothesis {
  rank: number;
  title: string;
  features: string[];
  mechanism: string;
  novelty?: string;
  confidence?: number;
  kind?: string;
  evidence?: Record<string, string>;
  figures?: Figure[];
}

export interface Meta {
  drug_name?: string;
  moa?: string;
  targets?: string;
  n_samples?: number;
  performance?: Record<string, number>;
  header_figures?: Figure[];
  feature_comparison?: FeatureComparison;
}

export type TraceEntry =
  | { event: 'assistant_text'; text: string }
  | { tool: string; input: unknown; output: unknown };

export interface Trace {
  compound_id: string;
  model: string;
  usage: {
    cost_usd?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    cached_tokens?: number;
    n_calls?: number;
  };
  seed_context: string;
  transcript: TraceEntry[];
}

export interface CompoundData {
  id: string;
  compound_id: string;
  meta: Meta;
  summary: string;
  clear_hypothesis: boolean;
  hypotheses: Hypothesis[];
  proposed_mechanisms: string[];
  proposed_biomarkers: string[];
  caveats: string[];
  trace: Trace | null;
}
```

- [ ] **Step 2: Create `viewer/src/lib/data.ts`**

```ts
import type { CompoundData, IndexEntry } from './types';

// BASE_URL is './' (see vite.config base), so these resolve relative to the
// served index.html — which is constant under HashRouter.
const BASE = import.meta.env.BASE_URL;

export async function loadIndex(): Promise<IndexEntry[]> {
  const res = await fetch(`${BASE}data/index.json`);
  if (!res.ok) throw new Error(`Failed to load index.json (HTTP ${res.status})`);
  return res.json();
}

export async function loadCompound(id: string): Promise<CompoundData> {
  const res = await fetch(`${BASE}data/${id}.json`);
  if (!res.ok) throw new Error(`Failed to load compound "${id}" (HTTP ${res.status})`);
  return res.json();
}

export function figureUrl(id: string, path: string): string {
  return `${BASE}data/${id}/${path}`;
}
```

- [ ] **Step 3: Typecheck**

Run: `cd viewer && npx tsc --noEmit -p tsconfig.json`
Expected: no output (passes). (`import.meta.env` is typed by Vite's client types via `vite`'s ambient declarations; if tsc complains, add `/// <reference types="vite/client" />` at the top of `data.ts`.)

- [ ] **Step 4: Commit**

```bash
git add viewer/src/lib/types.ts viewer/src/lib/data.ts
git commit -m "feat(viewer): shared types + data fetch helpers"
```

---

## Task 5: Fuse.js search

**Files:**
- Create: `viewer/src/lib/search.ts`, `viewer/src/test/search.test.ts`

- [ ] **Step 1: Write the failing test**

Create `viewer/src/test/search.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { makeFuse, search } from '../lib/search';
import type { IndexEntry } from '../lib/types';

const entries: IndexEntry[] = [
  {
    id: 'C1', compound_id: 'BRD:1', drug_name: 'FK-33-824', moa: 'OPIOID', targets: 'OPRM1',
    has_hypothesis: true, performance: { refit: 0.3, bootstrap: 0.1, baseline: 0.08 },
    divergence: 'moderate', top_hypothesis_title: 'MDM4 axis',
    refit_features: ['shRNA_MDM4', 'GE_KRT20'], baseline_features: ['CRISPR_TP53'],
    hypothesis_genes: ['MDM4', 'KRT20'], search_genes: ['MDM4', 'KRT20', 'TP53'],
  },
  {
    id: 'C2', compound_id: 'BRD:2', drug_name: 'Posaconazole', moa: 'STEROL', targets: '',
    has_hypothesis: false, performance: { refit: 0.1, bootstrap: 0, baseline: 0 },
    divergence: 'low', top_hypothesis_title: null,
    refit_features: ['GE_FOO'], baseline_features: [], hypothesis_genes: [], search_genes: ['FOO'],
  },
];

describe('search', () => {
  it('finds a compound by a refit-model gene', () => {
    const r = search(makeFuse(entries), 'MDM4', entries);
    expect(r[0].entry.id).toBe('C1');
    expect(r[0].matchedFields).toContain('search_genes');
  });

  it('finds a compound by drug name', () => {
    const r = search(makeFuse(entries), 'Posaconazole', entries);
    expect(r[0].entry.id).toBe('C2');
  });

  it('fuzzy-matches a gene typo', () => {
    const r = search(makeFuse(entries), 'MDM5', entries);
    expect(r.map((x) => x.entry.id)).toContain('C1');
  });

  it('returns all entries for an empty query', () => {
    expect(search(makeFuse(entries), '', entries).length).toBe(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run src/test/search.test.ts`
Expected: FAIL — cannot resolve `../lib/search`.

- [ ] **Step 3: Create `viewer/src/lib/search.ts`**

```ts
import Fuse from 'fuse.js';
import type { IndexEntry } from './types';

export interface SearchResult {
  entry: IndexEntry;
  matchedFields: string[];
}

const KEYS = [
  { name: 'drug_name', weight: 3 },
  { name: 'compound_id', weight: 3 },
  { name: 'moa', weight: 1 },
  { name: 'targets', weight: 1 },
  { name: 'refit_features', weight: 2 },
  { name: 'baseline_features', weight: 2 },
  { name: 'hypothesis_genes', weight: 2 },
  { name: 'search_genes', weight: 2 },
  { name: 'top_hypothesis_title', weight: 1 },
];

export function makeFuse(entries: IndexEntry[]): Fuse<IndexEntry> {
  return new Fuse(entries, {
    keys: KEYS,
    includeMatches: true,
    ignoreLocation: true,
    threshold: 0.4,
    minMatchCharLength: 2,
  });
}

export function search(
  fuse: Fuse<IndexEntry>,
  query: string,
  entries: IndexEntry[],
): SearchResult[] {
  const q = query.trim();
  if (!q) return entries.map((entry) => ({ entry, matchedFields: [] }));
  return fuse.search(q).map((r) => ({
    entry: r.item,
    matchedFields: Array.from(new Set((r.matches ?? []).map((m) => m.key as string))),
  }));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run src/test/search.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add viewer/src/lib/search.ts viewer/src/test/search.test.ts
git commit -m "feat(viewer): Fuse.js fuzzy search over compounds + features"
```

---

## Task 6: Presentational components (SearchBar, PerfBadges, Figure, RefitBaselineTable)

**Files:**
- Create: `viewer/src/components/SearchBar.tsx`, `PerfBadges.tsx`, `Figure.tsx`, `RefitBaselineTable.tsx`

- [ ] **Step 1: Create `viewer/src/components/SearchBar.tsx`**

```tsx
export default function SearchBar({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="searchbar">
      <input
        type="search"
        placeholder="Search compound id, drug name, or gene/feature (e.g. MDM4)…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoFocus
      />
    </div>
  );
}
```

- [ ] **Step 2: Create `viewer/src/components/PerfBadges.tsx`**

```tsx
import type { Performance } from '../lib/types';

function fmt(v: number | null | undefined): string {
  return typeof v === 'number' ? v.toFixed(3) : '—';
}

export default function PerfBadges({ perf }: { perf: Performance }) {
  const items: Array<[string, number | null]> = [
    ['refit', perf.refit ?? null],
    ['bootstrap', perf.bootstrap ?? null],
    ['baseline', perf.baseline ?? null],
  ];
  return (
    <span className="perf-badges">
      {items.map(([label, v]) => (
        <span key={label} className="badge perf" title={`${label} Pearson r`}>
          {label} r={fmt(v)}
        </span>
      ))}
    </span>
  );
}
```

- [ ] **Step 3: Create `viewer/src/components/Figure.tsx`**

```tsx
import { useState } from 'react';

export default function Figure({ src, caption }: { src: string; caption?: string }) {
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState(false);
  if (err) return null;
  return (
    <figure className="figure">
      <img
        src={src}
        alt={caption ?? ''}
        loading="lazy"
        onClick={() => setOpen(true)}
        onError={() => setErr(true)}
      />
      {caption && <figcaption>{caption}</figcaption>}
      {open && (
        <div className="lightbox" onClick={() => setOpen(false)} role="presentation">
          <img src={src} alt={caption ?? ''} />
        </div>
      )}
    </figure>
  );
}
```

- [ ] **Step 4: Create `viewer/src/components/RefitBaselineTable.tsx`**

```tsx
import type { FeatureComparison } from '../lib/types';

const DIV: Record<string, string> = {
  high: 'substantially different',
  moderate: 'partially overlapping',
  low: 'largely consistent',
};

export default function RefitBaselineTable({ cmp }: { cmp: FeatureComparison }) {
  return (
    <section className="refit-baseline">
      <h3>Refit vs baseline top features</h3>
      <p>
        The resampled refit model selected {cmp.n_refit} significant feature(s);{' '}
        {cmp.shared.length} overlap with the {cmp.baseline_model} baseline's top{' '}
        {cmp.n_baseline_top}. Top features are{' '}
        <strong>{DIV[cmp.divergence] ?? 'compared'}</strong> between the two models.
      </p>
      <table className="kv">
        <tbody>
          <tr><th>Selected by both</th><td>{cmp.shared.join(', ') || '—'}</td></tr>
          <tr><th>Refit only</th><td>{cmp.refit_only.join(', ') || '—'}</td></tr>
          <tr><th>Baseline only</th><td>{cmp.baseline_only.join(', ') || '—'}</td></tr>
        </tbody>
      </table>
    </section>
  );
}
```

- [ ] **Step 5: Typecheck**

Run: `cd viewer && npx tsc --noEmit -p tsconfig.json`
Expected: no output (passes).

- [ ] **Step 6: Commit**

```bash
git add viewer/src/components/SearchBar.tsx viewer/src/components/PerfBadges.tsx viewer/src/components/Figure.tsx viewer/src/components/RefitBaselineTable.tsx
git commit -m "feat(viewer): presentational components"
```

---

## Task 7: ReportView

**Files:**
- Create: `viewer/src/components/ReportView.tsx`, `viewer/src/test/ReportView.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `viewer/src/test/ReportView.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ReportView from '../components/ReportView';
import type { CompoundData } from '../lib/types';

const withHyp: CompoundData = {
  id: 'C1', compound_id: 'BRD:1',
  meta: { drug_name: 'FK', performance: {}, header_figures: [] },
  summary: 'A summary.', clear_hypothesis: true,
  hypotheses: [{
    rank: 1, title: 'MDM4 axis', features: ['shRNA_MDM4'], mechanism: 'Mechanism text.',
    novelty: 'off-MOA', confidence: 0.42, kind: 'biomarker',
    evidence: { model_performance: 'r=0.3' }, figures: [],
  }],
  proposed_mechanisms: ['a mechanism'], proposed_biomarkers: ['a biomarker'],
  caveats: ['a caveat'], trace: null,
};

const noHyp: CompoundData = {
  ...withHyp, hypotheses: [], clear_hypothesis: false,
  proposed_mechanisms: [], proposed_biomarkers: [],
};

describe('ReportView', () => {
  it('renders hypothesis title and features', () => {
    render(<ReportView data={withHyp} />);
    expect(screen.getByText('MDM4 axis')).toBeInTheDocument();
    expect(screen.getByText(/shRNA_MDM4/)).toBeInTheDocument();
    expect(screen.getByText(/Mechanism text/)).toBeInTheDocument();
  });

  it('renders the no-hypothesis panel when there are no hypotheses', () => {
    render(<ReportView data={noHyp} />);
    expect(screen.getByText(/No hypothesis proposed/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run src/test/ReportView.test.tsx`
Expected: FAIL — cannot resolve `../components/ReportView`.

- [ ] **Step 3: Create `viewer/src/components/ReportView.tsx`**

```tsx
import ReactMarkdown from 'react-markdown';
import type { CompoundData } from '../lib/types';
import { figureUrl } from '../lib/data';
import Figure from './Figure';
import RefitBaselineTable from './RefitBaselineTable';

export default function ReportView({ data }: { data: CompoundData }) {
  const { meta } = data;
  const headerFigs = (meta.header_figures ?? []).filter((f) => f.path);
  const hyps = [...(data.hypotheses ?? [])].sort(
    (a, b) => (a.rank ?? 999) - (b.rank ?? 999),
  );

  return (
    <div className="report">
      {headerFigs.length > 0 && (
        <section>
          <h3>Model feature attributions (SHAP)</h3>
          <div className="shap-row">
            {headerFigs.map((f) => (
              <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
            ))}
          </div>
        </section>
      )}

      {meta.feature_comparison && <RefitBaselineTable cmp={meta.feature_comparison} />}

      <section>
        <h3>Summary</h3>
        <ReactMarkdown>{data.summary || ''}</ReactMarkdown>
      </section>

      <section>
        <h3>Proposed mechanism(s) of anticancer action</h3>
        {data.proposed_mechanisms?.length ? (
          <ul>{data.proposed_mechanisms.map((m, i) => <li key={i}>{m}</li>)}</ul>
        ) : (
          <p className="muted"><em>No clear mechanism hypothesis is supported by the evidence.</em></p>
        )}
      </section>

      <section>
        <h3>Proposed biomarker(s) of response</h3>
        {data.proposed_biomarkers?.length ? (
          <ul>{data.proposed_biomarkers.map((b, i) => <li key={i}>{b}</li>)}</ul>
        ) : (
          <p className="muted"><em>No clear biomarker hypothesis is supported by the evidence.</em></p>
        )}
      </section>

      {hyps.length > 0 ? (
        <section>
          <h3>Supporting evidence</h3>
          {hyps.map((h) => (
            <details key={h.rank} className="hypothesis" open={h.rank === 1}>
              <summary>
                <span className="hyp-rank">{h.rank}.</span> {h.title}
                {h.novelty && <span className="badge novelty">{h.novelty}</span>}
                {h.kind && <span className="badge kind">{h.kind}</span>}
                {typeof h.confidence === 'number' && (
                  <span className="badge conf">conf {h.confidence.toFixed(2)}</span>
                )}
              </summary>
              <p className="features"><strong>Features:</strong> {h.features?.join(', ')}</p>
              <ReactMarkdown>{h.mechanism || ''}</ReactMarkdown>
              {h.evidence && Object.keys(h.evidence).length > 0 && (
                <>
                  <h4>Evidence</h4>
                  <dl className="evidence">
                    {Object.entries(h.evidence).map(([k, v]) => (
                      <div key={k}>
                        <dt>{k}</dt>
                        <dd>{v}</dd>
                      </div>
                    ))}
                  </dl>
                </>
              )}
              {(h.figures ?? []).map((f) => (
                <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
              ))}
            </details>
          ))}
        </section>
      ) : (
        <section className="no-hypothesis">
          <h3>No hypothesis proposed</h3>
          <p>
            The agent did not find a hypothesis supported by the evidence for this
            compound. See the summary above and the caveats below for the reasoning.
          </p>
        </section>
      )}

      {data.caveats?.length > 0 && (
        <section>
          <h3>Caveats</h3>
          <ul>{data.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </section>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run src/test/ReportView.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add viewer/src/components/ReportView.tsx viewer/src/test/ReportView.test.tsx
git commit -m "feat(viewer): report view with hypotheses, evidence, figures"
```

---

## Task 8: TraceView

**Files:**
- Create: `viewer/src/components/TraceView.tsx`, `viewer/src/test/TraceView.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `viewer/src/test/TraceView.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TraceView from '../components/TraceView';
import type { Trace } from '../lib/types';

const trace: Trace = {
  compound_id: 'BRD:1', model: 'sonnet',
  usage: { cost_usd: 0.22, prompt_tokens: 100, completion_tokens: 50, n_calls: 3 },
  seed_context: '## Seed context',
  transcript: [
    { event: 'assistant_text', text: '## Step 1\nReasoning here.' },
    { tool: 'drug_context', input: { compound_id: 'BRD:1' }, output: { drug_name: 'FK' } },
  ],
};

describe('TraceView', () => {
  it('renders assistant text and a tool-call card', () => {
    render(<TraceView trace={trace} />);
    expect(screen.getByText(/Reasoning here/)).toBeInTheDocument();
    expect(screen.getByText('drug_context')).toBeInTheDocument();
  });

  it('renders the absent-trace message when trace is null', () => {
    render(<TraceView trace={null} />);
    expect(screen.getByText(/No trace recorded/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd viewer && npx vitest run src/test/TraceView.test.tsx`
Expected: FAIL — cannot resolve `../components/TraceView`.

- [ ] **Step 3: Create `viewer/src/components/TraceView.tsx`**

```tsx
import ReactMarkdown from 'react-markdown';
import type { Trace, TraceEntry } from '../lib/types';

function isText(e: TraceEntry): e is { event: 'assistant_text'; text: string } {
  return 'event' in e && e.event === 'assistant_text';
}

function ToolCall({ entry }: { entry: { tool: string; input: unknown; output: unknown } }) {
  return (
    <details className="tool-call">
      <summary><span className="tool-name">{entry.tool}</span></summary>
      <div className="tool-io">
        <h5>Input</h5>
        <pre>{JSON.stringify(entry.input, null, 2)}</pre>
        <h5>Output</h5>
        <pre>{JSON.stringify(entry.output, null, 2)}</pre>
      </div>
    </details>
  );
}

export default function TraceView({ trace }: { trace: Trace | null }) {
  if (!trace) return <p className="muted">No trace recorded for this compound.</p>;
  const u = trace.usage ?? {};
  return (
    <div className="trace">
      <div className="trace-footer">
        <span className="badge">{trace.model}</span>
        {typeof u.prompt_tokens === 'number' && (
          <span className="badge">{u.prompt_tokens} prompt tok</span>
        )}
        {typeof u.completion_tokens === 'number' && (
          <span className="badge">{u.completion_tokens} completion tok</span>
        )}
        {typeof u.cached_tokens === 'number' && u.cached_tokens > 0 && (
          <span className="badge">{u.cached_tokens} cached</span>
        )}
        {typeof u.cost_usd === 'number' && (
          <span className="badge cost">${u.cost_usd.toFixed(3)}</span>
        )}
      </div>

      <details className="seed">
        <summary>Seed context</summary>
        <ReactMarkdown>{trace.seed_context || ''}</ReactMarkdown>
      </details>

      <ol className="timeline">
        {trace.transcript.map((e, i) => (
          <li key={i}>
            {isText(e) ? (
              <div className="assistant-text"><ReactMarkdown>{e.text}</ReactMarkdown></div>
            ) : (
              <ToolCall entry={e} />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd viewer && npx vitest run src/test/TraceView.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add viewer/src/components/TraceView.tsx viewer/src/test/TraceView.test.tsx
git commit -m "feat(viewer): agent trace timeline view"
```

---

## Task 9: IndexPage

**Files:**
- Create: `viewer/src/routes/IndexPage.tsx`

- [ ] **Step 1: Create `viewer/src/routes/IndexPage.tsx`**

```tsx
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { loadIndex } from '../lib/data';
import { makeFuse, search } from '../lib/search';
import type { IndexEntry } from '../lib/types';
import SearchBar from '../components/SearchBar';
import PerfBadges from '../components/PerfBadges';

const FIELD_LABEL: Record<string, string> = {
  drug_name: 'drug name',
  compound_id: 'compound id',
  moa: 'MOA',
  targets: 'target',
  refit_features: 'refit feature',
  baseline_features: 'baseline feature',
  hypothesis_genes: 'hypothesis gene',
  search_genes: 'gene',
  top_hypothesis_title: 'hypothesis',
};

export default function IndexPage() {
  const [entries, setEntries] = useState<IndexEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [hypOnly, setHypOnly] = useState(false);

  useEffect(() => {
    loadIndex().then(setEntries).catch((e) => setError(String(e)));
  }, []);

  const fuse = useMemo(() => (entries ? makeFuse(entries) : null), [entries]);

  const results = useMemo(() => {
    if (!entries || !fuse) return [];
    let r = search(fuse, query, entries);
    if (hypOnly) r = r.filter((x) => x.entry.has_hypothesis);
    if (!query.trim()) {
      r = [...r].sort((a, b) => {
        if (a.entry.has_hypothesis !== b.entry.has_hypothesis) {
          return a.entry.has_hypothesis ? -1 : 1;
        }
        return (b.entry.performance.refit ?? -1) - (a.entry.performance.refit ?? -1);
      });
    }
    return r;
  }, [entries, fuse, query, hypOnly]);

  if (error) return <div className="page"><p className="error">{error}</p></div>;
  if (!entries) return <div className="page"><p>Loading…</p></div>;

  return (
    <div className="page">
      <header className="site-header">
        <h1>Biomarker Interpretation Results</h1>
        <p className="muted">{entries.length} compounds</p>
      </header>

      <SearchBar value={query} onChange={setQuery} />
      <label className="toggle">
        <input
          type="checkbox"
          checked={hypOnly}
          onChange={(e) => setHypOnly(e.target.checked)}
        />
        Has hypothesis only
      </label>

      {results.length === 0 ? (
        <p className="muted">No compounds match “{query}”.</p>
      ) : (
        <ul className="compound-list">
          {results.map(({ entry, matchedFields }) => (
            <li key={entry.id} className="compound-card">
              <Link to={`/c/${entry.id}`} className="card-link">
                <div className="card-title">
                  {entry.drug_name || entry.compound_id}
                  {!entry.has_hypothesis && (
                    <span className="badge muted-badge">no hypothesis</span>
                  )}
                  {entry.divergence && (
                    <span className={`badge div-${entry.divergence}`}>
                      {entry.divergence} divergence
                    </span>
                  )}
                </div>
                <div className="card-id">{entry.compound_id}</div>
                <div className="card-moa">
                  {entry.moa}
                  {entry.targets ? ` · ${entry.targets}` : ''}
                </div>
                <PerfBadges perf={entry.performance} />
                {entry.has_hypothesis && entry.top_hypothesis_title && (
                  <div className="card-hyp">{entry.top_hypothesis_title}</div>
                )}
                {query.trim() && matchedFields.length > 0 && (
                  <div className="matched">
                    {Array.from(new Set(matchedFields.map((f) => FIELD_LABEL[f] ?? f))).map(
                      (f) => (
                        <span key={f} className="badge match">matched: {f}</span>
                      ),
                    )}
                  </div>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `cd viewer && npx tsc --noEmit -p tsconfig.json`
Expected: no output (passes). (`main.tsx` is created in Task 11; tsc on `src` is fine without it.)

- [ ] **Step 3: Commit**

```bash
git add viewer/src/routes/IndexPage.tsx
git commit -m "feat(viewer): index page with search + compound cards"
```

---

## Task 10: CompoundPage

**Files:**
- Create: `viewer/src/routes/CompoundPage.tsx`

- [ ] **Step 1: Create `viewer/src/routes/CompoundPage.tsx`**

```tsx
import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { loadCompound } from '../lib/data';
import type { CompoundData } from '../lib/types';
import ReportView from '../components/ReportView';
import TraceView from '../components/TraceView';
import PerfBadges from '../components/PerfBadges';

type Tab = 'report' | 'trace';

export default function CompoundPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<CompoundData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('report');

  useEffect(() => {
    if (!id) return;
    setData(null);
    setError(null);
    loadCompound(id).then(setData).catch((e) => setError(String(e)));
  }, [id]);

  if (error) {
    return (
      <div className="page">
        <Link to="/" className="back">← All compounds</Link>
        <p className="error">{error}</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page">
        <Link to="/" className="back">← All compounds</Link>
        <p>Loading…</p>
      </div>
    );
  }

  const { meta } = data;
  return (
    <div className="page compound-page">
      <Link to="/" className="back">← All compounds</Link>
      <header className="compound-header">
        <h1>{meta.drug_name || data.compound_id}</h1>
        <div className="card-id">{data.compound_id}</div>
        <div className="card-moa">
          {meta.moa}
          {meta.targets ? ` · target: ${meta.targets}` : ''}
          {typeof meta.n_samples === 'number' ? ` · n=${meta.n_samples} cell lines` : ''}
        </div>
        <PerfBadges
          perf={{
            refit: meta.performance?.selected_refit_oob_pearson ?? null,
            bootstrap: meta.performance?.bootstrap_pred_pearson ?? null,
            baseline: meta.performance?.baseline_pred_pearson ?? null,
          }}
        />
      </header>

      <nav className="tabs">
        <button className={tab === 'report' ? 'active' : ''} onClick={() => setTab('report')}>
          Report
        </button>
        <button className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
          Agent trace
        </button>
      </nav>

      {tab === 'report' ? <ReportView data={data} /> : <TraceView trace={data.trace} />}
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `cd viewer && npx tsc --noEmit -p tsconfig.json`
Expected: no output (passes).

- [ ] **Step 3: Commit**

```bash
git add viewer/src/routes/CompoundPage.tsx
git commit -m "feat(viewer): compound page with report + trace tabs"
```

---

## Task 11: App entry, routing, and end-to-end build

**Files:**
- Create: `viewer/src/main.tsx`
- Create: `viewer/src/styles.css` (minimal baseline; refined in Task 12)

- [ ] **Step 1: Create `viewer/src/styles.css` (baseline)**

```css
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  color: #1a1a1a;
  background: #f6f7f9;
}
.page { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
.muted { color: #6b7280; }
.error { color: #b91c1c; }
.badge {
  display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.45rem;
  border-radius: 999px; background: #e5e7eb; color: #374151; margin: 0 0.2rem 0.2rem 0;
}
.compound-list { list-style: none; padding: 0; display: grid; gap: 0.75rem; }
.compound-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; }
.card-link { display: block; padding: 1rem; text-decoration: none; color: inherit; }
.card-id { font-family: ui-monospace, monospace; font-size: 0.8rem; color: #6b7280; }
.tabs button { margin-right: 0.5rem; padding: 0.4rem 0.9rem; cursor: pointer; }
.tabs button.active { font-weight: 600; border-bottom: 2px solid #2563eb; }
.shap-row { display: flex; flex-wrap: wrap; gap: 1rem; }
.figure img { max-width: 100%; cursor: zoom-in; border: 1px solid #eee; border-radius: 6px; }
.lightbox {
  position: fixed; inset: 0; background: rgba(0,0,0,0.8);
  display: flex; align-items: center; justify-content: center; cursor: zoom-out; z-index: 50;
}
.lightbox img { max-width: 92vw; max-height: 92vh; }
.tool-call pre, .tool-io pre {
  background: #0f172a; color: #e2e8f0; padding: 0.75rem; border-radius: 6px;
  overflow: auto; font-size: 0.8rem;
}
.timeline { list-style: none; padding-left: 0; }
.evidence dt { font-weight: 600; }
.evidence dd { margin: 0 0 0.6rem 0; }
table.kv { border-collapse: collapse; width: 100%; }
table.kv th { text-align: left; vertical-align: top; padding: 0.3rem 0.6rem 0.3rem 0; width: 12rem; }
```

- [ ] **Step 2: Create `viewer/src/main.tsx`**

```tsx
import React from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter, Routes, Route } from 'react-router-dom';
import IndexPage from './routes/IndexPage';
import CompoundPage from './routes/CompoundPage';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<IndexPage />} />
        <Route path="/c/:id" element={<CompoundPage />} />
      </Routes>
    </HashRouter>
  </React.StrictMode>,
);
```

- [ ] **Step 3: Build the data bundle (if not already present)**

Run: `python viewer/scripts/build_data.py`
Expected: `Wrote 10 compounds to viewer/public/data`.

- [ ] **Step 4: Run the full test suite**

Run: `cd viewer && npx vitest run`
Expected: PASS — all suites (search, ReportView, TraceView).

- [ ] **Step 5: Production build (typecheck + bundle)**

Run: `cd viewer && npm run build`
Expected: tsc passes, Vite writes `viewer/dist/` with no errors.

- [ ] **Step 6: Manual smoke check (dev server)**

Run: `cd viewer && npm run dev` (then open the printed URL; Ctrl-C when done).
Expected: index lists 10 compounds; searching `MDM4` surfaces FK-33-824 with a "matched: gene" chip; opening it shows the report with SHAP figures; the **Agent trace** tab renders the step timeline and a `$` cost badge; Posaconazole shows the "No hypothesis proposed" panel.

- [ ] **Step 7: Commit**

```bash
git add viewer/src/main.tsx viewer/src/styles.css
git commit -m "feat(viewer): app entry, routing, baseline styles"
```

---

## Task 12: Visual design pass + README

**Files:**
- Modify: `viewer/src/styles.css`
- Create: `viewer/README.md`

- [ ] **Step 1: Refine the visual design**

**REQUIRED SUB-SKILL:** Use the `frontend-design` skill for this step.

Refine `viewer/src/styles.css` (and only class-level markup tweaks in components if needed) toward a clean, professional scientific aesthetic: a restrained palette, clear typographic hierarchy (distinct heading scale, comfortable line-height/measure for the long report prose), card hover/focus affordances, divergence/novelty/confidence badge colors that read as semantic (e.g. divergence high/moderate/low on a warm→cool scale), readable evidence definition lists, and a polished trace timeline with clear step separation. Keep it responsive (cards reflow; detail page single-column and readable on mobile). Do not change component logic or prop interfaces.

- [ ] **Step 2: Re-run tests and build to confirm nothing broke**

Run: `cd viewer && npx vitest run && npm run build`
Expected: tests PASS; build succeeds.

- [ ] **Step 3: Manual visual check**

Run: `cd viewer && npm run dev` (open the URL; Ctrl-C when done).
Expected: the index and a compound page (both a hypothesis compound and Posaconazole) look clean and professional; figures, badges, tabs, and the trace timeline are well-styled.

- [ ] **Step 4: Create `viewer/README.md`**

```markdown
# Biomarker Results Viewer

A static React + Vite site for browsing the `biomarker_agent`'s per-compound
interpretation reports. Search compounds by DepMap/BRD id, drug name, or
gene/feature (refit-selected, baseline-top, or hypothesis-discussed), and view
each compound's full report plus the agent's run trace.

## Build the data bundle

The viewer reads a generated bundle under `public/data/` (gitignored). Generate
it from the agent's outputs:

    python viewer/scripts/build_data.py \
        --results data/interpretation_results \
        --out viewer/public/data

This writes `index.json`, one `<compound>.json` per compound, and copies each
compound's figures.

## Develop

    cd viewer
    npm install
    npm run dev        # dev server with hot reload

## Test

    cd viewer
    npm run test       # Vitest (search + component tests)

## Production build

    cd viewer
    npm run build      # type-check + bundle to viewer/dist/
    npm run preview    # serve the built site locally

Because the app uses HashRouter and a relative base, `viewer/dist/` can be served
from any static host (or opened via `python -m http.server -d dist`).

## Regenerate after a new agent run

Re-run `build_data.py`, then `npm run build` (or just refresh `npm run dev`).
```

- [ ] **Step 5: Commit**

```bash
git add viewer/src/styles.css viewer/README.md viewer/src/components viewer/src/routes
git commit -m "feat(viewer): visual design pass + README"
```

---

## Self-Review

**Spec coverage:**
- Searchable index by compound id + drug name → Tasks 5, 9 (Fuse keys `compound_id`, `drug_name`). ✓
- Search by refit + baseline top features and hypothesis genes → Task 1 (`refit_features`/`baseline_features`/`search_genes` extraction) + Task 5 (Fuse keys). ✓
- React + Vite SPA → Tasks 3–11. ✓
- Per-compound detail with full report → Task 7. ✓
- Optional, well-rendered trace tab → Tasks 8, 10. ✓
- No-hypothesis first-class state → Tasks 1 (flag), 7 (panel), tested. ✓
- Figures embedded + enlargeable → Task 6 (`Figure`), 7. ✓
- Build step copies figures + emits JSON → Task 2, tested. ✓
- Professional styling via frontend-design → Task 12. ✓
- Generated artifacts gitignored → Task 3. ✓
- Error handling (missing report/trace/figures, empty index) → Task 1/2 (skip/None), Task 6 (`Figure` onError), Task 8 (null trace), Task 9 (empty/error states), Task 10 (fetch error). ✓
- Tests (Python build; FE search/report/trace) → Tasks 1, 2, 5, 7, 8. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has an expected result. ✓

**Type consistency:** `IndexEntry`/`Performance`/`CompoundData`/`Trace`/`FeatureComparison`/`Figure`/`Hypothesis` defined in Task 4 are used consistently in search (Task 5), components (6–8), and pages (9–10). `index_entry` output keys (Task 1) match `IndexEntry` (Task 4): `id, compound_id, drug_name, moa, targets, has_hypothesis, performance{refit,bootstrap,baseline}, divergence, top_hypothesis_title, refit_features, baseline_features, hypothesis_genes, search_genes`. `figureUrl(id, path)` signature matches all call sites. ✓
