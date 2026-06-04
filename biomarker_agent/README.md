# Biomarker Interpretation Agent

An LLM agent that reads the per-compound output of the biomarker-discovery ML
pipeline and answers one question for each drug:

> **What mechanism of action does the model's output actually support, and how
> strong is the evidence for it?**

The pipeline trains a drug-response regression model per compound and selects a
small set of reproducibly-predictive multi-omic features (gene expression,
CRISPR/shRNA dependency, proteomics, copy-number, …). This agent interprets that
feature set like a scientist — synthesising a single best hypothesis (a mechanism
and/or a biomarker of response), gathering corroborating evidence from public
bioinformatics resources and the literature, and producing an auditable report.

A companion **web viewer** (`../viewer/`) makes the reports searchable and
browsable.

---

## How the agent works (implementation)

The agent is a **single-agent, tool-use loop** (`agent.py: run_agent`) over an
Anthropic-style `messages.create(...)` interface. One compound = one loop.

**1. Grounded seed context, computed up front.** Before the model sees anything,
`loader.py` parses the compound directory into a typed `CompoundResult`, and
`context.precompute_internal` runs the feature↔response association
(`datactx.associate`) for *every* passing feature. `context.build_seed_context`
then renders a compact, deterministic prompt: the drug's MOA/targets, the model
performance line (framed as context, not a gate), the response-direction
convention, and the passing features **ranked by `mean_real_importance`** with
their precomputed associations (sign, effect size, q-value) and the
refit-vs-baseline comparison. Because the associations are precomputed, the agent
never spends tool calls rediscovering them.

**2. The loop.** The tool set is the evidence/figure registry plus the forced
`submit_report` schema. Each turn:

- the model emits reasoning text and/or `tool_use` blocks;
- if one of them is `submit_report`, its validated input **is** the report
  payload — the loop returns immediately;
- otherwise each tool call is dispatched through the registry, its result is
  appended as a `tool_result` (truncated to keep context bounded), and the loop
  continues. A per-compound counter enforces `--max-tool-calls`.

The system prompt and the (large, static) seed context are both marked with
`cache_control`, so the whole prefix is **prompt-cached** and re-read cheaply on
every subsequent turn — most of the per-compound cost is cached input.

**3. Forced, structured output.** The report is never free text. `submit_report`
(`prompts.REPORT_TOOL`) is a JSON-schema tool whose required fields include
`hypothesis_strength`, `clear_hypothesis`, `feature_dispositions` (one row per
passing feature), and the per-hypothesis `hypotheses[]`. If the tool-call budget
is exhausted (or the model stops without reporting), the loop sends one nudge and
**forces** the final turn with `tool_choice={submit_report}`, so a run always
yields a real, parseable report rather than an empty fallback. `MAX_TOKENS` is set
high enough that the report JSON is never truncated.

**4. Provider abstraction.** `providers.OpenAICompatClient` presents the exact
`client.messages.create(...)` surface the loop expects, but talks to any
OpenAI-compatible endpoint (OpenRouter by default; `--base-url` for others). It
translates Anthropic↔OpenAI tool formats, preserves `cache_control` breakpoints
(so prompt caching works on OpenRouter too), and hardens the transport: a 300 s
timeout, retry-with-backoff on timeouts/connection errors/5xx, and a lenient body
parser that strips OpenRouter's SSE keep-alive comment lines before JSON parsing.
The client is **dependency-injected**, so the whole loop is unit-tested offline
with a fake client — no network.

**5. Tools, caching, graceful degradation.** Every evidence tool caches its
external responses on disk under `--cache-dir`, so re-runs are cheap and resilient;
a failing API returns an `{"error": …}` the agent reasons around rather than
crashing. Figure tools render headless matplotlib (300 DPI) into `<out>/figures/`
and return only `{path, caption}` — generating a figure costs a path, not image
tokens, which is why the agent is free to attach every figure that adds evidence.

**6. Auditability and batch resilience.** Each run records a full `trace.json`
(every tool call's input/output, the model's reasoning text, and token/cost usage).
At batch scale (`cli.py`), each compound runs in its own `try/except`: a transient
failure is logged as `[FAILED]`, the batch continues, and a summary lists what to
re-run.

---

## How it thinks (design principles)

These are deliberate and load-bearing — they reflect how the agent is prompted
(`prompts.py`) and how context is built (`context.py`):

- **Interpret the whole feature set, not just the top feature.** The hypothesis
  may be a single dominant feature (e.g. the drug's own target) *or* a coherent
  multi-feature signature (a pathway, a protein complex, a lineage/cell-state
  program). Features are ranked by the model's `mean_real_importance`, but the
  agent synthesises across them.
- **On-MOA and off-MOA are both valuable.** If the model-supported mechanism
  matches the drug's known MOA, good; if it diverges, that's an interesting
  finding, not a defect.
- **Judge by hypothesis strength and plausibility — not by model lift.** The
  refit/bootstrap/baseline Pearson scores are reported as context, but the
  decision to propose vs. abstain rests on the coherence of the biology and the
  quality of the evidence. A modest whole-model score can still hold an obvious
  on-target biomarker.
- **Response-direction convention (applied mechanically from the sign of r):**
  *lower response = greater sensitivity (more cell killing); higher = resistance.*
  So feature–response `r < 0` ⇒ higher feature value → more sensitive; `r > 0` ⇒
  more resistant.
- **A justified "no clear hypothesis" is a valid result.** When the features are
  incoherent noise/lineage artifacts, the agent abstains.
- **Faithfulness guardrails:** every passing feature gets an explicit disposition
  (centered / supporting / uninterpretable / likely-noise / likely-lineage-confound)
  so none is silently dropped, and a low-importance feature is never falsely
  promoted to "the strongest predictor."

Every report carries a top-level **`hypothesis_strength`** score in `[0, 1]`:

| score | meaning |
|------:|---------|
| `0.0` | no clear mechanism/biomarker (abstention) |
| `0.2–0.4` | speculative / weakly supported |
| `0.5–0.7` | plausible, reasonably supported |
| `0.8–1.0` | definite, obvious biomarker **and** mechanism (e.g. the drug's own target) |

---

## Running it

Prerequisites: [`uv`](https://docs.astral.sh/uv/), and an API key for the chosen
backend. Run from the repo root so the default `data/` paths resolve.

```bash
# Backend A: Anthropic API directly (default)
export ANTHROPIC_API_KEY=sk-...

# Backend B: OpenRouter (OpenAI-compatible gateway to Claude + others)
export OPENROUTER_API_KEY=sk-or-...
```

```bash
# One compound (writes to <dir>/interpretation by default)
uv run biomarker-analyze data/priority_samples/BRD_BRD-K25244359-066-03-4 \
    --provider openrouter --out data/interpretation_results

# A whole batch — any directory containing a MANIFEST.csv runs every compound under it
uv run biomarker-analyze data/priority_samples \
    --provider openrouter --out data/interpretation_results
```

`uv run biomarker-analyze` is the console script for `biomarker_agent.cli`; the
module form `uv run python -m biomarker_agent.cli …` is equivalent.

### Flags

| flag | default | purpose |
|------|---------|---------|
| `--provider {anthropic,openrouter}` | `anthropic` | LLM backend |
| `--model` | per-provider* | model id override |
| `--base-url` | — | override the OpenAI-compatible endpoint (any compatible service) |
| `--out` | `<compound>/interpretation` | output root |
| `--literature {pubmed,paperclip}` | `pubmed` | literature backend (paperclip needs `PAPERCLIP_API_KEY`) |
| `--max-tool-calls` | `20` | hard tool-call backstop per compound |
| `--feature-file` / `--response-file` / `--treatment-info` | `data/…` | input data overrides |
| `--cache-dir` | `.biomarker_agent_cache` | external-API response cache |

\* default model: `claude-sonnet-4-6` (anthropic) / `anthropic/claude-sonnet-4.6`
(openrouter).

**Batch resilience:** each compound runs in isolation — a transient failure (API
timeout, a malformed provider response) is logged as `[FAILED]` and skipped, the
rest of the batch continues, and a warning lists what to re-run. External calls
are cached under `--cache-dir`, so re-runs are cheap.

---

## Inputs and outputs

**Input** — a compound directory produced by the pipeline, containing
`refract/significant/significant_features.csv` (the passing features + SHAP
importance), `refract/summary.json` (metrics), and `baselines/<model>/…` (baseline
SHAP). Plus the global `data/x-all_v4.pkl` (feature matrix), `responses_*.pkl`,
and `primary_screen_treatment_info.csv`.

**Output** — per compound, under the chosen `--out/<BRD_dir>/`:

- `report.md` — human-readable report: headline, `hypothesis_strength`, summary,
  proposed mechanism(s)/biomarker(s), a **feature-disposition table**, per-hypothesis
  supporting evidence, embedded figures, and caveats.
- `report.json` — the same payload, machine-readable (consumed by the viewer).
- `figures/` — the figures the agent generated for this compound.
- `trace.json` — the full agent trace: every tool call with its inputs/outputs and
  the model's reasoning text, plus token/cost usage — so each hypothesis is auditable.

Batch runs also write `interpretation_index.md`.

---

## Tools the agent can call

Evidence tools (all cached; each degrades gracefully to `{"error": …}` the agent
works around):

| tool | source | answers |
|------|--------|---------|
| `drug_context` | treatment info CSV | known MOA / target — on- or off-mechanism? |
| `internal_association` | feature matrix + responses | does the feature track response, and which direction? |
| `depmap_dependency` | local CRISPR columns | is the gene a selective dependency? co-dependencies? |
| `string_enrichment` | STRING-DB | do the genes interact / share enriched functions? |
| `opentargets_target` | Open Targets | cancer association, druggability, known drugs |
| `cbioportal_mutations` | cBioPortal | mutation frequency in patient tumors |
| `reactome_pathways` | Reactome | pathway membership / convergence |
| `literature_search` | PubMed (default) / paperclip | established vs. novel gene–mechanism links (reads abstracts) |

Figure tools (headless matplotlib, 300 DPI; the agent shows every figure that
materially supports the hypothesis — they render behind a dropdown in the viewer):
`plot_feature_response`, `plot_two_feature_response`, `plot_feature_panel`,
`plot_dependency_distribution`, `plot_codependency_bar`, `plot_string_network`,
`plot_mutation_frequency`, `plot_pathway_membership`.

---

## Architecture

A deterministic data/tool layer feeds a single-agent tool-use loop:

```
loader.py        parse a compound dir → typed CompoundResult (features, metrics, baselines)
datactx.py       feature↔response associations (sign-correct sensitivity/resistance framing)
context.py       build the grounded seed context (drug MOA + importance-ranked features + associations)
tools/           the evidence + figure tools, behind a cached registry
agent.py         the provider-agnostic Anthropic-style tool-use loop
providers.py     OpenAI-compatible adapter (OpenRouter/etc.) with timeout + retry + lenient body parse
prompts.py       system prompt + the forced structured submit_report schema
report.py        render report.md / report.json from the structured payload
```

The LLM client is dependency-injected, so the whole pipeline is testable offline
with no live API or network.

---

## The viewer

A React + Vite SPA (`../viewer/`) for searching and reading the reports — search by
DepMap/BRD id, drug name, and feature genes (refit / baseline / hypothesis); each
report page shows the `hypothesis_strength` meter, the digest, the figures, the
feature-disposition table, and an optional full agent-trace tab.

```bash
# 1. build the static data bundle the viewer reads (re-run after producing new reports)
uv run python viewer/scripts/build_data.py \
    --results data/interpretation_results \
    --source  data/priority_samples \
    --responses data/responses_primary_v4.pkl

# 2. start the dev server
cd viewer && npm install && npm run dev   # → http://localhost:5173/
```

`--source` points at the pipeline dir holding each compound's
`significant_features.csv` and full-resolution SHAP / predicted-vs-actual figures.

---

## Tests

```bash
uv run pytest                 # Python: agent, tools, loader, report, context, providers, viewer build
cd viewer && npm test         # frontend: search, ReportView, trace rendering
```

The agent loop and tools are exercised with injected fakes — no network required.
