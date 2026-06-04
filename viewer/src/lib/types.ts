export interface Performance {
  refit: number | null;
  bootstrap: number | null;
  baseline: number | null;
}

export interface TopFeature {
  name: string; // full token, e.g. "GE_KRT20"
  gene: string; // bare symbol, e.g. "KRT20"
  klass: string; // perturbation type, e.g. "GE" | "shRNA" | "CRISPR"
  importance: number; // mean real SHAP importance (refit model)
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
  top_features: TopFeature[];
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
  headline?: string;
  summary: string;
  clear_hypothesis: boolean;
  hypotheses: Hypothesis[];
  proposed_mechanisms: string[];
  proposed_biomarkers: string[];
  caveats: string[];
  trace: Trace | null;
  top_features?: TopFeature[];
}
