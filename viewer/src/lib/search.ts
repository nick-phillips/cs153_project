import Fuse from 'fuse.js';
import type { IndexEntry } from './types';

export interface SearchResult {
  entry: IndexEntry;
  matchedFields: string[];
}

// `search_genes` is the union of bare gene symbols (e.g. "MDM4"); the
// *_features keys hold the full tokens (e.g. "shRNA_MDM4"). Both are indexed so
// a query matches whether the user types the gene or the full feature name, and
// the matched-field metadata can tell them which model surfaced it (refit vs
// baseline vs hypothesis). The overlap is intentional, not redundant.
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
