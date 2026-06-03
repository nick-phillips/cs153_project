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
  'drug_name',
  'compound_id',
  'moa',
  'targets',
  'refit_features',
  'baseline_features',
  'hypothesis_genes',
  'search_genes',
  'top_hypothesis_title',
] as const;

type Key = (typeof KEYS)[number];

function fieldValues(entry: IndexEntry, key: Key): string[] {
  const v = entry[key];
  if (v == null) return [];
  return Array.isArray(v) ? v.map(String) : [String(v)];
}

// Score how well a field value matches the needle (already lower-cased):
// exact > prefix > substring. 0 means no match.
function valueScore(value: string, needle: string): number {
  const v = value.toLowerCase();
  if (v === needle) return 100;
  if (v.startsWith(needle)) return 60;
  if (v.includes(needle)) return 20;
  return 0;
}

// Primary search: exact case-insensitive substring match across the indexed
// fields. This is strict and predictable — "NEK" only matches entries that
// literally contain "NEK" (e.g. the gene NEK6), never fuzzy near-misses.
function substringSearch(entries: IndexEntry[], needle: string): SearchResult[] {
  const out: Array<SearchResult & { score: number }> = [];
  for (const entry of entries) {
    const matchedFields: string[] = [];
    let best = 0;
    for (const key of KEYS) {
      let fieldBest = 0;
      for (const val of fieldValues(entry, key)) {
        fieldBest = Math.max(fieldBest, valueScore(val, needle));
      }
      if (fieldBest > 0) matchedFields.push(key);
      best = Math.max(best, fieldBest);
    }
    if (matchedFields.length) out.push({ entry, matchedFields, score: best });
  }
  // Best match quality first; ties broken by refit performance.
  out.sort(
    (a, b) =>
      b.score - a.score ||
      (b.entry.performance.refit ?? -1) - (a.entry.performance.refit ?? -1),
  );
  return out.map(({ entry, matchedFields }) => ({ entry, matchedFields }));
}

export function makeFuse(entries: IndexEntry[]): Fuse<IndexEntry> {
  // Fallback only (when nothing matches as a substring): tolerate small typos
  // but stay tight so we don't resurrect the over-permissive matching.
  return new Fuse(entries, {
    keys: KEYS.map((name) => ({ name })),
    includeMatches: true,
    ignoreLocation: true,
    threshold: 0.3,
    minMatchCharLength: 3,
  });
}

export function search(
  fuse: Fuse<IndexEntry>,
  query: string,
  entries: IndexEntry[],
): SearchResult[] {
  const q = query.trim();
  if (!q) return entries.map((entry) => ({ entry, matchedFields: [] }));

  const exact = substringSearch(entries, q.toLowerCase());
  if (exact.length) return exact;

  // No literal match — fall back to fuzzy so typos still find something.
  return fuse.search(q).map((r) => ({
    entry: r.item,
    matchedFields: Array.from(new Set((r.matches ?? []).map((m) => m.key as string))),
  }));
}
