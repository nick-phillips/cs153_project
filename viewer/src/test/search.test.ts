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
    top_features: [],
  },
  {
    id: 'C2', compound_id: 'BRD:2', drug_name: 'Posaconazole', moa: 'STEROL', targets: '',
    has_hypothesis: false, performance: { refit: 0.1, bootstrap: 0, baseline: 0 },
    divergence: 'low', top_hypothesis_title: null,
    refit_features: ['GE_FOO'], baseline_features: [], hypothesis_genes: [], search_genes: ['FOO'],
    top_features: [],
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

  it('fuzzy-matches a gene typo only as a fallback', () => {
    const r = search(makeFuse(entries), 'MDM5', entries);
    expect(r.map((x) => x.entry.id)).toContain('C1');
  });

  it('matches a gene prefix as a strict substring (no fuzzy bleed)', () => {
    // "KRT" is a substring of C1's gene KRT20 and unrelated to C2 — so C2 must
    // not appear via approximate matching.
    const r = search(makeFuse(entries), 'KRT', entries);
    expect(r.map((x) => x.entry.id)).toEqual(['C1']);
  });

  it('does not return unrelated entries for a short fragment', () => {
    // "OPIOID" is C1's MOA; C2 (STEROL) must not fuzzy-match.
    const r = search(makeFuse(entries), 'OPIOID', entries);
    expect(r.map((x) => x.entry.id)).toEqual(['C1']);
  });

  it('returns nothing for a query that neither matches nor nearly matches', () => {
    expect(search(makeFuse(entries), 'ZZZQQQ', entries)).toEqual([]);
  });

  it('returns all entries for an empty query', () => {
    expect(search(makeFuse(entries), '', entries).length).toBe(2);
  });
});
