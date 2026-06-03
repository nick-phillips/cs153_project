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
