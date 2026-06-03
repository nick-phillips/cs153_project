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
