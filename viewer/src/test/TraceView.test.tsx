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
