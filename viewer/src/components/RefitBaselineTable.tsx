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
