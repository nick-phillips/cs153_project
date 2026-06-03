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
