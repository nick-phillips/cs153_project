import type { TopFeature } from '../lib/types';

// Ranked refit-model features (top N, importance desc) with a mini bar scaled
// to the largest importance in the set. The perturbation type (GE/shRNA/CRISPR)
// is shown as a colored tag.
export default function TopFeatures({ features }: { features: TopFeature[] }) {
  if (!features.length) return null;
  const max = Math.max(...features.map((f) => f.importance)) || 1;
  return (
    <ol className="top-features" aria-label="Top refit-model features">
      {features.map((f) => (
        <li key={f.name} className="top-feature" title={`${f.name} · importance ${f.importance.toFixed(4)}`}>
          <span className={`pert pert-${f.klass}`}>{f.klass}</span>
          <span className="tf-gene">{f.gene}</span>
          <span className="tf-bar">
            <span className="tf-fill" style={{ width: `${(f.importance / max) * 100}%` }} />
          </span>
        </li>
      ))}
    </ol>
  );
}
