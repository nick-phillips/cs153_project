import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { loadIndex } from '../lib/data';
import { makeFuse, search } from '../lib/search';
import type { IndexEntry } from '../lib/types';
import SearchBar from '../components/SearchBar';
import PerfBadges from '../components/PerfBadges';

const FIELD_LABEL: Record<string, string> = {
  drug_name: 'drug name',
  compound_id: 'compound id',
  moa: 'MOA',
  targets: 'target',
  refit_features: 'refit feature',
  baseline_features: 'baseline feature',
  hypothesis_genes: 'hypothesis gene',
  search_genes: 'gene',
  top_hypothesis_title: 'hypothesis',
};

export default function IndexPage() {
  const [entries, setEntries] = useState<IndexEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [hypOnly, setHypOnly] = useState(false);

  useEffect(() => {
    loadIndex().then(setEntries).catch((e) => setError(String(e)));
  }, []);

  const fuse = useMemo(() => (entries ? makeFuse(entries) : null), [entries]);

  const results = useMemo(() => {
    if (!entries || !fuse) return [];
    let r = search(fuse, query, entries);
    if (hypOnly) r = r.filter((x) => x.entry.has_hypothesis);
    if (!query.trim()) {
      r = [...r].sort((a, b) => {
        if (a.entry.has_hypothesis !== b.entry.has_hypothesis) {
          return a.entry.has_hypothesis ? -1 : 1;
        }
        return (b.entry.performance.refit ?? -1) - (a.entry.performance.refit ?? -1);
      });
    }
    return r;
  }, [entries, fuse, query, hypOnly]);

  if (error) return <div className="page"><p className="error">{error}</p></div>;
  if (!entries) return <div className="page"><p>Loading…</p></div>;

  return (
    <div className="page">
      <header className="site-header">
        <h1>Biomarker Interpretation Results</h1>
        <p className="muted">{entries.length} compounds</p>
      </header>

      <SearchBar value={query} onChange={setQuery} />
      <label className="toggle">
        <input
          type="checkbox"
          checked={hypOnly}
          onChange={(e) => setHypOnly(e.target.checked)}
        />
        Has hypothesis only
      </label>

      {results.length === 0 ? (
        <p className="muted">
          {query.trim()
            ? `No compounds match "${query}".`
            : 'No compounds match.'}
        </p>
      ) : (
        <ul className="compound-list">
          {results.map(({ entry, matchedFields }) => (
            <li key={entry.id} className="compound-card">
              <Link to={`/c/${entry.id}`} className="card-link">
                <div className="card-title">
                  {entry.drug_name || entry.compound_id}
                  {!entry.has_hypothesis && (
                    <span className="badge muted-badge">no hypothesis</span>
                  )}
                  {entry.divergence && (
                    <span className={`badge div-${entry.divergence}`}>
                      {entry.divergence} divergence
                    </span>
                  )}
                </div>
                <div className="card-id">{entry.compound_id}</div>
                <div className="card-moa">
                  {entry.moa}
                  {entry.targets ? ` · ${entry.targets}` : ''}
                </div>
                <PerfBadges perf={entry.performance} />
                {entry.has_hypothesis && entry.top_hypothesis_title && (
                  <div className="card-hyp">{entry.top_hypothesis_title}</div>
                )}
                {query.trim() && matchedFields.length > 0 && (
                  <div className="matched">
                    {Array.from(new Set(matchedFields.map((f) => FIELD_LABEL[f] ?? f))).map(
                      (f) => (
                        <span key={f} className="badge match">matched: {f}</span>
                      ),
                    )}
                  </div>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
