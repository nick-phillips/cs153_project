import { useEffect, useMemo, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { loadIndex } from '../lib/data';
import { makeFuse, search } from '../lib/search';
import type { IndexEntry } from '../lib/types';
import SearchBar from './SearchBar';
import PerfBadges from './PerfBadges';
import TopFeatures from './TopFeatures';

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

export default function Sidebar() {
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

  return (
    <aside className="sidebar">
      <header className="sidebar-header">
        <h1>Biomarker results</h1>
        {entries && <p className="muted">{entries.length} compounds</p>}
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

      {error && <p className="error">{error}</p>}
      {!entries && !error && <p className="muted">Loading…</p>}

      {entries && results.length === 0 && (
        <p className="muted">No compounds match{query.trim() ? ` “${query}”` : ''}.</p>
      )}

      {entries && results.length > 0 && (
        <ul className="compound-list">
          {results.map(({ entry, matchedFields }) => (
            <li key={entry.id} className="compound-card">
              <NavLink
                to={`/c/${entry.id}`}
                className={({ isActive }) => `card-link${isActive ? ' active' : ''}`}
              >
                <div className="card-title">{entry.drug_name || entry.compound_id}</div>
                <div className="card-id">{entry.compound_id}</div>
                <div className="card-moa">
                  {entry.moa}
                  {entry.targets ? ` · ${entry.targets}` : ''}
                </div>

                <div className="stat-row">
                  {typeof entry.hypothesis_strength === 'number' && (
                    <span
                      className={`badge strength-chip ${
                        entry.hypothesis_strength < 0.2
                          ? 'sc-none'
                          : entry.hypothesis_strength < 0.45
                            ? 'sc-weak'
                            : entry.hypothesis_strength < 0.7
                              ? 'sc-mid'
                              : 'sc-strong'
                      }`}
                      title="Hypothesis strength (0–1)"
                    >
                      strength {entry.hypothesis_strength.toFixed(2)}
                    </span>
                  )}
                  <PerfBadges perf={entry.performance} />
                  {entry.divergence && (
                    <span className={`badge div-${entry.divergence}`}>
                      {entry.divergence} divergence
                    </span>
                  )}
                  {!entry.has_hypothesis && (
                    <span className="badge muted-badge">no hypothesis</span>
                  )}
                </div>

                <TopFeatures features={entry.top_features} />

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
              </NavLink>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
