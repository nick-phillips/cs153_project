import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { loadCompound } from '../lib/data';
import type { CompoundData } from '../lib/types';
import ReportView from '../components/ReportView';
import TraceView from '../components/TraceView';
import PerfBadges from '../components/PerfBadges';

type Tab = 'report' | 'trace';

export default function CompoundPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<CompoundData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('report');

  useEffect(() => {
    if (!id) return;
    setData(null);
    setError(null);
    loadCompound(id).then(setData).catch((e) => setError(String(e)));
  }, [id]);

  if (error) {
    return (
      <div className="page">
        <Link to="/" className="back">← All compounds</Link>
        <p className="error">{error}</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="page">
        <Link to="/" className="back">← All compounds</Link>
        <p>Loading…</p>
      </div>
    );
  }

  const { meta } = data;
  return (
    <div className="page compound-page">
      <Link to="/" className="back">← All compounds</Link>
      <header className="compound-header">
        <h1>{meta.drug_name || data.compound_id}</h1>
        <div className="card-id">{data.compound_id}</div>
        <div className="card-moa">
          {meta.moa}
          {meta.targets ? ` · target: ${meta.targets}` : ''}
          {typeof meta.n_samples === 'number' ? ` · n=${meta.n_samples} cell lines` : ''}
        </div>
        <PerfBadges
          perf={{
            refit: meta.performance?.selected_refit_oob_pearson ?? null,
            bootstrap: meta.performance?.bootstrap_pred_pearson ?? null,
            baseline: meta.performance?.baseline_pred_pearson ?? null,
          }}
        />
      </header>

      <nav className="tabs">
        <button className={tab === 'report' ? 'active' : ''} onClick={() => setTab('report')}>
          Report
        </button>
        <button className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
          Agent trace
        </button>
      </nav>

      {tab === 'report' ? <ReportView data={data} /> : <TraceView trace={data.trace} />}
    </div>
  );
}
