import ReactMarkdown from 'react-markdown';
import type { CompoundData } from '../lib/types';
import { figureUrl } from '../lib/data';
import Figure from './Figure';
import RefitBaselineTable from './RefitBaselineTable';

// Progressive disclosure: a short, digestible "headline + subheadings" digest is
// always visible; heavier supporting material (model figures, per-hypothesis
// evidence, caveats) lives in collapsible drill-downs.
export default function ReportView({ data }: { data: CompoundData }) {
  const { meta } = data;
  const headerFigs = (meta.header_figures ?? []).filter((f) => f.path);
  const hyps = [...(data.hypotheses ?? [])].sort(
    (a, b) => (a.rank ?? 999) - (b.rank ?? 999),
  );
  const hasHyp = hyps.length > 0;
  // Older reports predate the `headline` field — fall back to the top hypothesis title.
  const headline = data.headline?.trim() || (hasHyp ? hyps[0].title : '');
  const mechanisms = data.proposed_mechanisms ?? [];
  const biomarkers = data.proposed_biomarkers ?? [];

  return (
    <div className="report">
      {/* --- Digest (always visible) --- */}
      {headline && <p className="headline">{headline}</p>}

      {hasHyp ? (
        <div className="findings">
          {mechanisms.length > 0 && (
            <div className="finding-block">
              <span className="finding-label">Proposed mechanism</span>
              <ul>{mechanisms.map((m, i) => <li key={i}>{m}</li>)}</ul>
            </div>
          )}
          {biomarkers.length > 0 && (
            <div className="finding-block">
              <span className="finding-label">Proposed biomarker(s)</span>
              <ul>{biomarkers.map((b, i) => <li key={i}>{b}</li>)}</ul>
            </div>
          )}
        </div>
      ) : (
        <div className="no-hypothesis">
          <h3>No hypothesis proposed</h3>
          <p>
            The agent did not find a hypothesis supported by the evidence for this compound —
            see the summary and caveats below for the reasoning.
          </p>
        </div>
      )}

      {data.summary && (
        <div className="summary-block">
          <ReactMarkdown>{data.summary}</ReactMarkdown>
        </div>
      )}

      {/* --- Drill-downs (collapsed) --- */}
      {headerFigs.length > 0 && (
        <details className="drill">
          <summary>Model attributions &amp; performance</summary>
          <div className="drill-body">
            <div className="shap-row">
              {headerFigs.map((f) => (
                <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
              ))}
            </div>
            {meta.feature_comparison && <RefitBaselineTable cmp={meta.feature_comparison} />}
          </div>
        </details>
      )}

      {hasHyp && (
        <details className="drill">
          <summary>Supporting evidence</summary>
          <div className="drill-body">
            {hyps.map((h) => (
              <div key={h.rank} className="hypothesis-detail">
                <div className="hyp-head">
                  {h.novelty && <span className="badge novelty">{h.novelty}</span>}
                  {h.kind && <span className="badge kind">{h.kind}</span>}
                  {typeof h.confidence === 'number' && (
                    <span className="badge conf">conf {h.confidence.toFixed(2)}</span>
                  )}
                </div>
                <p className="features"><strong>Features:</strong> {h.features?.join(', ')}</p>
                <ReactMarkdown>{h.mechanism || ''}</ReactMarkdown>
                {h.evidence && Object.keys(h.evidence).length > 0 && (
                  <>
                    <h4>Evidence</h4>
                    <dl className="evidence">
                      {Object.entries(h.evidence).map(([k, v]) => (
                        <div key={k}>
                          <dt>{k}</dt>
                          <dd>{v}</dd>
                        </div>
                      ))}
                    </dl>
                  </>
                )}
                {(h.figures ?? []).map((f) => (
                  <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
                ))}
              </div>
            ))}
          </div>
        </details>
      )}

      {data.caveats?.length > 0 && (
        <details className="drill">
          <summary>Caveats ({data.caveats.length})</summary>
          <div className="drill-body">
            <ul>{data.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
          </div>
        </details>
      )}
    </div>
  );
}
