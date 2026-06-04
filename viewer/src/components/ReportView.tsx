import ReactMarkdown from 'react-markdown';
import type { CompoundData, FeatureDisposition } from '../lib/types';
import { figureUrl } from '../lib/data';
import Figure from './Figure';
import RefitBaselineTable from './RefitBaselineTable';

// Overall hypothesis strength (0..1): 0 = no clear mechanism/biomarker,
// 1 = definite, obvious biomarker & mechanism. Judged on biology/evidence.
function strengthLabel(v: number): string {
  if (v < 0.2) return 'no clear hypothesis';
  if (v < 0.45) return 'speculative / weak';
  if (v < 0.7) return 'plausible';
  return 'strong';
}

export function StrengthMeter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tier = value < 0.2 ? 'none' : value < 0.45 ? 'weak' : value < 0.7 ? 'mid' : 'strong';
  return (
    <div className={`strength strength-${tier}`} title="Overall hypothesis strength (0–1), judged on biology & evidence">
      <span className="strength-label">Hypothesis strength</span>
      <span className="strength-track"><span className="strength-fill" style={{ width: `${pct}%` }} /></span>
      <span className="strength-value">{value.toFixed(2)}</span>
      <span className="strength-tier">{strengthLabel(value)}</span>
    </div>
  );
}

function DispositionTable({ rows }: { rows: FeatureDisposition[] }) {
  return (
    <div className="dispositions">
      <h4>Feature dispositions (ranked by model importance)</h4>
      <table className="disp-table">
        <thead>
          <tr><th>#</th><th>Feature</th><th>imp</th><th>r</th><th>disposition</th><th>note</th></tr>
        </thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.feature}>
              <td>{d.rank}</td>
              <td>{d.feature}</td>
              <td>{typeof d.importance_ratio === 'number' ? d.importance_ratio.toFixed(2) : ''}</td>
              <td>{typeof d.r === 'number' ? (d.r >= 0 ? '+' : '') + d.r.toFixed(2) : ''}</td>
              <td><span className={`badge disp-${d.disposition}`}>{d.disposition}</span></td>
              <td className="disp-note">{d.note ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

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
  const dispositions = [...(data.feature_dispositions ?? [])].sort(
    (a, b) => (a.rank ?? 999) - (b.rank ?? 999),
  );
  const strength = data.hypothesis_strength;

  return (
    <div className="report">
      {/* --- Digest (always visible) --- */}
      {headline && <p className="headline">{headline}</p>}

      {typeof strength === 'number' && <StrengthMeter value={strength} />}

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
      {(headerFigs.length > 0 || meta.feature_comparison || dispositions.length > 0) && (
        <details className="drill">
          <summary>Model attributions &amp; performance</summary>
          <div className="drill-body">
            {headerFigs.length > 0 && (
              <div className="shap-row">
                {headerFigs.map((f) => (
                  <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
                ))}
              </div>
            )}
            {meta.feature_comparison && <RefitBaselineTable cmp={meta.feature_comparison} />}
            {dispositions.length > 0 && <DispositionTable rows={dispositions} />}
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
