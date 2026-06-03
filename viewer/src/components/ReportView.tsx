import ReactMarkdown from 'react-markdown';
import type { CompoundData } from '../lib/types';
import { figureUrl } from '../lib/data';
import Figure from './Figure';
import RefitBaselineTable from './RefitBaselineTable';

export default function ReportView({ data }: { data: CompoundData }) {
  const { meta } = data;
  const headerFigs = (meta.header_figures ?? []).filter((f) => f.path);
  const hyps = [...(data.hypotheses ?? [])].sort(
    (a, b) => (a.rank ?? 999) - (b.rank ?? 999),
  );

  return (
    <div className="report">
      {headerFigs.length > 0 && (
        <section>
          <h3>Model feature attributions (SHAP)</h3>
          <div className="shap-row">
            {headerFigs.map((f) => (
              <Figure key={f.path} src={figureUrl(data.id, f.path)} caption={f.caption} />
            ))}
          </div>
        </section>
      )}

      {meta.feature_comparison && <RefitBaselineTable cmp={meta.feature_comparison} />}

      <section>
        <h3>Summary</h3>
        <ReactMarkdown>{data.summary || ''}</ReactMarkdown>
      </section>

      <section>
        <h3>Proposed mechanism(s) of anticancer action</h3>
        {data.proposed_mechanisms?.length ? (
          <ul>{data.proposed_mechanisms.map((m, i) => <li key={i}>{m}</li>)}</ul>
        ) : (
          <p className="muted"><em>No clear mechanism hypothesis is supported by the evidence.</em></p>
        )}
      </section>

      <section>
        <h3>Proposed biomarker(s) of response</h3>
        {data.proposed_biomarkers?.length ? (
          <ul>{data.proposed_biomarkers.map((b, i) => <li key={i}>{b}</li>)}</ul>
        ) : (
          <p className="muted"><em>No clear biomarker hypothesis is supported by the evidence.</em></p>
        )}
      </section>

      {hyps.length > 0 ? (
        <section>
          <h3>Supporting evidence</h3>
          {hyps.map((h) => (
            <details key={h.rank} className="hypothesis" open={h.rank === 1}>
              <summary>
                <span className="hyp-rank">{h.rank}.</span> {h.title}
                {h.novelty && <span className="badge novelty">{h.novelty}</span>}
                {h.kind && <span className="badge kind">{h.kind}</span>}
                {typeof h.confidence === 'number' && (
                  <span className="badge conf">conf {h.confidence.toFixed(2)}</span>
                )}
              </summary>
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
            </details>
          ))}
        </section>
      ) : (
        <section className="no-hypothesis">
          <h3>No hypothesis proposed</h3>
          <p>
            The agent did not find a hypothesis supported by the evidence for this
            compound. See the summary above and the caveats below for the reasoning.
          </p>
        </section>
      )}

      {data.caveats?.length > 0 && (
        <section>
          <h3>Caveats</h3>
          <ul>{data.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </section>
      )}
    </div>
  );
}
