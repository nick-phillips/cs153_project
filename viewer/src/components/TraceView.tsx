import ReactMarkdown from 'react-markdown';
import type { Trace, TraceEntry } from '../lib/types';

function isText(e: TraceEntry): e is { event: 'assistant_text'; text: string } {
  return 'event' in e && e.event === 'assistant_text';
}

function ToolCall({ entry }: { entry: { tool: string; input: unknown; output: unknown } }) {
  return (
    <details className="tool-call">
      <summary><span className="tool-name">{entry.tool}</span></summary>
      <div className="tool-io">
        <h5>Input</h5>
        <pre>{JSON.stringify(entry.input, null, 2)}</pre>
        <h5>Output</h5>
        <pre>{JSON.stringify(entry.output, null, 2)}</pre>
      </div>
    </details>
  );
}

export default function TraceView({ trace }: { trace: Trace | null }) {
  if (!trace) return <p className="muted">No trace recorded for this compound.</p>;
  const u = trace.usage ?? {};
  return (
    <div className="trace">
      <div className="trace-footer">
        <span className="badge">{trace.model}</span>
        {typeof u.prompt_tokens === 'number' && (
          <span className="badge">{u.prompt_tokens} prompt tok</span>
        )}
        {typeof u.completion_tokens === 'number' && (
          <span className="badge">{u.completion_tokens} completion tok</span>
        )}
        {typeof u.cached_tokens === 'number' && u.cached_tokens > 0 && (
          <span className="badge">{u.cached_tokens} cached</span>
        )}
        {typeof u.cost_usd === 'number' && (
          <span className="badge cost">${u.cost_usd.toFixed(3)}</span>
        )}
      </div>

      <details className="seed">
        <summary>Seed context</summary>
        <ReactMarkdown>{trace.seed_context || ''}</ReactMarkdown>
      </details>

      <ol className="timeline">
        {trace.transcript.map((e, i) => (
          <li key={i}>
            {isText(e) ? (
              <div className="assistant-text"><ReactMarkdown>{e.text}</ReactMarkdown></div>
            ) : (
              <ToolCall entry={e} />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
