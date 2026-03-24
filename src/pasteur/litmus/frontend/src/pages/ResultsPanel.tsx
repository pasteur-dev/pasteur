import { useEffect, useState } from "react";
import { fetchResults } from "../api";
import type {
  ExperimentResults,
  SourceResults,
  ResponseTimeStats,
  InterRaterResult,
  HumanLLMCorrelation,
} from "../api";

interface Props {
  experimentId: string;
}

const HEATMAP_COLORS = [
  "#ff2244",
  "#ff6633",
  "#cc9900",
  "#44bb55",
  "#00dd77",
];

const BAR_COLORS = [
  "#ff2244",
  "#ff6633",
  "#cc9900",
  "#44bb55",
  "#00dd77",
  "#6699dd",
  "#aa66cc",
];

export default function ResultsPanel({ experimentId }: Props) {
  const [results, setResults] = useState<ExperimentResults | null>(null);

  useEffect(() => {
    fetchResults(experimentId).then(setResults);
  }, [experimentId]);

  if (!results) return <div className="card">Loading results...</div>;

  if (
    results.total_rated === 0 &&
    results.llm_scores.length === 0
  ) {
    return (
      <div className="card">
        <h2>Results</h2>
        <p>No ratings yet.</p>
      </div>
    );
  }

  return (
    <div className="card results-panel">
      <h2>Results: {results.name || "Unnamed"}</h2>
      <p className="results-summary">
        {results.view} &middot; {results.num_runs} run
        {results.num_runs !== 1 ? "s" : ""} &middot;{" "}
        {results.total_rated} rated, {results.total_skipped} skipped
      </p>

      {results.by_source.length > 0 && (
        <>
          <h3>Human Rating Distribution</h3>
          <RatingHeatmap sources={results.by_source} />
        </>
      )}

      {results.llm_scores.length > 0 && (
        <>
          <h3>LLM Rating Distribution</h3>
          <RatingHeatmap sources={results.llm_scores} />
        </>
      )}

      {/* Response Times */}
      {results.response_times && results.response_times.length > 0 && (
        <>
          <hr className="results-divider" />
          <h3>Response Time</h3>
          <ResponseTimeChart data={results.response_times} />
        </>
      )}

      {/* Inter-rater Agreement & Correlation */}
      {(results.inter_rater || results.human_llm_correlation) && (
        <>
          <hr className="results-divider" />
          <h3>Statistical Analysis</h3>
          <div className="stats-grid">
            {results.inter_rater && (
              <InterRaterPanel data={results.inter_rater} />
            )}
            {results.human_llm_correlation && (
              <CorrelationPanel data={results.human_llm_correlation} />
            )}
          </div>
        </>
      )}
    </div>
  );
}

function RatingHeatmap({ sources }: { sources: SourceResults[] }) {
  let maxCount = 1;
  for (const data of sources) {
    for (const s of [1, 2, 3, 4, 5]) {
      maxCount = Math.max(maxCount, data.distribution[s] || 0);
    }
  }

  return (
    <table className="heatmap-table">
      <colgroup>
        <col className="heatmap-col-label" />
        <col className="heatmap-col-score" />
        <col className="heatmap-col-score" />
        <col className="heatmap-col-score" />
        <col className="heatmap-col-score" />
        <col className="heatmap-col-score" />
        <col className="heatmap-col-n" />
        <col className="heatmap-col-stat" />
        <col className="heatmap-col-stat" />
        <col className="heatmap-col-stat" />
      </colgroup>
      <thead>
        <tr>
          <th></th>
          {[1, 2, 3, 4, 5].map((s) => (
            <th key={s} style={{ color: HEATMAP_COLORS[s - 1] }}>
              {s}
            </th>
          ))}
          <th>N</th>
          <th>Mean</th>
          <th>Std</th>
          <th>Med</th>
        </tr>
      </thead>
      <tbody>
        {sources.map((data) => (
          <tr key={data.pretty_name}>
            <td className="heatmap-label">{data.pretty_name}</td>
            {[1, 2, 3, 4, 5].map((s) => {
              const count = data.distribution[s] || 0;
              const pct =
                data.count > 0 ? (count / data.count) * 100 : 0;
              const intensity = count / maxCount;
              return (
                <td
                  key={s}
                  className="heatmap-cell"
                  style={{
                    backgroundColor: `${HEATMAP_COLORS[s - 1]}${alphaHex(intensity)}`,
                    color:
                      intensity > 0.4 ? "white" : "var(--text-muted)",
                  }}
                >
                  {count > 0 && (
                    <>
                      <span className="heatmap-count">{count}</span>
                      <span className="heatmap-pct">
                        ({Math.min(99, Math.round(pct))}%)
                      </span>
                    </>
                  )}
                </td>
              );
            })}
            <td className="heatmap-n">{data.count}</td>
            <td className="heatmap-mean">{data.mean.toFixed(2)}</td>
            <td className="heatmap-stat">{data.std.toFixed(2)}</td>
            <td className="heatmap-stat">{data.median.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResponseTimeChart({ data }: { data: ResponseTimeStats[] }) {
  const maxMean = Math.max(...data.map((d) => d.mean), 1);

  return (
    <div className="rt-chart">
      {data.map((d, i) => {
        const pct = (d.mean / maxMean) * 100;
        const color = BAR_COLORS[i % BAR_COLORS.length];
        return (
          <div key={d.source} className="rt-row">
            <span className="rt-label">{d.pretty_name}</span>
            <div className="rt-bar-container">
              <div
                className="rt-bar"
                style={{
                  width: `${pct}%`,
                  backgroundColor: color,
                }}
              />
            </div>
            <span className="rt-value">{d.mean.toFixed(1)}s</span>
          </div>
        );
      })}
    </div>
  );
}

function InterRaterPanel({ data }: { data: InterRaterResult }) {
  const interpretAlpha = (a: number | null) => {
    if (a === null) return { label: "N/A", color: "var(--text-dim)" };
    if (a >= 0.8) return { label: "Excellent", color: "#00dd77" };
    if (a >= 0.67) return { label: "Good", color: "#44bb55" };
    if (a >= 0.33) return { label: "Fair", color: "#cc9900" };
    return { label: "Poor", color: "#ff2244" };
  };

  const overall = interpretAlpha(data.overall);

  return (
    <div className="stat-card">
      <h4>Inter-Rater Agreement</h4>
      <p className="stat-subtitle">
        Krippendorff's &alpha; (ordinal) &middot; {data.n_raters} raters
      </p>
      <div className="stat-main">
        <span className="stat-value" style={{ color: overall.color }}>
          {data.overall !== null ? data.overall.toFixed(3) : "—"}
        </span>
        <span className="stat-interpretation" style={{ color: overall.color }}>
          {overall.label}
        </span>
      </div>
      {data.per_source.length > 0 && (
        <table className="stat-table">
          <thead>
            <tr>
              <th>Source</th>
              <th>&alpha;</th>
              <th>Items</th>
            </tr>
          </thead>
          <tbody>
            {data.per_source.map((s) => {
              const interp = interpretAlpha(s.alpha);
              return (
                <tr key={s.pretty_name}>
                  <td>{s.pretty_name}</td>
                  <td style={{ color: interp.color }}>{s.alpha.toFixed(3)}</td>
                  <td>{s.n_items}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function CorrelationPanel({ data }: { data: HumanLLMCorrelation }) {
  const interpretCorr = (r: number) => {
    const abs = Math.abs(r);
    if (abs >= 0.8) return { label: "Strong", color: "#00dd77" };
    if (abs >= 0.5) return { label: "Moderate", color: "#44bb55" };
    if (abs >= 0.3) return { label: "Weak", color: "#cc9900" };
    return { label: "Negligible", color: "#ff6633" };
  };

  const spearman = interpretCorr(data.spearman_rho);

  return (
    <div className="stat-card">
      <h4>Human vs LLM Correlation</h4>
      <p className="stat-subtitle">
        {data.n_sources} sources compared
      </p>
      <div className="stat-main">
        <span className="stat-value" style={{ color: spearman.color }}>
          &rho; = {data.spearman_rho.toFixed(3)}
        </span>
        <span className="stat-interpretation" style={{ color: spearman.color }}>
          {spearman.label}
        </span>
      </div>
      {data.pearson_r !== null && (
        <div className="stat-secondary">
          Pearson r = {data.pearson_r.toFixed(3)}
        </div>
      )}
    </div>
  );
}

function alphaHex(intensity: number): string {
  const alpha = Math.round(intensity * 0.8 * 255);
  return alpha.toString(16).padStart(2, "0");
}
