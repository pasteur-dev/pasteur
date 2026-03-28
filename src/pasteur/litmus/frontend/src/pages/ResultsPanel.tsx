import { useEffect, useState } from "react";
import { fetchResults } from "../api";
import type {
  ExperimentResults,
  SourceResults,
  ResponseTimeStats,
  InterRaterResult,
  HumanLLMComparison,
} from "../api";

interface Props {
  experimentId: string;
  refreshKey?: number;
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

export default function ResultsPanel({ experimentId, refreshKey }: Props) {
  const [results, setResults] = useState<ExperimentResults | null>(null);

  useEffect(() => {
    fetchResults(experimentId).then(setResults);
  }, [experimentId, refreshKey]);

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

      {/* Inter-rater Agreement & Correlation */}
      {(results.inter_rater || results.human_llm_comparison) && (
        <>
          <hr className="results-divider" />
          <h3>Statistical Analysis</h3>
          <div className="stats-grid">
            {results.inter_rater && (
              <InterRaterPanel data={results.inter_rater} />
            )}
            {results.human_llm_comparison && (
              <ComparisonPanel data={results.human_llm_comparison} />
            )}
          </div>
        </>
      )}

      {/* Response Times (bottom) */}
      {results.response_times && results.response_times.length > 0 && (
        <>
          <hr className="results-divider" />
          <h3>Response Time</h3>
          <ResponseTimeChart data={results.response_times} />
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
                      intensity > (isLightMode ? 0.55 : 0.4) ? "white" : "var(--text-muted)",
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
  // Compute histogram bins across all sources, dropping top 5% outliers
  const allTimes = data.flatMap((d) => d.times);
  if (allTimes.length === 0) return null;

  const sorted = [...allTimes].sort((a, b) => a - b);
  const p95Idx = Math.floor(sorted.length * 0.95);
  const maxTime = sorted[Math.max(p95Idx - 1, 0)];
  const NUM_BINS = 20;
  const binWidth = maxTime / NUM_BINS;

  // Build histogram per source
  const histograms = data.map((d) => {
    const bins = new Array(NUM_BINS).fill(0);
    for (const t of d.times) {
      if (t > maxTime) continue; // drop outliers
      const idx = Math.min(Math.floor(t / binWidth), NUM_BINS - 1);
      bins[idx]++;
    }
    return bins;
  });

  // Max bin count for scaling
  const maxBin = Math.max(...histograms.flat(), 1);

  return (
    <div className="rt-chart">
      {data.map((d, i) => {
        const color = BAR_COLORS[i % BAR_COLORS.length];
        const bins = histograms[i];
        const meanPct = Math.min((d.mean / maxTime) * 100, 100);
        const std = d.times.length > 1
          ? Math.sqrt(d.times.reduce((acc, t) => acc + (t - d.mean) ** 2, 0) / d.times.length)
          : 0;

        return (
          <div key={d.source} className="rt-row">
            <span className="rt-label">{d.pretty_name}</span>
            <div className="rt-bar-container">
              {/* Histogram bins */}
              {bins.map((count, bi) => (
                <div
                  key={bi}
                  className="rt-hist-bin"
                  style={{
                    left: `${(bi / NUM_BINS) * 100}%`,
                    width: `${100 / NUM_BINS}%`,
                    height: `${(count / maxBin) * 100}%`,
                    backgroundColor: color,
                    opacity: 0.4,
                  }}
                />
              ))}
              {/* Mean marker */}
              <div
                className="rt-mean-line"
                style={{
                  left: `${meanPct}%`,
                  backgroundColor: color,
                }}
              />
            </div>
            <span className="rt-value">
              {d.mean.toFixed(1)}s
              <span className="rt-std"> ±{std.toFixed(1)}</span>
            </span>
          </div>
        );
      })}
      <div className="rt-axis">
        <span>0s</span>
        <span>{(maxTime / 2).toFixed(0)}s</span>
        <span>{maxTime.toFixed(0)}s</span>
      </div>
    </div>
  );
}

function InterRaterPanel({ data }: { data: InterRaterResult }) {
  const interpretAlpha = (a: number | null) => {
    if (a === null) return { label: "N/A", color: "var(--text-dim)", desc: "" };
    if (a >= 0.81) return { label: "Near-perfect", color: "#00dd77", desc: "Raters strongly agree on quality assessments." };
    if (a >= 0.61) return { label: "Substantial", color: "#44bb55", desc: "Raters mostly agree, with minor differences." };
    if (a >= 0.41) return { label: "Moderate", color: "#88aa44", desc: "Moderate agreement. Results are usable but should be interpreted with care." };
    if (a >= 0.21) return { label: "Fair", color: "#cc9900", desc: "Some agreement beyond chance. Subjective differences between raters." };
    if (a >= 0.0) return { label: "Slight", color: "#dd7733", desc: "Minimal agreement. Individual judgments vary widely." };
    return { label: "No agreement", color: "#ff2244", desc: "Less than chance agreement." };
  };

  const overall = interpretAlpha(data.overall);

  return (
    <div className="stat-card">
      <h4>Inter-Rater Agreement</h4>
      <p className="stat-desc">
        Do different evaluators rate the same data similarly?
        Higher values mean raters are consistent with each other.
      </p>
      <p className="stat-subtitle">
        Krippendorff's &alpha; (ordinal) &middot; {data.n_raters} rater{data.n_raters !== 1 ? "s" : ""}
      </p>
      <div className="stat-main">
        <span className="stat-value" style={{ color: overall.color }}>
          {data.overall !== null ? data.overall.toFixed(3) : "—"}
        </span>
        <span className="stat-interpretation" style={{ color: overall.color }}>
          {overall.label}
        </span>
      </div>
      {overall.desc && (
        <p className="stat-verdict">{overall.desc}</p>
      )}
      {data.per_source.length > 0 && (
        <>
          <p className="stat-breakdown-label">Per source:</p>
          <table className="stat-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>&alpha;</th>
                <th>Items</th>
                <th></th>
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
                    <td style={{ color: interp.color, fontSize: "0.65rem" }}>{interp.label}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
      <p className="stat-scale">
        Scale: 0 = random &middot; 1 = perfect agreement &middot; &lt;0 = systematic disagreement
      </p>
    </div>
  );
}

function ComparisonPanel({ data }: { data: HumanLLMComparison }) {
  const maxAbsDiff = Math.max(...data.per_source.map((s) => Math.abs(s.diff)), 0.5);

  const diffColor = (diff: number) => {
    const abs = Math.abs(diff);
    if (abs < 0.3) return "#44bb55";
    if (abs < 0.7) return "#cc9900";
    return "#ff4455";
  };

  return (
    <div className="stat-card">
      <h4>Human vs LLM Comparison</h4>
      <p className="stat-desc">
        How do human and LLM scores differ for each source?
        Bars show the gap between mean human and mean LLM ratings.
      </p>

      {/* Per-source mean difference bars */}
      <div className="diff-chart">
        {data.per_source.map((s) => {
          const pct = (s.diff / maxAbsDiff) * 50;
          const color = diffColor(s.diff);
          return (
            <div key={s.source} className="diff-row">
              <span className="diff-label">{s.pretty_name}</span>
              <div className="diff-bar-container">
                <div className="diff-center-line" />
                <div
                  className="diff-bar"
                  style={{
                    left: s.diff >= 0 ? "50%" : `${50 + pct}%`,
                    width: `${Math.abs(pct)}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
              <span className="diff-values">
                <span style={{ color: "var(--text-muted)" }}>
                  H:{s.human_mean.toFixed(1)}
                </span>
                {" "}
                <span style={{ color: "var(--text-dim)" }}>
                  L:{s.llm_mean.toFixed(1)}
                </span>
                {" "}
                <span style={{ color, fontWeight: 600 }}>
                  {s.diff > 0 ? "+" : ""}{s.diff.toFixed(2)}
                </span>
              </span>
            </div>
          );
        })}
        <div className="diff-axis">
          <span>LLM higher</span>
          <span>Equal</span>
          <span>Human higher</span>
        </div>
      </div>

      {/* Rank comparison */}
      <div className="rank-comparison">
        <p className="stat-breakdown-label">Ranking</p>
        <div className="rank-rows">
          <div className="rank-row">
            <span className="rank-label">Human</span>
            <span className="rank-items">
              {data.human_ranking.map((name, i) => (
                <span key={i}>
                  {i > 0 && <span className="rank-arrow">&gt;</span>}
                  <span className="rank-item">{name}</span>
                </span>
              ))}
            </span>
          </div>
          <div className="rank-row">
            <span className="rank-label">LLM</span>
            <span className="rank-items">
              {data.llm_ranking.map((name, i) => (
                <span key={i}>
                  {i > 0 && <span className="rank-arrow">&gt;</span>}
                  <span className="rank-item">{name}</span>
                </span>
              ))}
            </span>
          </div>
        </div>
        <p className="stat-verdict" style={{ color: data.rank_match ? "#00dd77" : "#cc9900" }}>
          {data.rank_match
            ? "Human and LLM rankings match"
            : "\u26A0 Human and LLM rankings differ"}
        </p>
      </div>
    </div>
  );
}

const isLightMode = window.matchMedia("(prefers-color-scheme: light)").matches;

function alphaHex(intensity: number): string {
  // On light backgrounds linear alpha fades to invisible too quickly.
  // Apply a power curve in light mode so mid-range cells stay visibly colored.
  const adjusted =
    isLightMode && intensity > 0 ? Math.pow(intensity, 0.55) : intensity;
  const alpha = Math.round(adjusted * 0.85 * 255);
  return alpha.toString(16).padStart(2, "0");
}
