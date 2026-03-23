import { useEffect, useState } from "react";
import { fetchResults } from "../api";
import type { ExperimentResults, SourceResults } from "../api";

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
        <col className="heatmap-col-mean" />
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
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function alphaHex(intensity: number): string {
  const alpha = Math.round(intensity * 0.8 * 255);
  return alpha.toString(16).padStart(2, "0");
}
