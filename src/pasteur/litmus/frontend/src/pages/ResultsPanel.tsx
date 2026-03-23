import { useEffect, useState } from "react";
import { fetchResults } from "../api";
import type { ExperimentResults } from "../api";

interface Props {
  experimentId: string;
}

export default function ResultsPanel({ experimentId }: Props) {
  const [results, setResults] = useState<ExperimentResults | null>(null);

  useEffect(() => {
    fetchResults(experimentId).then(setResults);
  }, [experimentId]);

  if (!results) return <div className="card">Loading results...</div>;

  if (results.total_rated === 0) {
    return (
      <div className="card">
        <h2>Results</h2>
        <p>No ratings yet.</p>
      </div>
    );
  }

  const sources = Object.entries(results.by_source);

  return (
    <div className="card results-panel">
      <h2>Results: {results.name || "Unnamed"}</h2>
      <p>
        {results.view} &middot; {results.total_rated} rated,{" "}
        {results.total_skipped} skipped
      </p>

      {/* Mean scores bar chart */}
      <h3>Mean Realism Score</h3>
      <div className="bar-chart">
        {sources.map(([source, data]) => (
          <div key={source} className="bar-row">
            <span className="bar-label">{source}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${(data.mean / 5) * 100}%`,
                  backgroundColor: source === "real" ? "#28a745" : "#4a90d9",
                }}
              />
            </div>
            <span className="bar-value">{data.mean.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* Distribution table */}
      <h3>Rating Distribution</h3>
      <table className="dist-table">
        <thead>
          <tr>
            <th>Source</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {sources.map(([source, data]) => (
            <tr key={source}>
              <td>{source}</td>
              {[1, 2, 3, 4, 5].map((s) => (
                <td key={s}>{data.distribution[s] || 0}</td>
              ))}
              <td>{data.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
