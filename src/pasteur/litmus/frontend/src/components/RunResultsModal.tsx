import type { RunResults, SourceResults } from "../api";

const HEATMAP_COLORS = [
  "#ff2244",
  "#ff6633",
  "#cc9900",
  "#44bb55",
  "#00dd77",
];

interface Props {
  results: RunResults;
  onClose: () => void;
}

export default function RunResultsModal({ results, onClose }: Props) {
  let maxCount = 1;
  for (const data of results.by_source) {
    for (const s of [1, 2, 3, 4, 5]) {
      maxCount = Math.max(maxCount, data.distribution[s] || 0);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{results.name || "Run Results"}</h2>
          <button className="btn btn-small" onClick={onClose}>
            &times;
          </button>
        </div>
        <p className="results-summary">
          {results.total_rated} rated, {results.total_skipped} skipped
        </p>

        {results.by_source.length > 0 ? (
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
              {results.by_source.map((data) => (
                <HeatmapRow
                  key={data.pretty_name}
                  data={data}
                  maxCount={maxCount}
                />
              ))}
            </tbody>
          </table>
        ) : (
          <p>No ratings in this run.</p>
        )}
      </div>
    </div>
  );
}

function HeatmapRow({
  data,
  maxCount,
}: {
  data: SourceResults;
  maxCount: number;
}) {
  return (
    <tr>
      <td className="heatmap-label">{data.pretty_name}</td>
      {[1, 2, 3, 4, 5].map((s) => {
        const count = data.distribution[s] || 0;
        const pct = data.count > 0 ? (count / data.count) * 100 : 0;
        const intensity = count / maxCount;
        return (
          <td
            key={s}
            className="heatmap-cell"
            style={{
              backgroundColor: `${HEATMAP_COLORS[s - 1]}${alphaHex(intensity)}`,
              color: intensity > 0.4 ? "white" : "var(--text-muted)",
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
  );
}

function alphaHex(intensity: number): string {
  const alpha = Math.round(intensity * 0.8 * 255);
  return alpha.toString(16).padStart(2, "0");
}
