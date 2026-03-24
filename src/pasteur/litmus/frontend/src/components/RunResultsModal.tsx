import type { RunResults, SourceResults, ResponseTimeStats } from "../api";

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

        {results.response_times && results.response_times.length > 0 && (
          <>
            <hr className="results-divider" />
            <h3>Response Time</h3>
            <ResponseTimeChart data={results.response_times} />
          </>
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
      <td className="heatmap-stat">{data.std.toFixed(2)}</td>
      <td className="heatmap-stat">{data.median.toFixed(1)}</td>
    </tr>
  );
}

function ResponseTimeChart({ data }: { data: ResponseTimeStats[] }) {
  const allTimes = data.flatMap((d) => d.times);
  if (allTimes.length === 0) return null;

  const sorted = [...allTimes].sort((a, b) => a - b);
  const p95Idx = Math.floor(sorted.length * 0.95);
  const maxTime = sorted[Math.max(p95Idx - 1, 0)];
  const NUM_BINS = 20;
  const binWidth = maxTime / NUM_BINS;

  const histograms = data.map((d) => {
    const bins = new Array(NUM_BINS).fill(0);
    for (const t of d.times) {
      if (t > maxTime) continue;
      const idx = Math.min(Math.floor(t / binWidth), NUM_BINS - 1);
      bins[idx]++;
    }
    return bins;
  });

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

function alphaHex(intensity: number): string {
  const alpha = Math.round(intensity * 0.8 * 255);
  return alpha.toString(16).padStart(2, "0");
}
