/**
 * Displays a model reference.
 *
 * - Default: short label using prettyName (context-aware, only differing params)
 * - verbose: full overrides, one per line, with timestamp
 */
import type { ModelRef } from "../api";

interface Props {
  model: ModelRef;
  /** Context-aware short name (only params differing within experiment) */
  prettyName?: string;
  /** Show full overrides, one per line, with timestamp */
  verbose?: boolean;
  showTimestamp?: boolean;
}

export default function ModelLabel({
  model,
  prettyName,
  verbose = false,
  showTimestamp = false,
}: Props) {
  const overrides = Object.entries(model.overrides || {});

  if (verbose) {
    return (
      <div className="model-tag model-tag-verbose">
        <span className="model-tag-alg">{model.algorithm}</span>
        {overrides.length > 0 && (
          <div className="model-tag-overrides-list">
            {overrides.map(([k, v]) => (
              <span key={k} className="model-tag-override-line">
                {k}={String(v)}
              </span>
            ))}
          </div>
        )}
        <span className="model-tag-ts">{formatTimestamp(model.timestamp)}</span>
      </div>
    );
  }

  // Short mode: use prettyName if available
  const label = prettyName || model.algorithm;

  return (
    <span className="model-tag">
      <span className="model-tag-alg">{label}</span>
      {showTimestamp && (
        <span className="model-tag-ts">{formatTimestamp(model.timestamp)}</span>
      )}
    </span>
  );
}

function formatTimestamp(ts: string): string {
  return ts
    .replace(/T/, " ")
    .replace(/\.\d+Z$/, "")
    .replace(/\./g, ":");
}
