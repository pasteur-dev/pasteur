/**
 * Displays a model reference.
 *
 * On ExperimentPage (showTimestamp=true): overrides first, timestamp as dim tag.
 * Elsewhere (showTimestamp=false): just algorithm + overrides, no timestamp.
 */
import type { ModelRef } from "../api";

interface Props {
  model: ModelRef;
  showTimestamp?: boolean;
}

export default function ModelLabel({ model, showTimestamp = false }: Props) {
  const overrides = Object.entries(model.overrides || {});
  const hasOverrides = overrides.length > 0;

  return (
    <span className="model-tag">
      <span className="model-tag-alg">{model.algorithm}</span>
      {hasOverrides &&
        overrides.map(([k, v]) => (
          <span key={k} className="model-tag-override">
            {k}={String(v)}
          </span>
        ))}
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
