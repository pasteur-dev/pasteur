/**
 * Displays a model reference.
 *
 * - Default: short label using prettyName (context-aware, only differing params)
 * - showTimestamp: include timestamp after the label
 *
 * For table-style verbose display, use ModelTable instead.
 */
import type { ModelRef } from "../api";

interface Props {
  model: ModelRef;
  /** Context-aware short name (only params differing within experiment) */
  prettyName?: string;
  showTimestamp?: boolean;
}

export default function ModelLabel({
  model,
  prettyName,
  showTimestamp = false,
}: Props) {
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

/** Table layout for model lists: Algorithm | Timestamp | Overrides */
export function ModelTable({
  models,
  prettyNames,
  includeReal,
}: {
  models: ModelRef[];
  prettyNames?: Record<string, string>;
  includeReal?: boolean;
}) {
  return (
    <table className="model-table">
      <tbody>
        {models.map((m, i) => {
          const overrides = Object.entries(m.overrides || {});
          const key = `${m.algorithm}_${m.timestamp || "latest"}`;
          const rawName = prettyNames?.[key] || m.algorithm;
          // Split trailing (N) suffix from name into overrides
          const suffixMatch = rawName.match(/^(.+?)\s+\((\d+)\)$/);
          const name = suffixMatch ? suffixMatch[1] : rawName;
          const suffix = suffixMatch ? suffixMatch[2] : null;
          return (
            <tr key={i}>
              <td className="model-table-alg">{name}</td>
              <td className="model-table-ts">{formatTimestamp(m.timestamp)}</td>
              <td className="model-table-overrides">
                {suffix && (
                  <span className="model-table-param model-table-suffix">
                    #{suffix}
                  </span>
                )}
                {overrides.map(([k, v]) => (
                  <span key={k} className="model-table-param">
                    {k}={String(v)}
                  </span>
                ))}
              </td>
            </tr>
          );
        })}
        {includeReal && (
          <tr className="model-table-row-real">
            <td className="model-table-alg model-table-real">Real Data</td>
            <td className="model-table-ts"></td>
            <td className="model-table-overrides"></td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

function formatTimestamp(ts: string): string {
  return ts
    .replace(/T/, " ")
    .replace(/\.\d+Z$/, "")
    .replace(/\./g, ":");
}
