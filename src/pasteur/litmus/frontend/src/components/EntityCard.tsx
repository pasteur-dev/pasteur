/**
 * Renders an entity dict as structured cards.
 *
 * Top-level scalar fields render as a key-value card.
 * Top-level object fields render as their own card.
 * Top-level arrays of objects render as a table card.
 */

interface Props {
  data: Record<string, unknown>;
  streaming?: boolean;
}

export default function EntityCard({ data, streaming }: Props) {
  // Separate scalars, objects, and arrays
  const scalars: [string, string | number][] = [];
  const objects: [string, Record<string, unknown>][] = [];
  const arrays: [string, Record<string, unknown>[]][] = [];

  for (const [key, value] of Object.entries(data)) {
    if (value === null || value === undefined) {
      scalars.push([key, "-"]);
    } else if (Array.isArray(value)) {
      if (value.length > 0 && typeof value[0] === "object") {
        arrays.push([key, value as Record<string, unknown>[]]);
      } else {
        scalars.push([key, String(value)]);
      }
    } else if (typeof value === "object") {
      objects.push([key, value as Record<string, unknown>]);
    } else {
      scalars.push([key, value as string | number]);
    }
  }

  return (
    <div className={`entity-cards ${streaming ? "entity-streaming" : ""}`}>
      {/* Main scalar fields */}
      {scalars.length > 0 && (
        <div className="entity-card">
          <div className="entity-fields">
            {scalars.map(([key, value]) => (
              <div key={key} className="entity-field">
                <span className="field-key">{formatKey(key)}</span>
                <span className="field-value">{formatValue(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Nested object fields as separate cards */}
      {objects.map(([key, obj]) => (
        <div key={key} className="entity-card">
          <div className="entity-card-title">{formatKey(key)}</div>
          <div className="entity-fields">
            {Object.entries(obj).map(([k, v]) => (
              <div key={k} className="entity-field">
                <span className="field-key">{formatKey(k)}</span>
                <span className="field-value">
                  {formatValue(v as string | number | null)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Array fields as table cards */}
      {arrays.map(([key, rows]) => (
        <div key={key} className="entity-card entity-card-wide">
          <div className="entity-card-title">{formatKey(key)}</div>
          <div className="entity-table-wrap">
            <table className="entity-table">
              <thead>
                <tr>
                  {Object.keys(rows[0]).map((col) => (
                    <th key={col}>{formatKey(col)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((val, j) => (
                      <td key={j}>
                        {typeof val === "object" && val !== null
                          ? JSON.stringify(val)
                          : formatValue(val as string | number | null)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}
