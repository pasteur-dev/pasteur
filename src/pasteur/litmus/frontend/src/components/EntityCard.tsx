/**
 * Recursively renders entity data as nested cards.
 *
 * Inline fields (scalars, small objects) render in a table grid.
 * Nested fields (arrays of objects, large objects) render below as sub-cards.
 * Compact entities (few short scalar fields, no nesting) render as a single line.
 */

interface Props {
  data: Record<string, unknown>;
  streaming?: boolean;
  title?: string;
  depth?: number;
}

type InlineEntry = { key: string; value: unknown };
type NestedEntry =
  | { kind: "array"; key: string; items: Record<string, unknown>[] }
  | { kind: "object"; key: string; obj: Record<string, unknown> };

function classifyEntries(data: Record<string, unknown>) {
  const inline: InlineEntry[] = [];
  const nested: NestedEntry[] = [];

  for (const [key, value] of Object.entries(data)) {
    if (value === null || value === undefined) {
      inline.push({ key, value });
    } else if (Array.isArray(value)) {
      if (
        value.length > 0 &&
        typeof value[0] === "object" &&
        value[0] !== null
      ) {
        nested.push({
          kind: "array",
          key,
          items: value as Record<string, unknown>[],
        });
      } else {
        inline.push({ key, value });
      }
    } else if (typeof value === "object") {
      const obj = value as Record<string, unknown>;
      const entries = Object.entries(obj);
      const allScalar = entries.every(
        ([, v]) => v === null || typeof v !== "object"
      );
      if (allScalar && entries.length <= 5) {
        inline.push({ key, value });
      } else {
        nested.push({ kind: "object", key, obj });
      }
    } else {
      inline.push({ key, value });
    }
  }

  return { inline, nested };
}

/** Check if an entity is compact enough to render in one line. */
function isCompact(
  inline: InlineEntry[],
  nested: NestedEntry[]
): boolean {
  if (nested.length > 0) return false;
  if (inline.length > 4) return false;
  // Check total character length of keys+values
  let total = 0;
  for (const { key, value } of inline) {
    total += key.length;
    if (value === null || value === undefined) {
      total += 1;
    } else if (typeof value === "object" && !Array.isArray(value)) {
      // Multi-value objects are too wide for compact
      return false;
    } else {
      total += String(value).length;
    }
  }
  return total < 60;
}

export default function EntityCard({
  data,
  streaming,
  title,
  depth = 0,
}: Props) {
  const { inline, nested } = classifyEntries(data);
  const compact = depth > 0 && isCompact(inline, nested);

  if (compact) {
    return (
      <div
        className={`entity-compact ${streaming ? "entity-streaming" : ""}`}
      >
        {inline.map(({ key, value }, i) => (
          <span key={key} className="compact-field">
            {i > 0 && <span className="compact-sep">&middot;</span>}
            <span className="field-key">{formatKey(key)}:</span>{" "}
            <span className="field-value">
              {formatScalar(value as string | number | null)}
            </span>
          </span>
        ))}
      </div>
    );
  }

  return (
    <div
      className={`entity-card ${depth === 0 ? "entity-card-root" : ""} ${streaming ? "entity-streaming" : ""}`}
    >
      {title && <div className="entity-card-title">{formatKey(title)}</div>}

      {/* Inline fields as table */}
      {inline.length > 0 && (
        <table className="entity-field-table">
          <tbody>
            {inline.map(({ key, value }) => (
              <InlineField key={key} fieldKey={key} value={value} />
            ))}
          </tbody>
        </table>
      )}

      {/* Nested fields below */}
      {nested.map((entry) => {
        if (entry.kind === "array") {
          return (
            <div key={entry.key} className="entity-field-nested">
              <div className="field-key field-key-section">
                {formatKey(entry.key)} ({entry.items.length})
              </div>
              <div className="entity-subcard-list">
                {entry.items.map((item, i) => (
                  <EntityCard
                    key={i}
                    data={item}
                    depth={depth + 1}
                  />
                ))}
              </div>
            </div>
          );
        }
        return (
          <div key={entry.key} className="entity-field-nested">
            <EntityCard
              data={entry.obj}
              title={entry.key}
              depth={depth + 1}
            />
          </div>
        );
      })}
    </div>
  );
}

function InlineField({
  fieldKey,
  value,
}: {
  fieldKey: string;
  value: unknown;
}) {
  // null
  if (value === null || value === undefined) {
    return (
      <tr>
        <td className="field-key">{formatKey(fieldKey)}</td>
        <td className="field-value field-null">-</td>
      </tr>
    );
  }

  // Small object → inline multi-value
  if (typeof value === "object" && !Array.isArray(value)) {
    const parts = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${formatKey(k)}: ${formatScalar(v)}`)
      .join(", ");
    return (
      <tr>
        <td className="field-key">{formatKey(fieldKey)}</td>
        <td className="field-value field-multivalue">{parts || "-"}</td>
      </tr>
    );
  }

  // Array of scalars
  if (Array.isArray(value)) {
    return (
      <tr>
        <td className="field-key">{formatKey(fieldKey)}</td>
        <td className="field-value">{value.map(String).join(", ")}</td>
      </tr>
    );
  }

  // Scalar
  return (
    <tr>
      <td className="field-key">{formatKey(fieldKey)}</td>
      <td className="field-value">{formatScalar(value)}</td>
    </tr>
  );
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "✅" : "❌";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}
