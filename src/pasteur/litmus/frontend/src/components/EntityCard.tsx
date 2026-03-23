/**
 * Recursively renders entity data as nested cards.
 *
 * Scalars and small objects (multi-value fields like dates) render inline.
 * Arrays of objects render as sub-cards.
 * Deep nesting is handled recursively.
 */

interface Props {
  data: Record<string, unknown>;
  streaming?: boolean;
  title?: string;
  depth?: number;
}

export default function EntityCard({
  data,
  streaming,
  title,
  depth = 0,
}: Props) {
  const entries = Object.entries(data);

  return (
    <div
      className={`entity-card ${depth === 0 ? "entity-card-root" : ""} ${streaming ? "entity-streaming" : ""}`}
    >
      {title && <div className="entity-card-title">{formatKey(title)}</div>}
      <div className="entity-fields">
        {entries.map(([key, value]) => (
          <FieldRenderer key={key} fieldKey={key} value={value} depth={depth} />
        ))}
      </div>
    </div>
  );
}

function FieldRenderer({
  fieldKey,
  value,
  depth,
}: {
  fieldKey: string;
  value: unknown;
  depth: number;
}) {
  // null / undefined
  if (value === null || value === undefined) {
    return (
      <div className="entity-field">
        <span className="field-key">{formatKey(fieldKey)}</span>
        <span className="field-value field-null">-</span>
      </div>
    );
  }

  // Array of objects → render as list of sub-cards
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <div className="entity-field">
          <span className="field-key">{formatKey(fieldKey)}</span>
          <span className="field-value field-null">(empty)</span>
        </div>
      );
    }
    if (typeof value[0] === "object" && value[0] !== null) {
      return (
        <div className="entity-field-nested">
          <div className="field-key field-key-section">
            {formatKey(fieldKey)} ({value.length})
          </div>
          <div className="entity-subcard-list">
            {value.map((item, i) => (
              <EntityCard
                key={i}
                data={item as Record<string, unknown>}
                title={`${formatKey(fieldKey)} ${i + 1}`}
                depth={depth + 1}
              />
            ))}
          </div>
        </div>
      );
    }
    // Array of scalars
    return (
      <div className="entity-field">
        <span className="field-key">{formatKey(fieldKey)}</span>
        <span className="field-value">{value.map(String).join(", ")}</span>
      </div>
    );
  }

  // Object — check if it's a "small" multi-value field (like a date/time)
  // or a larger nested structure
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const objEntries = Object.entries(obj);

    // Small object (all scalar values, ≤5 fields): render inline
    const allScalar = objEntries.every(
      ([, v]) => v === null || typeof v !== "object"
    );
    if (allScalar && objEntries.length <= 5) {
      const parts = objEntries
        .filter(([, v]) => v !== null && v !== undefined)
        .map(([k, v]) => `${formatKey(k)}: ${formatScalar(v)}`)
        .join(", ");
      return (
        <div className="entity-field">
          <span className="field-key">{formatKey(fieldKey)}</span>
          <span className="field-value field-multivalue">{parts || "-"}</span>
        </div>
      );
    }

    // Larger nested object → sub-card
    return (
      <div className="entity-field-nested">
        <EntityCard data={obj} title={fieldKey} depth={depth + 1} />
      </div>
    );
  }

  // Scalar
  return (
    <div className="entity-field">
      <span className="field-key">{formatKey(fieldKey)}</span>
      <span className="field-value">{formatScalar(value)}</span>
    </div>
  );
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}
