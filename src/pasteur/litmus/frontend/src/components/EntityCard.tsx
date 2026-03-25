/**
 * Recursively renders entity data as nested cards.
 *
 * Inline fields (scalars, small objects) render in a table grid.
 * Nested fields (arrays of objects, large objects) render below as sub-cards.
 * Compact entities (few short scalar fields, no nesting) render as a single line.
 *
 * Date objects ({age_years, week, day}) are detected and formatted as dates.
 * If dateRefs is provided, absolute dates are reconstructed; otherwise a
 * relative format is used as fallback.
 */

interface DateRefs {
  [table: string]: { [field: string]: string }; // table -> field -> ISO ref date
}

interface Props {
  data: Record<string, unknown>;
  streaming?: boolean;
  title?: string;
  depth?: number;
  dateRefs?: DateRefs;
  tablePath?: string; // tracks current table for date ref lookup
  tableOrder?: string[];
}

type InlineEntry = { key: string; value: unknown };
type NestedEntry =
  | { kind: "array"; key: string; items: Record<string, unknown>[] }
  | { kind: "object"; key: string; obj: Record<string, unknown> };

/** Detect if an object is a decomposed date: {age_years, week, day}. */
function isDateObject(
  obj: Record<string, unknown>
): obj is { age_years: number; week: string | number; day: string } {
  const keys = Object.keys(obj);
  return (
    keys.includes("age_years") &&
    keys.includes("week") &&
    keys.includes("day")
  );
}

const DAY_INDEX: Record<string, number> = {
  Monday: 1,
  Tuesday: 2,
  Wednesday: 3,
  Thursday: 4,
  Friday: 5,
  Saturday: 6,
  Sunday: 7,
};

const MONTH_NAMES = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/**
 * Format a decomposed date. If refDateISO is provided, reconstruct an
 * approximate absolute date. Otherwise fall back to relative display.
 */
function formatDate(
  obj: { age_years: number; week: string | number; day: string },
  refDateISO?: string
): string {
  const { age_years, week, day } = obj;
  const weekNum = typeof week === "string" ? parseInt(week, 10) : week;

  if (refDateISO) {
    const ref = new Date(refDateISO);
    const year = ref.getFullYear() + age_years;

    // Approximate: ISO week to month/day
    // Jan 4 is always in week 1. Each week is 7 days.
    const dayOfWeek = DAY_INDEX[day] ?? 1;
    const jan4 = new Date(year, 0, 4);
    const jan4Day = jan4.getDay() || 7; // ISO: Monday=1
    const weekStart = new Date(jan4.getTime());
    weekStart.setDate(jan4.getDate() - jan4Day + 1 + (weekNum - 1) * 7);
    weekStart.setDate(weekStart.getDate() + dayOfWeek - 1);

    const month = MONTH_NAMES[weekStart.getMonth()];
    const dateNum = weekStart.getDate();
    return `${day}, ${month} ${dateNum}, ${weekStart.getFullYear()}`;
  }

  // Fallback: relative format
  return `${day}, Wk ${weekNum}, +${age_years}y`;
}

function classifyEntries(data: Record<string, unknown>) {
  const inline: InlineEntry[] = [];
  const nested: NestedEntry[] = [];

  for (const [key, value] of Object.entries(data)) {
    if (value === null || value === undefined) {
      inline.push({ key, value });
    } else if (Array.isArray(value)) {
      if (value.length === 0) {
        // Skip empty arrays entirely
      } else if (
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
      // Check if it's a date object first
      if (isDateObject(obj)) {
        inline.push({ key, value });
      } else {
        const entries = Object.entries(obj);
        const allScalar = entries.every(
          ([, v]) => v === null || typeof v !== "object"
        );
        if (allScalar && entries.length <= 5) {
          inline.push({ key, value });
        } else {
          nested.push({ kind: "object", key, obj });
        }
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

/** Reorder nested entries to match tableOrder if provided. */
function sortNested(
  nested: NestedEntry[],
  tableOrder?: string[]
): NestedEntry[] {
  if (!tableOrder || tableOrder.length === 0) return nested;
  return [...nested].sort((a, b) => {
    const ai = tableOrder.indexOf(a.key);
    const bi = tableOrder.indexOf(b.key);
    const oa = ai >= 0 ? ai : tableOrder.length;
    const ob = bi >= 0 ? bi : tableOrder.length;
    return oa - ob;
  });
}

export default function EntityCard({
  data,
  streaming,
  title,
  depth = 0,
  dateRefs,
  tablePath,
  tableOrder,
}: Props) {
  const { inline, nested: rawNested } = classifyEntries(data);
  const nested = sortNested(rawNested, tableOrder);
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
              {formatValue(key, value, dateRefs, tablePath)}
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
              <InlineField
                key={key}
                fieldKey={key}
                value={value}
                dateRefs={dateRefs}
                tablePath={tablePath}
              />
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
                    dateRefs={dateRefs}
                    tablePath={entry.key}
                    tableOrder={tableOrder}
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
              dateRefs={dateRefs}
              tablePath={entry.key}
              tableOrder={tableOrder}
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
  dateRefs,
  tablePath,
}: {
  fieldKey: string;
  value: unknown;
  dateRefs?: DateRefs;
  tablePath?: string;
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

  // Date object
  if (
    typeof value === "object" &&
    !Array.isArray(value) &&
    isDateObject(value as Record<string, unknown>)
  ) {
    const refISO =
      tablePath && dateRefs?.[tablePath]?.[fieldKey]
        ? dateRefs[tablePath][fieldKey]
        : undefined;
    return (
      <tr>
        <td className="field-key">{formatKey(fieldKey)}</td>
        <td className="field-value">
          {formatDate(
            value as { age_years: number; week: string | number; day: string },
            refISO
          )}
        </td>
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

function formatValue(
  key: string,
  value: unknown,
  dateRefs?: DateRefs,
  tablePath?: string
): string {
  if (value === null || value === undefined) return "-";
  if (
    typeof value === "object" &&
    !Array.isArray(value) &&
    isDateObject(value as Record<string, unknown>)
  ) {
    const refISO =
      tablePath && dateRefs?.[tablePath]?.[key]
        ? dateRefs[tablePath][key]
        : undefined;
    return formatDate(
      value as { age_years: number; week: string | number; day: string },
      refISO
    );
  }
  return formatScalar(value);
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "\u2705" : "\u274c";
  if (value === "True" || value === "true") return "\u2705";
  if (value === "False" || value === "false") return "\u274c";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}
