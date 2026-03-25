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
  parentDate?: Date; // resolved absolute date from parent for relative datetime fields
}

type InlineEntry = { key: string; value: unknown };
type NestedEntry =
  | { kind: "array"; key: string; items: Record<string, unknown>[] }
  | { kind: "object"; key: string; obj: Record<string, unknown> };

/**
 * Detect decomposed date/datetime objects. Supported patterns:
 * - {age_years, week, day}           — date with age offset
 * - {age_years, week, day, time}     — datetime with age offset
 * - {years_passed, week, day, time}  — datetime with recursive offset
 * - {day, time}                      — relative datetime (day number + time)
 */
function isDateObject(obj: Record<string, unknown>): boolean {
  const keys = new Set(Object.keys(obj));
  if (keys.has("age_years") && keys.has("week") && keys.has("day")) return true;
  if (keys.has("years_passed") && keys.has("week") && keys.has("day")) return true;
  if (keys.has("day") && keys.has("time") && keys.size === 2) return true;
  return false;
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

/** Reconstruct approximate date from ISO week + day name. */
function weekDayToDate(year: number, weekNum: number, day: string): Date {
  const dayOfWeek = DAY_INDEX[day] ?? 1;
  const jan4 = new Date(year, 0, 4);
  const jan4Day = jan4.getDay() || 7;
  const weekStart = new Date(jan4.getTime());
  weekStart.setDate(jan4.getDate() - jan4Day + 1 + (weekNum - 1) * 7);
  weekStart.setDate(weekStart.getDate() + dayOfWeek - 1);
  return weekStart;
}

/** Pad day name to 9 chars (length of "Wednesday") for alignment. */
function padDay(day: string): string {
  return day.padEnd(9);
}

/** Format date string with aligned day number. */
function fmtDate(day: string, month: string, dateNum: number, year: number, timeSuffix: string): string {
  return `${padDay(day)} ${month} ${String(dateNum).padStart(2)}, ${year}${timeSuffix}`;
}

/** Format a decomposed date/datetime object. */
function formatDateObj(
  obj: Record<string, unknown>,
  refDateISO?: string,
  parentDate?: Date
): string {
  const rawTime = obj.time != null ? String(obj.time) : null;
  const time = rawTime && rawTime !== "00:00" ? rawTime : null;
  const timeSuffix = time ? ` ${time}` : "";

  // Pattern: {day (number), time} — relative datetime
  if ("time" in obj && "day" in obj && !("week" in obj)) {
    const dayOffset = obj.day as number;
    if (parentDate) {
      const d = new Date(parentDate.getTime());
      d.setDate(d.getDate() + dayOffset);
      const dayName = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][d.getDay()];
      const month = MONTH_NAMES[d.getMonth()];
      return fmtDate(dayName, month, d.getDate(), d.getFullYear(), timeSuffix);
    }
    return `Day ${String(dayOffset).padStart(3)}${timeSuffix}`;
  }

  // Extract years and week
  const years = (obj.age_years ?? obj.years_passed) as number;
  const week = obj.week as string | number;
  const day = obj.day as string;
  const weekNum = typeof week === "string" ? parseInt(week, 10) : week;
  const yearsLabel = "years_passed" in obj ? "years_passed" : "age_years";

  if (refDateISO) {
    const ref = new Date(refDateISO);
    const year = ref.getFullYear() + years;
    const d = weekDayToDate(year, weekNum, day);
    const month = MONTH_NAMES[d.getMonth()];
    return fmtDate(day, month, d.getDate(), d.getFullYear(), timeSuffix);
  }

  // Fallback: relative format
  const label = yearsLabel === "years_passed" ? "yrs" : "y";
  return `${padDay(day)} Wk ${String(weekNum).padStart(2)}, +${String(years).padStart(3)}${label}${timeSuffix}`;
}

/** Resolve a decomposed date object to an absolute Date, if possible. */
function resolveDateToAbsolute(
  obj: Record<string, unknown>,
  refDateISO?: string
): Date | undefined {
  if (!refDateISO) return undefined;
  const years = (obj.age_years ?? obj.years_passed) as number | undefined;
  if (years == null || !("week" in obj) || !("day" in obj)) return undefined;

  const week = obj.week as string | number;
  const day = obj.day as string;
  const weekNum = typeof week === "string" ? parseInt(week, 10) : week;

  const ref = new Date(refDateISO);
  const year = ref.getFullYear() + years;
  return weekDayToDate(year, weekNum, day);
}

/**
 * Find the first resolvable date field in an entity to use as parentDate
 * for relative datetime children. Looks for fields with age_years/years_passed.
 */
function extractParentDate(
  data: Record<string, unknown>,
  dateRefs?: DateRefs,
  tablePath?: string
): Date | undefined {
  if (!dateRefs || !tablePath) return undefined;
  const tableRefs = dateRefs[tablePath];
  if (!tableRefs) return undefined;

  for (const [key, value] of Object.entries(data)) {
    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      isDateObject(value as Record<string, unknown>)
    ) {
      const obj = value as Record<string, unknown>;
      if ("age_years" in obj || "years_passed" in obj) {
        const refISO = tableRefs[key];
        if (refISO) {
          return resolveDateToAbsolute(obj, refISO);
        }
      }
    }
  }
  return undefined;
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

  // Hide context date fields that duplicate a field inside a child table.
  // E.g., top-level "admittime" when "admissions" array items also have "admittime".
  const childFieldNames = new Set<string>();
  for (const entry of nested) {
    if (entry.kind === "array" && entry.items.length > 0) {
      for (const k of Object.keys(entry.items[0])) {
        childFieldNames.add(k);
      }
    }
  }
  const filteredInline = inline.filter(({ key, value }) => {
    if (!childFieldNames.has(key)) return true;
    // Hide null/undefined context fields that duplicate a child field
    if (value === null || value === undefined) return false;
    // Hide date context fields that duplicate a child field
    if (
      typeof value === "object" &&
      !Array.isArray(value) &&
      isDateObject(value as Record<string, unknown>)
    ) {
      return false;
    }
    return true;
  });

  return { inline: filteredInline, nested };
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
  parentDate,
}: Props) {
  const { inline, nested: rawNested } = classifyEntries(data);
  const nested = sortNested(rawNested, tableOrder);
  const compact = depth > 0 && isCompact(inline, nested);

  // Resolve this entity's anchor date for relative child datetimes.
  // Uses the first resolvable date field (age_years/years_passed + week + day).
  const resolvedDate = extractParentDate(data, dateRefs, tablePath) ?? parentDate;

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
              {formatValue(key, value, dateRefs, tablePath, resolvedDate)}
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
                parentDate={resolvedDate}
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
                    parentDate={resolvedDate}
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
              parentDate={resolvedDate}
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
  parentDate,
}: {
  fieldKey: string;
  value: unknown;
  dateRefs?: DateRefs;
  tablePath?: string;
  parentDate?: Date;
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
          {formatDateObj(
            value as Record<string, unknown>,
            refISO,
            parentDate
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
  tablePath?: string,
  parentDate?: Date
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
    return formatDateObj(
      value as Record<string, unknown>,
      refISO,
      parentDate
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
