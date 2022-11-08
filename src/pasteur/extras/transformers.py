


class NumericalTransformer(Transformer):
    """Clips numerical values and attaches metadata to them."""

    name = "numerical"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self,
        bins: int = 20,
        find_edges: bool = False,
        min: float | int | None = None,
        max: float | int | None = None,
        nullable: bool = False,
        **_,
    ):
        self.nullable = nullable
        self.bins = bins
        self.min = min
        self.max = max
        self.find_edges = find_edges

    def fit(self, data: pd.Series):
        self.col = data.name
        self.dtype = data.dtype
        if self.min is None and self.find_edges:
            self.min = data.min()
        if self.max is None and self.find_edges:
            self.max = data.max()
        self.attr = NumAttribute(self.col, self.bins, self.min, self.max, self.nullable)
        return self.attr

    def transform(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(data.clip(self.min, self.max).astype("float32"))

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        return data[self.col].clip(self.min, self.max).astype(self.dtype)


class IdxTransformer(Transformer):
    """Transforms categorical values of any type into integer based values.

    If the values are sortable, they will have adjacent integer values"""

    name = "categorical"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(self, unknown_value=None, nullable: bool = False, **_):
        self.unknown_value = unknown_value
        self.nullable = nullable
        self.ordinal = False

    def fit(self, data: pd.Series):
        vals = [v for v in data.unique() if not pd.isna(v)]

        # Try to sort vals
        try:
            vals = sorted(vals)
        except:
            assert not self.ordinal, "Ordinal Array is not sortable"

        vals = list(vals)
        ofs = 0
        if self.nullable:
            ofs += 1
        if self.unknown_value is not None:
            ofs += 1

        self.mapping = {val: i + ofs for i, val in enumerate(vals)}
        self.vals = {i + ofs: val for i, val in enumerate(vals)}
        self.col = data.name
        self.domain = ofs + len(vals)
        self.ofs = ofs

        self.type = data.dtype
        cls = OrdAttribute if self.ordinal else CatAttribute
        self.attr = cls(data.name, vals, self.nullable, self.unknown_value)
        return self.attr

    def transform(self, data: pd.Series) -> pd.DataFrame:
        mapping = self.mapping
        type = get_dtype(self.domain)
        out_col = data.replace(mapping)

        # Handle categorical columns without blowing them up to full blown columns
        if is_categorical_dtype(out_col):
            out_col = out_col.cat.add_categories(range(self.ofs))

        # Handle NAs correctly
        if self.nullable:
            out_col = out_col.fillna(0)
        else:
            assert not np.any(data.isna())

        if self.unknown_value is not None:
            out_col = out_col.where(
                data.isin(mapping.keys()) | data.isna(), 1 if self.nullable else 0
            )
        else:
            assert np.all(data.isin(mapping.keys()) | data.isna())

        return pd.DataFrame(out_col.astype(type))

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        col = data[self.col]
        out = col.map(self.vals)

        if self.nullable:
            out = out.where(col != 0, pd.NA)
        if self.unknown_value is not None:
            out = out.where(col != (1 if self.nullable else 0), self.unknown_value)

        return out.astype(self.type)


class OrdinalTransformer(IdxTransformer):
    name = "ordinal"

    def __init__(self, unknown_value=None, nullable: bool = False, **_):
        super().__init__(unknown_value, nullable, **_)
        self.ordinal = True


class DateTransformer(RefTransformer):
    name = "date"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self, span: str = "year", nullable: bool = False, bins=64, max_len=63, **_
    ):
        self.weeks53 = span == "year53"
        if self.weeks53:
            self.span = "year"
        else:
            # Since last week is trimmed, transform is not lossless
            self.lossless = span == "year"
            self.span = span

        self.nullable = nullable
        self.bins = bins
        self.max_len = max_len

    def fit(
        self,
        data: pd.Series,
        ref: pd.Series | None = None,
    ):
        self.ref = data.min() if ref is None else None
        col = data.name
        self.col = col

        # Generate constraints for columns
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        match self.span:
            case "year":
                self.attr = Attribute(
                    col,
                    {
                        f"{col}_year": NumColumn(self.bins, 0, self.max_len),
                        f"{col}_week": OrdColumn(
                            range(53 if self.weeks53 else 52), na=self.nullable
                        ),
                        f"{col}_day": OrdColumn(days, na=self.nullable),
                    },
                    self.nullable,
                )
            case "week":
                self.attr = Attribute(
                    col,
                    {
                        f"{col}_week": NumColumn(self.bins, 0, self.max_len),
                        f"{col}_day": OrdColumn(days, na=self.nullable),
                    },
                    self.nullable,
                )
            case "day":
                self.attr = Attribute(
                    col,
                    {
                        f"{col}_day": NumColumn(self.bins, 0, self.max_len),
                    },
                    self.nullable,
                )

        return self.attr

    @staticmethod
    def iso_year_start(iso_year):
        "The gregorian calendar date of the first day of the given ISO year"
        # Based on https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar
        fourth_jan = pd.to_datetime(
            pd.DataFrame({"year": iso_year, "month": 1, "day": 4}), errors="coerce"
        )

        delta = pd.to_timedelta(fourth_jan.dt.day_of_week, unit="day")
        return fourth_jan - delta

    @staticmethod
    def iso_to_gregorian(iso_year, iso_week, iso_day):
        "Gregorian calendar date for the given ISO year, week and day"
        year_start = DateTransformer.iso_year_start(iso_year)
        return year_start + pd.to_timedelta(
            (iso_week - 1) * 7 + iso_day - 1, unit="day"
        )

    def transform(self, data: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
        out = pd.DataFrame()
        col = self.col
        vals = data

        if self.nullable:
            na_mask = pd.isna(vals)
            if ref is not None:
                na_mask |= pd.isna(ref)
                ref = ref[~na_mask]
            vals = vals[~na_mask]
        else:
            assert not np.any(
                pd.isna(vals)
            ), f"NA values detected in non-NA field: {self.col}"

        rf = self.ref if self.ref else ref
        assert rf is not None
        # When using a ref column accessing the date parameters is done by the dt member.
        # When self referencing to the minimum value, its type is a Timestamp
        # which doesn't have the dt member and requires direct access.
        rf_dt = rf if isinstance(rf, pd.Timestamp) else rf.dt

        iso = vals.dt.isocalendar()
        iso_rf = rf_dt.isocalendar()

        if isinstance(rf, pd.Timestamp):
            rf_year = rf_dt.year
            rf_day = iso_rf.weekday
        else:
            rf_year = rf_dt.year
            rf_day = iso_rf["day"]

        ofs = 1 if self.nullable else 0

        match self.span:
            case "year":
                year = vals.dt.year - rf_year

                weeks = iso["week"] - 1
                if not self.weeks53:
                    # Put days in week 53 at the beginning of next year
                    m = weeks == 52
                    year[m] = year[m] + 1
                    weeks[m] = 0

                out[f"{col}_year"] = year.astype("float32")
                out[f"{col}_week"] = (weeks + ofs).astype("uint8")
                out[f"{col}_day"] = (iso["day"] - 1 + ofs).astype("uint8")
            case "week":
                week = (
                    (vals.dt.normalize() - rf_dt.normalize()).dt.days + rf_day - 1
                ) // 7
                out[f"{col}_week"] = week.astype("float32")
                out[f"{col}_day"] = (iso["day"] - 1 + ofs).astype("uint8")
            case "day":
                day = (vals.dt.normalize() - rf_dt.normalize()).dt.days + rf_day - 1
                out[f"{col}_day"] = day.astype("float32")

        if self.nullable:
            out = out.reindex(data.index, fill_value=0)
            # NAs were set as 0, change them to floats
            out.loc[na_mask, f"{col}_{self.span}"] = np.nan

        return out

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.Series:
        col = self.col

        vals = data

        # Check for nullability in the columns below
        fcol = f"{self.col}_{self.span}"
        match self.span:
            case "year":
                dcols = [f"{self.col}_week", f"{self.col}_day"]
            case "week":
                dcols = [f"{self.col}_day"]
            case _:
                dcols = []

        if self.nullable:
            na_mask = pd.isna(vals[fcol])
            if dcols:
                na_mask |= np.any(vals[dcols] == 0, axis=1)

            if ref is not None:
                na_mask |= pd.isna(ref)
                ref = ref[~na_mask]
            vals = vals[~na_mask]
            ofs = 1
        else:
            ofs = 0
            assert not np.any(pd.isna(vals[fcol])), "NAN values found on nonNAN col"

        rf = self.ref if self.ref is not None else ref
        assert rf is not None
        # When using a ref column accessing the date parameters is done by the dt member.
        # When self referencing to the minimum value, its type is a Timestamp
        # which doesn't have the dt member and requires direct access.
        if isinstance(rf, pd.Timestamp):
            rf_dt = rf
            rf_year = rf_dt.year
            iso_rf = rf.isocalendar()
            rf_day = iso_rf.weekday
        else:
            rf_dt = rf.dt
            rf_year = rf_dt.year
            iso_rf = rf.dt.isocalendar()
            rf_day = iso_rf["day"]

        match self.span:
            case "year":
                out = self.iso_to_gregorian(
                    rf_year + np.round(vals[f"{col}_year"]).clip(0),
                    (vals[f"{col}_week"] + 1 - ofs)
                    .clip(1, 53 if self.weeks53 else 52)
                    .astype("uint16"),
                    (vals[f"{col}_day"] + 1 - ofs).clip(0, 7).astype("uint16"),
                )
            case "week":
                out = rf + pd.to_timedelta(
                    np.round(vals[f"{col}_week"]).clip(0) * 7
                    + (vals[f"{col}_day"] - ofs).clip(0, 6).astype("uint16")
                    - rf_day
                    + 1,
                    unit="days",
                )
            case "day":
                # TODO: fix negative spans
                out = rf_dt.normalize() + pd.to_timedelta(
                    (np.round(vals[f"{col}_day"]) - rf_day + 1).astype("int32"),
                    unit="days",
                )

        return out.reindex(data.index, fill_value=pd.NaT).rename(self.col)


class TimeTransformer(Transformer):
    name = "time"

    deterministic = True
    lossless = True
    stateful = True

    def __init__(self, span: str = "minute", nullable: bool = False, **_):
        self.span = span
        self.nullable = nullable

    def fit(
        self,
        data: pd.Series,
    ):
        span = self.span
        self.col = data.name

        hours = []
        for hour in range(24):
            if span == "hour":
                hours.append(f"{hour:02d}:00")
            elif span == "halfhour":
                hours.append(
                    Level(
                        "ord",
                        [f"{hour:02d}:00", f"{hour:02d}:30"],
                    )
                )
            else:
                mins = []
                for min in range(60):
                    if span == "minute":
                        mins.append(f"{hour:02d}:{min:02d}")
                    if span == "halfminute":
                        mins.append(
                            Level(
                                "ord",
                                [
                                    f"{hour:02d}:{min:02d}:00",
                                    f"{hour:02d}:{min:02d}:30",
                                ],
                            )
                        )
                    if span == "second":
                        secs = []
                        for sec in range(60):
                            secs.append(f"{hour:02d}:{min:02d}:{sec:02d}")
                        mins.append(Level("ord", secs))

                hours.append(Level("ord", mins))
        lvl = Level("ord", hours)
        if self.nullable:
            lvl = Level("cat", [None, lvl])

        self.domain = lvl.size

        self.attr = Attribute(
            data.name, {f"{data.name}_time": LevelColumn(lvl)}, self.nullable
        )
        return self.attr

    def transform(self, date: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=date.index)
        span = self.span

        out = date.dt.hour
        if span == "halfhour":
            out = out * 2 + (date.dt.minute > 29)
        if span in ("minute", "halfminute", "second"):
            out = out * 60 + date.dt.minute
        if span == "halfminute":
            out = out * 2 + (date.dt.second > 29)
        if span == "second":
            out = out * 60 + date.dt.second

        if self.nullable:
            out += 1
            out = out.where(~pd.isna(date), 0)
        else:
            assert not np.any(
                pd.isna(date)
            ), f"NA values detected in non-NA field: {self.col}"

        out = out.astype(get_dtype(self.domain))
        return pd.DataFrame({f"{self.col}_time": out})

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        span = self.span
        col = self.col

        vals = data[f"{col}_time"]

        if self.nullable:
            na_mask = vals == 0
            vals = vals[~na_mask] - 1

        match span:
            case "hour":
                hour = vals
                min = 0
                sec = 0
            case "halfhour":
                hour = vals // 2
                min = 30 * (vals % 2)
                sec = 0
            case "minute":
                hour = vals // 60
                min = vals % 60
                sec = 0
            case "halfminute":
                hour = vals // 120
                min = (vals // 2) % 60
                sec = 30 * (vals % 2)
            case "second":
                hour = vals // 3600
                min = (vals // 60) % 60
                sec = vals % 60

        out = pd.to_datetime(
            {
                "year": 2000,
                "month": 1,
                "day": 1,
                "hour": hour,
                "minute": min,
                "second": sec,
            }
        )

        if self.nullable:
            out_data = out
            out = pd.Series(pd.NaT, index=data.index, name=col)
            out[~na_mask] = out_data
        else:
            out.name = col

        return out


class DatetimeTransformer(RefTransformer):
    name = "datetime"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(self, span="year.halfhour", **kwargs):
        date_span, time_span = span.split(".")
        self.nullable = kwargs.get("nullable", False)
        self.dt = DateTransformer(date_span, **kwargs)
        self.tt = TimeTransformer(time_span, **kwargs)

    def fit(
        self,
        data: pd.Series,
        ref: pd.Series | None = None,
    ) -> dict[str, dict] | None:
        self.col = data.name

        cdt = self.dt.fit(data, ref)
        ctt = self.tt.fit(data)
        self.attr = Attribute(self.col, cols={**cdt.cols, **ctt.cols}, na=self.nullable)
        return self.attr

    def transform(self, data: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
        date_enc = self.dt.transform(data, ref)
        time_enc = self.tt.transform(data)
        return pd.concat([date_enc, time_enc], axis=1)

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        date_dec = self.dt.reverse(data, ref)
        time_dec = self.tt.reverse(data)

        out = pd.to_datetime(
            {
                "year": date_dec.dt.year,
                "month": date_dec.dt.month,
                "day": date_dec.dt.day,
                "hour": time_dec.dt.hour,
                "minute": time_dec.dt.minute,
                "second": time_dec.dt.second,
            }
        )
        out.name = self.col

        return out


class FixedValueTransformer(Transformer):
    """The transform function of this transformer returns an empty dataframe and
    when reversing it returns the columns with a fixed value.

    Used for the anchoring date of a table."""

    name = "fixed"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self, dtype: Literal["date", "int", "float"] = "date", value: any = None, **_
    ) -> None:
        match dtype:
            case "date":
                val = value or "1/1/2000"
                self.value = pd.to_datetime(val)
            case "int":
                self.value = int(value) or 0
            case "float":
                self.value = float(value) or 0.0

    def fit(self, data: pd.Series):
        self.col = data.name

        self.attr = Attribute(self.col, {})
        return self.attr

    def transform(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=data.index, name=self.col)

import logging
from typing import Collection, Dict, Optional

import pandas as pd

from ..metadata import Metadata
from ..attribute import Attribute
from .base import (
    RefTransformer,
    Transformer,
    DatetimeTransformer,
    DateTransformer,
    TimeTransformer,
    FixedValueTransformer,
    IdxTransformer,
    OrdinalTransformer,
    NumericalTransformer,
)
from .encoding import EncodingTransformer

logger = logging.getLogger(__name__)

_TRANSFORMERS: dict[str, type[Transformer]] = {}


def register_transformers(*cls: type[Transformer]):
    for c in cls:
        _TRANSFORMERS[c.name] = c


# Register base transformers as its assumed the user needs them
register_transformers(
    DatetimeTransformer,
    DateTransformer,
    TimeTransformer,
    FixedValueTransformer,
    IdxTransformer,
    OrdinalTransformer,
    NumericalTransformer,
)


class ReferenceManager:
    """Manages the foreign relationships of a table"""

    def __init__(
        self,
        meta: Metadata,
        name: str,
    ) -> None:
        self.name = name
        self.meta = meta

    def find_parents(self, table: str):
        """Finds the reference cols that link a table to its parents and
        returns tuples with the name of that column, the name of the parent table
        and the name of the parent key column (usually the primary key)."""

        res = []

        meta = self.meta[table]
        for name, col in meta.cols.items():
            if col.is_id():
                if col.ref:
                    ref_table = col.ref.table
                    ref_col = col.ref.col
                    if not ref_col:
                        ref_col = self.meta[ref_table].primary_key
                    res.append((name, ref_table, ref_col))
        return res

    def get_id_cols(self, name: str, ref: bool = False):
        """Returns the id column names of the provided table. If ref is set to True,
        only the ids with a reference are returned."""
        meta = self.meta[name]
        return [
            n
            for n, col in meta.cols.items()
            if col.is_id() and n != meta.primary_key and (not ref or meta[n].ref)
        ]

    def get_table_data(self, name: str, table: pd.DataFrame):
        """Returns the data columns of a table."""
        return table.drop(columns=self.get_id_cols(name, False))

    def get_foreign_keys(self, name: str, table: pd.DataFrame):
        """Returns the id columns of a table with a foreign reference."""
        return table[self.get_id_cols(name, True)]

    def find_foreign_ids(self, name: str, tables: Dict[str, pd.DataFrame]):
        """Creates an id lookup table for the provided table.

        The id lookup table is composed of columns that have the foreign table name and
        contain its index key to join on."""
        ids = self.get_foreign_keys(name, tables[name])

        for col, table, foreign_col in self.find_parents(name):
            assert (
                foreign_col == self.meta[table].primary_key
            ), "Only referencing primary keys supported for now."

            ids = ids.rename(columns={col: table})
            foreign_ids = self.find_foreign_ids(table, tables)
            ids = ids[~pd.isna(ids[table])].join(foreign_ids, on=table, how="inner")

        return ids

    def table_has_reference(self):
        """Checks whether the table has a column that depends on another table's
        column for transformation."""
        name = self.name
        meta = self.meta[name]

        for col in meta.cols.values():
            if not col.is_id() and col.ref is not None and col.ref.table is not None:
                return True
        return False

    def find_ids(self, tables: Dict[str, pd.DataFrame]):
        return self.find_foreign_ids(self.name, tables)


class BaseTableTransformer:
    def __init__(
        self,
        ref: ReferenceManager,
        meta: Metadata,
        name: str,
    ) -> None:
        self.ref = ref
        self.name = name
        self.meta = meta

        self.transformers: Dict[str, Transformer] = {}
        self.fitted = False

    def fit(
        self,
        tables: Dict[str, pd.DataFrame] | pd.DataFrame,
        ids: pd.DataFrame | None = None,
    ):
        # Only load foreign keys if required by column
        ids = None

        meta = self.meta[self.name]
        if isinstance(tables, dict):
            table = tables[self.name]
        else:
            table = tables

        if self.ref.table_has_reference():
            if ids is None:
                ids = self.ref.find_foreign_ids(self.name, tables)

            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(ids)} rows without ids on table {self.name}. Dropping before fitting..."
                )
                table = table.loc[ids.index]

        if not meta.primary_key == table.index.name:
            assert (
                False
            ), "Properly formatted datasets should have their primary key as their index column"
            # table.reindex(meta.primary_key)

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            # Add foreign column if required
            ref_col = None
            if col.ref:
                f_table, f_col = col.ref.table, col.ref.col
                if f_table:
                    # Foreign column from another table
                    ref_col = ids.join(tables[f_table][f_col], on=f_table)[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            # Fit transformer
            if "main_param" in col.args:
                t = _TRANSFORMERS[col.type](col.args["main_param"], **col.args)
            else:
                t = _TRANSFORMERS[col.type](**col.args)

            if isinstance(t, RefTransformer) and ref_col is not None:
                t.fit(table[name], ref_col)
            else:
                t.fit(table[name])
            self.transformers[name] = t

        self.fitted = True

    def transform(
        self,
        tables: Dict[str, pd.DataFrame] | pd.DataFrame,
        ids: pd.DataFrame | None = None,
    ):
        assert self.fitted

        meta = self.meta[self.name]
        if isinstance(tables, dict):
            table = tables[self.name]
        else:
            table = tables

        # Only load foreign keys if required by column
        ids = None
        if self.ref.table_has_reference():
            if ids is None:
                ids = self.ref.find_foreign_ids(self.name, tables)
            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(ids)} rows without ids on table {self.name}. Dropping before transforming..."
                )
                table = table.loc[ids.index]

        tts = []
        for name, col in meta.cols.items():
            if col.is_id():
                continue

            # Add foreign column if required
            ref_col = None
            if col.ref:
                f_table, f_col = col.ref.table, col.ref.col
                if f_table:
                    # Foreign column from another table
                    ref_col = ids.join(tables[f_table][f_col], on=f_table)[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            t = self.transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.transform(table[name], ref_col)
            else:
                tt = t.transform(table[name])
            tts.append(tt)

        return pd.concat(tts, axis=1)

    def reverse(
        self,
        table: pd.DataFrame,
        ids: Optional[pd.DataFrame] = None,
        parent_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        # If there are no ids that reference a foreign table, the ids and
        # parent_table parameters can be set to None (ex. tabular data).
        assert self.fitted
        transformers = self.transformers

        meta = self.meta[self.name]

        # Process columns with no intra-table dependencies first
        tts = []
        for name, col in meta.cols.items():
            if col.is_id():
                continue
            if col.ref is not None and col.ref.table is None:
                # Intra-table dependencies
                continue

            ref_col = None
            if col.ref is not None:
                f_table, f_col = col.ref.table, col.ref.col
                assert (
                    f_table in parent_tables
                ), f"Attempted to reverse table {self.name} before reversing {f_table}, which is required by it."
                ref_col = ids.join(parent_tables[f_table][f_col], on=f_table)[f_col]

            t = transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.reverse(table, ref_col)
            else:
                tt = t.reverse(table)
            tts.append(tt)

        # Process columns with intra-table dependencies afterwards.
        # fix-me: assumes no nested dependencies within the same table
        parent_cols = pd.concat(tts, axis=1)
        for name, col in meta.cols.items():
            if col.is_id() or col.ref is None:
                # Columns with no dependencies have been processed
                continue
            if col.ref is not None and col.ref.table is not None:
                # Columns with inter-table dependencies have been processed
                continue

            ref_col = parent_cols[col.ref.col]
            t = transformers[name]
            if isinstance(t, RefTransformer):
                tt = t.reverse(table, ref_col)
            else:
                tt = t.reverse(table)

            # todo: make sure this solves inter-table dependencies
            parent_cols[name] = tt
            tts.append(tt)

        # Re-add ids
        # If an id references another table it will be merged from the ids
        # dataframe. Otherwise, it will be set to 0 (irrelevant to data synthesis).
        dec_table = pd.concat(tts, axis=1)
        for name, col in meta.cols.items():
            if not col.is_id() or name == meta.primary_key:
                continue

            if col.ref is not None:
                dec_table[name] = ids[col.ref.table]
            else:
                dec_table[name] = 0

        # Re-order columns to metadata based order
        cols = [key for key in meta.cols.keys() if key != meta.primary_key]
        dec_table = dec_table[cols]

        return dec_table

    def get_attributes(self) -> dict[str, Attribute]:
        """Returns information about the transformed columns and their hierarchical attributes."""
        assert self.fitted

        attrs = {}
        for t in self.transformers.values():
            attrs[t.attr.name] = t.attr
        return attrs

    def find_ids(self, tables: Dict[str, pd.DataFrame]):
        return self.ref.find_foreign_ids(self.name, tables)