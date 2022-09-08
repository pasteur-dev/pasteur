import logging
from typing import Literal

import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype

from ..utils import find_subclasses
from .attribute import (
    Attribute,
    Attributes,
    CatAttribute,
    IdxColumn,
    LeafLevel,
    NodeLevel,
    NumAttribute,
    NumColumn,
    OrdAttribute,
    Column,
    Level,
    OrdColumn,
    get_dtype,
)

logger = logging.getLogger(__name__)

"""Package with base transformers. 

Contains transformers that convert raw data into 2 types (with name suffix):
    - numerical (num): floating point values (float32), including NaN
    - discrete (idx): integer values (uintX) with metadata that make them:
        - categorical: integer values from 0-N with columns with no relations
        - ordinal: integer values from 0-N where `k` val is closer to `k + 1` than other vals.
        - hierarchical: contains a hierarchy of ordinal and categorical values.
"""


class Transformer:
    name = "base"
    "The name of the transformer, with which it's looked up in the dictionary."

    deterministic = True
    "For a given output, the input is the same."
    lossless = True
    "The decoded output equals the input."
    stateful = False
    "Transformer fits variables."

    def __init__(self, **_):
        self.attr: Attribute = None

    def fit(self, data: pd.Series) -> Attribute:
        """Fits to the provided data"""
        return self.attr

    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def transform(self, data: pd.Series) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"


class RefTransformer(Transformer):
    """Reference Transformers use a reference column as an input to create their embeddings.

    They can be used to integrate constraints (and domain knowledge) into embeddings,
    in such a way that all embeddings produce valid solutions and learning is
    easier.

    For example, consider an end date embedding that references a start date.
    The embedding will form a stable histogram with much less entropy, based
    on the period length.
    In addition, provided that the embedding is forced to be positive, any value
    it takes will produce a valid solution."""

    def fit(
        self,
        data: pd.Series,
        ref: pd.Series | None = None,
    ) -> Attribute:
        pass

    def fit_transform(
        self, data: pd.Series, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        self.fit(data, ref)
        return self.transform(data, ref)

    def transform(self, data: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        """When reversing, the data column contains encoded data, whereas the ref
        column contains decoded/original data. Therefore, the referred columns have
        to be decoded first."""
        assert 0, "Unimplemented"


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

    def __init__(self, span: str = "year", nullable: bool = False, bins=128, **_):
        self.weeks53 = span == "year53"
        if self.weeks53:
            self.span = "year"
        else:
            # Since last week is trimmed, transform is not lossless
            self.lossless = span == "year"
            self.span = span

        self.nullable = nullable
        self.bins = bins

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
                        f"{col}_year": NumColumn(self.bins),
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
                        f"{col}_week": NumColumn(self.bins),
                        f"{col}_day": OrdColumn(days, na=self.nullable),
                    },
                    self.nullable,
                )
            case "day":
                self.attr = Attribute(
                    col,
                    {
                        f"{col}_day": NumColumn(self.bins),
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
                out = rf_dt.normalize() + pd.to_timedelta(
                    np.round(vals[f"{col}_day"]).clip(0) - rf_day + 1,
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
                hours.append(LeafLevel(hour))
            if span == "halfhour":
                hours.append(
                    NodeLevel(
                        "ord",
                        [LeafLevel(f"{hour:02d}:00"), LeafLevel(f"{hour:02d}:00")],
                    )
                )
            else:
                mins = []
                for min in range(60):
                    if span == "minute":
                        mins.append(LeafLevel(f"{hour:02d}:{min:02d}"))
                    if span == "halfminute":
                        mins.append(
                            NodeLevel(
                                "ord",
                                [
                                    LeafLevel(f"{hour:02d}:{min:02d}:00"),
                                    LeafLevel(f"{hour:02d}:{min:02d}:30"),
                                ],
                            )
                        )
                    if span == "second":
                        secs = []
                        for sec in range(60):
                            secs.append(LeafLevel(f"{hour:02d}:{min:02d}:{sec:02d}"))
                        mins.append(NodeLevel("ord", secs))

                hours.append(NodeLevel("ord", mins))
        lvl = NodeLevel("ord", hours)
        if self.nullable:
            lvl = NodeLevel("cat", [None, lvl])

        self.domain = lvl.size

        self.attr = Attribute(
            data.name, {f"{data.name}_time": IdxColumn(lvl)}, self.nullable
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
