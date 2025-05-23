from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from pasteur.attribute import Attributes
from pasteur.transform import RefTransformer, Transformer

from ..attribute import (
    Attribute,
    CatAttribute,
    CommonValue,
    Grouping,
    NumAttribute,
    NumValue,
    OrdAttribute,
    OrdValue,
    StratifiedValue,
    get_dtype,
)
from ..transform import RefTransformer, SeqTransformer, Transformer
from ..utils import list_unique


class NumericalTransformer(Transformer):
    """Clips numerical values and attaches metadata to them."""

    name = "numerical"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self,
        bins: int = 20,
        min: float | int | None = None,
        max: float | int | None = None,
        nullable: bool = False,
        **_,
    ):
        self.nullable = nullable
        self.bins = bins
        self.min = min
        self.max = max

    def fit(self, data: pd.Series):
        self.col = cast(str, data.name)
        self.dtype = data.dtype
        if self.min is None:
            self.min = data.min()
        if self.max is None:
            self.max = data.max()
        self.attr = NumAttribute(self.col, self.bins, self.min, self.max, self.nullable)

    def reduce(self, other: "NumericalTransformer"):
        if self.min is not None and other.min is not None:
            self.min = min(self.min, other.min)
        elif other.min is not None:
            self.min = other.min

        if self.max is not None and other.max is not None:
            self.max = min(self.max, other.max)
        elif other.max is not None:
            self.max = other.max

        self.attr = NumAttribute(self.col, self.bins, self.min, self.max, self.nullable)

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

    def transform(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(pd.Series(data).clip(self.min, self.max).astype("float32"))

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        d = data[self.col].copy().clip(self.min, self.max)
        if self.dtype.name.lower().startswith("int"):
            d = d.round()
        return d.astype(self.dtype)


class IdxTransformer(Transformer):
    """Transforms categorical values of any type into integer based values.

    If the values are sortable, they will have adjacent integer values"""

    name = "categorical"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self,
        unknown_value=None,
        nullable: bool = False,
        partition: bool = False,
        **_,
    ):
        self.unknown_value = unknown_value
        self.nullable = nullable
        self.ordinal = False
        self.raw_vals = []
        self.partition = partition

    def fit(self, data: pd.Series):
        # Makes fit run out of core by storing the unique values seen previously in `raw_vals`
        new_vals = [v for v in data.unique() if not pd.isna(v)]
        self.raw_vals = list_unique(new_vals, self.raw_vals)

        ofs = 0
        if self.nullable:
            ofs += 1
        if self.unknown_value is not None:
            ofs += 1

        self.ofs = ofs
        self.col = cast(str, data.name)
        self.type = data.dtype
        self._finalize_props()

    def reduce(self, other: "IdxTransformer"):
        self.raw_vals = list_unique(self.raw_vals, other.raw_vals)
        self._finalize_props()

    def _finalize_props(self):
        # Try to sort vals
        vals = self.raw_vals
        try:
            vals = sorted(vals)
        except Exception:
            assert not self.ordinal, "Ordinal Array is not sortable"

        vals = list(vals)
        ofs = self.ofs
        self.mapping = {val: i + ofs for i, val in enumerate(vals)}
        self.vals = {i + ofs: val for i, val in enumerate(vals)}
        self.domain = ofs + len(vals)

        # FIXME: If a column is empty it causes problems for the algorithm
        # add 1 fake value as fix
        if not vals:
            vals = [7777777]

        cls = OrdAttribute if self.ordinal else CatAttribute
        self.attr = cls(
            self.col,
            vals,
            self.nullable,
            self.unknown_value,
            partition=self.partition,
        )

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

    def transform(self, data: pd.Series) -> pd.DataFrame:
        mapping = self.mapping
        type = get_dtype(self.domain)
        out_col = data.map(mapping)

        # Handle categorical columns without blowing them up to full blown columns
        if out_col.dtype == "category":
            out_col = out_col.cat.add_categories(range(self.ofs))

        # Handle NAs correctly
        if self.nullable:
            out_col = out_col.fillna(0)
        else:
            assert not np.any(
                data.isna()
            ), f"Not nullable '{self.col}' has nullable values"

        if self.unknown_value is not None:
            out_col = out_col.where(
                data.isin(mapping.keys()) | data.isna(), 1 if self.nullable else 0
            )
        else:
            assert np.all(
                data.isin(mapping.keys()) | data.isna()
            ), f"Uknown values found in '{self.col}', but no unknown value provided."

        # Remove old categories to change dtype
        if isinstance(out_col, pd.CategoricalDtype):
            out_col = out_col.cat.set_categories(range(self.domain))

        return pd.DataFrame(out_col.astype(type))

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        col = data.loc[:, self.col]
        if self.type.name == "category":
            out = col.astype(
                pd.CategoricalDtype(list(range(self.domain)))
            ).cat.rename_categories(self.vals)

            if self.nullable:
                out = out.where(col != 0, pd.NA).cat.remove_categories([0])
            if self.unknown_value is not None:
                out = (
                    out.cat.add_categories([self.unknown_value])
                    .where(col != (1 if self.nullable else 0), self.unknown_value)
                    .cat.remove_categories([1 if self.nullable else 0])
                )

            return out
        else:
            out = col.map(self.vals)

            if self.nullable:
                out = out.where(col != 0, pd.NA)
            if self.unknown_value is not None:
                out = out.where(col != (1 if self.nullable else 0), self.unknown_value)

            return out.astype(self.type)


class OrdinalTransformer(IdxTransformer):
    name = "ordinal"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordinal = True


class DateTransformer(RefTransformer):
    name = "date"
    deterministic = True
    lossless = True
    stateful = True

    def __init__(
        self,
        span: str = "year",
        nullable: bool = False,
        bins=64,
        max_len=63,
        ignore_nan=False,
        **_,
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
        self.ignore_nan = ignore_nan
        self.ref = None

    def fit(
        self,
        data: pd.Series,
        ref: pd.Series | None = None,
    ):
        if self.ref is None:
            self.ref = data.min()
        else:
            self.ref = min(data.min(), self.ref)
        self.col = cast(str, data.name)
        self._finalize_props()

    def reduce(self, other: "DateTransformer"):
        if self.ref is not None and other.ref is not None:
            self.ref = min(other.ref, self.ref)
        elif other.ref is not None:
            self.ref = other.ref
        self._finalize_props()

    def _finalize_props(self):
        col = self.col
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

        if self.nullable:
            common = CommonValue(
                f"{col}_common",
                True,
                ignore_nan=self.ignore_nan,
            )
        else:
            common = None

        match self.span:
            case "year":
                self.attr = Attribute(
                    col,
                    [
                        NumValue(
                            f"{col}_year",
                            self.bins,
                            self.nullable,
                            0,
                            self.max_len,
                            ignore_nan=self.ignore_nan,
                        ),
                        OrdValue(
                            f"{col}_week",
                            range(53 if self.weeks53 else 52),
                            na=self.nullable,
                            ignore_nan=self.ignore_nan,
                        ),
                        OrdValue(
                            f"{col}_day",
                            days,
                            na=self.nullable,
                            ignore_nan=self.ignore_nan,
                        ),
                    ],
                    common,
                )
            case "week":
                self.attr = Attribute(
                    col,
                    [
                        NumValue(
                            f"{col}_week",
                            self.bins,
                            self.nullable,
                            0,
                            self.max_len,
                            ignore_nan=self.ignore_nan,
                        ),
                        OrdValue(
                            f"{col}_day",
                            days,
                            na=self.nullable,
                            ignore_nan=self.ignore_nan,
                        ),
                    ],
                    common,
                )
            case "day":
                self.attr = Attribute(
                    col,
                    [
                        NumValue(
                            f"{col}_day", self.bins, self.nullable, 0, self.max_len, ignore_nan=self.ignore_nan
                        ),
                    ],
                )

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

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

        rf = ref.fillna(self.ref) if ref is not None else self.ref
        assert rf is not None
        # When using a ref column accessing the date parameters is done by the dt member.
        # When self referencing to the minimum value, its type is a Timestamp
        # which doesn't have the dt member and requires direct access.
        rf_dt = rf if isinstance(rf, pd.Timestamp) else cast(pd.Series, rf).dt

        iso = vals.dt.isocalendar()
        iso_rf = rf_dt.isocalendar()

        if isinstance(rf, pd.Timestamp):
            rf_year = rf_dt.year
            rf_day = iso_rf.weekday  # type: ignore
        else:
            rf_year = rf_dt.year
            rf_day = iso_rf["day"]  # type: ignore

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
            out[f"{col}_common"] = pd.Series(1, index=out.index, dtype="uint8")
            out = out.reindex(data.index, fill_value=0)
            # NAs were set as 0, change them to floats
            out.loc[na_mask, f"{col}_{self.span}"] = np.nan  # type: ignore

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
                na_mask = pd.isna(ref) | na_mask
                ref = ref[~na_mask]
            vals = vals[~na_mask.reindex(vals.index)]
            ofs = 1
        else:
            ofs = 0
            assert not np.any(pd.isna(vals[fcol])), "NAN values found on nonNAN col"

        rf = ref.fillna(self.ref) if ref is not None else self.ref
        assert rf is not None
        # When using a ref column accessing the date parameters is done by the dt member.
        # When self referencing to the minimum value, its type is a Timestamp
        # which doesn't have the dt member and requires direct access.
        if isinstance(rf, pd.Timestamp):
            rf_dt = rf
            rf_year = rf_dt.year
            iso_rf = rf.isocalendar()
            rf_day = iso_rf.weekday  # type: ignore
        else:
            rf_dt = cast(pd.Series, rf).dt
            rf_year = rf_dt.year
            iso_rf = rf_dt.isocalendar()
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
                    (
                        np.round(vals[f"{col}_week"]).astype("int32").clip(0) * 7
                        + (vals[f"{col}_day"] - ofs).clip(0, 6).astype("int32")
                        - rf_day
                        + 1
                    ).clip(0),
                    unit="days",
                )  # type: ignore
            case "day":
                # TODO: fix negative spans
                out = rf_dt.normalize() + pd.to_timedelta(
                    (np.round(vals[f"{col}_day"]) - rf_day + 1).astype("int32"),
                    unit="days",
                )
            case _:
                assert False, f"Unsupported span {self.span}"

        return out.reindex(data.index, fill_value=pd.NaT).rename(self.col)  # type: ignore


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
        self.col = cast(str, data.name)
        self._finalize_props()

    def _finalize_props(self):
        span = self.span

        hours = []
        for hour in range(24):
            if span == "hour":
                hours.append(f"{hour:02d}:00")
            elif span == "halfhour":
                hours.append(
                    Grouping(
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
                            Grouping(
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
                        mins.append(Grouping("ord", secs))

                hours.append(Grouping("ord", mins))
        lvl = Grouping("ord", hours)
        if self.nullable:
            lvl = Grouping("cat", [None, lvl])

        self.domain = lvl.size
        self.attr = Attribute(self.col, [StratifiedValue(f"{self.col}_time", lvl)])

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

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

        out = out.astype(get_dtype(self.domain))  # type: ignore
        return pd.DataFrame({f"{self.col}_time": out})

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
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
            case _:
                assert False

        out = pd.to_datetime(
            {
                "year": 2000,
                "month": 1,
                "day": 1,
                "hour": hour,
                "minute": min,
                "second": sec,
            }  # type: ignore
        )

        if self.nullable:
            out_data = out
            out = pd.Series(pd.NaT, index=data.index, name=col)
            out[~na_mask] = out_data  # type: ignore
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
    ):
        self.col = cast(str, data.name)

        self.dt.fit(data, ref)
        self.tt.fit(data)

        self._finalize_props()

    def reduce(self, other: "DatetimeTransformer"):
        self.dt.reduce(other.dt)
        self.tt.reduce(other.tt)
        self._finalize_props()

    def _finalize_props(self):
        cdt = next(iter(self.dt.get_attributes().values()))
        ctt = next(iter(self.tt.get_attributes().values()))

        if self.nullable:
            common = CommonValue(f"{self.col}_cmn", na=True, normal_name="Datetime")
        else:
            common = None

        self.attr = Attribute(
            self.col,
            common=common,
            vals=[*cdt.vals.values(), *ctt.vals.values()],
        )

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

    def transform(self, data: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
        date_enc = self.dt.transform(data, ref)
        time_enc = self.tt.transform(data)
        del data, ref
        if self.nullable:
            c = date_enc[next(iter(date_enc))]
            time_enc[pd.isna(c) if is_float_dtype(c) else c == 0] = 0
        return pd.concat([date_enc, time_enc], axis=1, copy=False, join="inner")

    def reverse(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame | pd.Series:
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
        self, dtype: Literal["date", "int", "float"] = "date", value: Any = None, **_
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
        self.attr = Attribute(cast(str, self.col), [])

    def get_attributes(self) -> Attributes:
        return {self.attr.name: self.attr}

    def transform(self, data: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)

    def reverse(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=data.index, name=self.col)
