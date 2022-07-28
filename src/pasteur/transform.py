from itertools import chain
from typing import Dict, List
import pandas as pd
import numpy as np
import math
import logging
from .utils import find_subclasses

logger = logging.getLogger(__name__)


class Transformer:
    name = "base"
    in_type = None
    out_type = None

    deterministic = True
    lossless = True
    stateful = False
    handles_na = False

    def __init__(self, **_):
        pass

    def fit(self, data: pd.DataFrame):
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def fit(self, data: pd.DataFrame, ref: pd.Series | None = None):
        pass

    def fit_transform(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        self.fit(data, ref)
        return self.transform(data, ref)

    def transform(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        """When reversing, the data column contains encoded data, whereas the ref
        column contains decoded/original data. Therefore, the referred columns have
        to be decoded first."""
        assert 0, "Unimplemented"


class ChainTransformer(RefTransformer):
    """Allows chain applying transformers together to a column.

    If nullable is set to true, null columns will be omitted when fitting the transformers.
    This is to prevent the na_value from biasing the transformers.

    When transforming, an NA bool column will be added and the NA value will be replaced
    by na_val before transforming."""

    name = "chain"
    in_type = None
    out_type = None

    deterministic = True
    lossless = True
    stateful = False
    handles_na = True

    @staticmethod
    def from_dict(transformer_names: List[str] | str, args: Dict):
        if isinstance(transformer_names, str):
            transformer_names = [transformer_names]

        tdict = TRANSFORMERS()
        if "main_param" in args and len(transformer_names) > 0:
            # Pass main_param from extended syntax to first transformer if it exists
            first = tdict[transformer_names[0]](args["main_param"], **args)
            other = [tdict[name](**args) for name in transformer_names[1:]]
            transformers = [first, *other]
        else:
            # Otherwise just create transformers normally
            transformers = [tdict[name](**args) for name in transformer_names]

        return ChainTransformer(transformers=transformers, **args)

    def __init__(
        self, transformers: List[Transformer], nullable=None, na_val=0, **_
    ) -> None:
        self.transformers = transformers
        if len(transformers):
            self.in_type = transformers[0].in_type
            self.out_type = transformers[-1].out_type

            # Check transformer chain has valid types
            out_type = transformers[0].out_type
            for t in self.transformers[1:]:
                assert (
                    out_type == t.in_type
                    if isinstance(t.in_type, str)
                    else (out_type in t.in_type)
                )
                out_type = t.out_type

        na_handled = any(t.handles_na for t in transformers)
        self.handle_na = not na_handled and nullable
        if self.handle_na:
            logger.warning(
                f"Nullable column doesn't have transformers for na vals, using <name>_NA col."
            )
        self.nullable = nullable
        self.na_val = na_val

        self.deterministic = all(t.deterministic for t in transformers)
        self.lossless = all(t.lossless for t in transformers)

    def fit(self, data: pd.DataFrame, ref: pd.Series | None = None):
        if self.handle_na:
            assert (
                len(data.columns) == 1
            ), "Can only handle one column when checking for NA"
            self.na_col = f"{data.columns[0]}_na"
            na_col = np.any(data.isna(), axis=1)
            data = data[~na_col].infer_objects()
        else:
            # If not nullable and null value was passed halt.
            assert self.nullable or not np.any(
                data.isna()
            ), f"NA value in non nullable column {data.columns[0]}."

        for t in self.transformers:
            if isinstance(t, RefTransformer) and ref is not None:
                data = t.fit_transform(data, ref)
            else:
                data = t.fit_transform(data)

    def transform(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        if self.handle_na:
            na_col = np.any(data.isna(), axis=1)
            data = data.where(~na_col, other=self.na_val)
        else:
            # If not nullable and null value was passed halt.
            assert self.nullable or not np.any(
                data.isna()
            ), f"NA value in non nullable column {data.columns[0]}."

        for t in self.transformers:
            if isinstance(t, RefTransformer) and ref is not None:
                data = t.transform(data, ref)
            else:
                data = t.transform(data)

        if self.handle_na:
            data[self.na_col] = na_col
        return data

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        if self.handle_na:
            na_col = data[self.na_col]
            data = data.drop(columns=[self.na_col])

        for t in reversed(self.transformers):
            if isinstance(t, RefTransformer) and ref is not None:
                data = t.reverse(data, ref)
            else:
                data = t.reverse(data)

        if self.handle_na:
            data[na_col] = pd.NA
        return data


class BinTransformer(Transformer):
    """Splits a DataFrame of numerical data (float/int) into bins and outputs idx integers.

    Reversed output has a step effect due to discretization, but is deterministic."""

    name = "discrete"
    in_type = "numerical"
    out_type = "ordinal"

    deterministic = True
    lossless = False
    stateful = True
    handles_na = False

    def __init__(self, bins=32, **_):
        self.n_bins = bins

    def fit(self, data: pd.DataFrame):
        self.bins = {}

        for col in data:
            self.bins[col] = np.histogram_bin_edges(data[col], bins=self.n_bins)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            out[col] = np.digitize(data[col], bins=self.bins[col]) - 1

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, bin in self.bins.items():
            out[col] = bin[data[col]]

        return out


class IdxTransformer(Transformer):
    """Transforms categorical values of any type into integer based values.

    If the values are sortable, they will have adjacent integer values"""

    name = "idx"
    in_type = ("categorical", "ordinal")
    out_type = "ordinal"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = True

    def __init__(self, unknown_value=-1, **_):
        self.unknown_value = unknown_value

    def fit(self, data: pd.DataFrame):
        self.vals = {}
        self.mapping = {}
        self.types = {}

        for col in data:
            vals = [v for v in data[col].unique() if not pd.isna(v)]

            # Try to sort vals
            try:
                vals = sorted(vals)
            except:
                pass

            vals = list(vals)
            self.mapping[col] = {val: i for i, val in enumerate(vals)}
            self.vals[col] = {
                i: val for i, val in enumerate(vals + [self.unknown_value])
            }

            self.types[col] = data[col].dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, vals in data.items():
            mapping = self.mapping[col]
            out_col = vals.replace(mapping)
            ukn_val = len(mapping)
            na_val = len(mapping) + 1

            # Handle categorical columns without blowing them up to full blown columns
            if vals.dtype.name == "category":
                out_col = out_col.cat.add_categories(ukn_val)

            # Handle NAs correctly
            if np.any(vals.isna()):
                if vals.dtype.name == "category":
                    out_col = out_col.cat.add_categories(na_val)
                out_col = out_col.fillna(na_val)

            out_col = out_col.where(
                vals.isin(mapping.keys()) | vals.isna(), ukn_val
            ).astype("int16")
            out[col] = out_col

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in self.types:
            vals = data[col]
            na_val = len(self.mapping[col]) + 1
            out[col] = (
                vals.map(self.vals[col])
                .where(vals != na_val, pd.NA)
                .astype(self.types[col])
            )

        return out


class OneHotTransformer(Transformer):
    """Transforms a categorical array of any type (fixed num of values) into a set of one hot encoded arrays (suffixed with _i)

    The array with idx len(vals.unique()) becomes True when the transform encounters a value that's not in the fit data.
    This value becomes `unknown_value` after the reverse transform."""

    name = "onehot"
    in_type = "categorical"
    out_type = "bin"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = True

    def __init__(self, unknown_value=-1, **_):
        self.unknown_value = unknown_value

    def fit(self, data: pd.DataFrame):
        self.vals = {}
        self.types = {}

        for col in data:
            self.vals[col] = data[col].unique()
            self.types[col] = data[col].dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            null_col = np.ones((len(data))).astype(np.bool_)
            vals = self.vals[col]
            for i, val in enumerate(vals):
                if pd.isna(val):
                    out[f"{col}_{i}"] = pd.isna(data[col])
                else:
                    out[f"{col}_{i}"] = data[col] == val
                null_col[out[f"{col}_{i}"]] = False

            out[f"{col}_{i + 1}"] = null_col

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, vals in self.vals.items():
            type = self.types[col]
            l = len(data[f"{col}_0"])

            out_col = pd.Series(
                np.empty((l), dtype=type if type.name != "category" else "object"),
                index=data.index,
            )

            for i in range(len(vals)):
                out_col[data[f"{col}_{i}"]] = vals[i]

            out_col[data[f"{col}_{len(vals)}"]] = self.unknown_value

            if type.name == "category":
                out_col = out_col.astype("category")

            out[col] = out_col

        return out


class GrayTransformer(Transformer):
    """Converts an ordinal variable into a gray encoding."""

    name = "gray"
    in_type = "ordinal"
    out_type = "bin"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def fit(self, data: pd.DataFrame):
        self.digits = {}

        for col in data:
            self.digits[col] = math.ceil(math.log2(np.max(data[col]) + 1))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col].to_numpy()
            gray = n ^ (n >> 1)

            for i in range(self.digits[col]):
                bin_col = (gray & (1 << i)) != 0
                out[f"{col}_{i}"] = pd.Series(bin_col, index=data.index)

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, digits in self.digits.items():
            l = len(data[f"{col}_0"])
            gray = np.zeros((l), dtype=np.int32)

            for i in range(digits):
                gray |= data[f"{col}_{i}"].to_numpy() << i

            n = gray
            n = n ^ (n >> 1)
            n = n ^ (n >> 2)
            n = n ^ (n >> 4)
            n = n ^ (n >> 8)
            n = n ^ (n >> 16)

            out[col] = pd.Series(n, index=data.index)

        return out


class BaseNTransformer(Transformer):
    """Converts an ordinal integer based value into a fixed base-n encoding."""

    name = "basen"
    in_type = "ordinal"
    out_type = "basen"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def __init__(self, base: int = 2, **_) -> None:
        self.base = base
        self.out_type = f"b{base}" if base != 2 else "bin"

    def fit(self, data: pd.DataFrame):
        self.digits = {}
        self.types = {}

        for col in data:
            self.digits[col] = math.ceil(
                math.log(np.max(data[col]) + 1) / math.log(self.base)
            )
            self.types[col] = data[col].dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col].to_numpy().copy()

            for i in range(self.digits[col]):
                out[f"{col}_{i}"] = n % self.base
                n //= self.base

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, digits in self.digits.items():
            l = len(data[f"{col}_0"])
            out_col = np.zeros((l), dtype=self.types[col])

            for i in range(digits):
                out_col += data[f"{col}_{i}"].to_numpy() * (self.base**i)

            out[col] = out_col

        return out


class NormalizeTransformer(Transformer):
    """Normalizes numerical columns to (0, 1).

    The max, min values are chosen when calling fit(), if a larger value appears
    during transform it is clipped to (0, 1)."""

    name = "normalize"
    in_type = ("numerical", "ordinal")
    out_type = "numerical"

    deterministic = True
    lossless = False
    stateful = True
    handles_na = False

    def fit(self, data: pd.DataFrame):
        self.min = {}
        self.max = {}
        self.types = {}

        for col in data:
            self.min[col] = data[col].min(axis=0)
            self.max[col] = data[col].max(axis=0)
            self.types[col] = data[col].dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col].to_numpy().copy()
            n_min = self.min[col]
            n_max = self.max[col]

            out[col] = ((n - n_min) / (n_max - n_min)).clip(0, 1)

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in self.min:
            n = data[col]
            n_min = self.min[col]
            n_max = self.max[col]

            out[col] = (n_min + (n_max - n_min) * n).astype(self.types[col])

        return out


class NormalDistTransformer(Transformer):
    """Normalizes column to std 1, mean 0 on a normal distribution."""

    name = "normdist"
    in_type = ("numerical", "ordinal")
    out_type = "numerical"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def fit(self, data: pd.DataFrame):
        self.std = {}
        self.mean = {}
        self.types = {}

        for col in data:
            self.std[col] = data[col].mean()
            self.mean[col] = data[col].std()
            self.types[col] = data[col].dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col].to_numpy()
            std = self.std[col]
            mean = self.mean[col]

            out[col] = (n - mean) / std

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in self.std:
            n = data[col].to_numpy()
            std = self.std[col]
            mean = self.mean[col]

            out[col] = (n * std + mean).astype(self.types[col])

        return out


class DateTransformer(RefTransformer):

    name = "date"
    in_type = ("date", "datetime")
    out_type = "ordinal"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def __init__(self, span: str = "year", **_):
        self.span = span

    def fit(self, data: pd.DataFrame, ref: pd.Series | None = None):
        self.types = {}
        self.ref = {}
        for col, vals in data.items():
            self.types[col] = vals.dtype
            if ref is None:
                self.ref[col] = vals.min()

    @staticmethod
    def iso_year_start(iso_year):
        "The gregorian calendar date of the first day of the given ISO year"
        # Based on https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar
        fourth_jan = pd.to_datetime(
            pd.DataFrame({"year": iso_year, "month": 1, "day": 4})
        )

        delta = pd.to_timedelta(fourth_jan.dt.day_of_week, unit="day")
        return fourth_jan - delta

    @staticmethod
    def iso_to_gregorian(iso_year, iso_week, iso_day):
        "Gregorian calendar date for the given ISO year, week and day"
        year_start = DateTransformer.iso_year_start(iso_year)
        return year_start + pd.to_timedelta((iso_week - 1) * 7 + iso_day, unit="day")

    def transform(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, vals in data.items():
            rf = self.ref.get(col, ref)
            assert rf is not None
            # When using a ref column accessing the date parameters is done by the dt member.
            # When self referencing to the minimum value, its type is a Timestamp
            # which doesn't have the dt member and requires direct access.
            rf_dt = rf if isinstance(rf, pd.Timestamp) else rf.dt

            match self.span:
                case "year":
                    out[f"{col}_year"] = vals.dt.year - rf_dt.year
                    out[f"{col}_week"] = vals.dt.week
                    out[f"{col}_day"] = vals.dt.day_of_week
                case "week":
                    out[f"{col}_week"] = (
                        (vals.dt.normalize() - rf_dt.normalize()).dt.days
                        + rf_dt.day_of_week
                    ) // 7
                    out[f"{col}_day"] = vals.dt.day_of_week
                case "day":
                    out[f"{col}_day"] = (
                        vals.dt.normalize() - rf_dt.normalize()
                    ).dt.days + rf_dt.day_of_week

        return out

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in self.types:
            rf = self.ref.get(col, ref)
            assert rf is not None
            # When using a ref column accessing the date parameters is done by the dt member.
            # When self referencing to the minimum value, its type is a Timestamp
            # which doesn't have the dt member and requires direct access.
            rf_dt = rf if isinstance(rf, pd.Timestamp) else rf.dt

            match self.span:
                case "year":
                    out[col] = self.iso_to_gregorian(
                        rf_dt.year + data[f"{col}_year"],
                        data[f"{col}_week"],
                        data[f"{col}_day"],
                    )
                case "week":
                    out[col] = rf + pd.to_timedelta(
                        data[f"{col}_week"] * 7
                        + data[f"{col}_day"]
                        - rf_dt.day_of_week,
                        unit="days",
                    )
                case "day":
                    out[col] = rf_dt.normalize() + pd.to_timedelta(
                        data[f"{col}_day"] - rf_dt.day_of_week, unit="days"
                    )

        return out


class TimeTransformer(Transformer):

    name = "time"
    in_type = ("time", "datetime")
    out_type = "ordinal"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def __init__(self, span: str = "minute", **_):
        self.span = span

    def fit(self, data: pd.DataFrame):
        self.types = {}
        for col, vals in data.items():
            self.types[col] = vals.dtype

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        span = self.span

        for col in self.types:
            date = data[col]
            out[f"{col}_hour"] = date.dt.hour
            if span == "halfhour":
                out[f"{col}_halfhour"] = date.dt.minute > 29
            if span in ("minute", "halfminute", "second"):
                out[f"{col}_min"] = date.dt.minute
            if span == "halfminute":
                out[f"{col}_halfmin"] = date.dt.second > 29
            if span == "second":
                out[f"{col}_sec"] = date.dt.second

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        span = self.span

        for col in self.types:
            hour = data[f"{col}_hour"]
            min = 0
            sec = 0
            if span == "halfhour":
                min = 30 * data[f"{col}_halfhour"]
            if span in ("minute", "halfminute", "second"):
                min = data[f"{col}_min"]
            if span == "halfminute":
                sec = 30 * data[f"{col}_halfmin"]
            if span == "second":
                sec = data[f"{col}_sec"]

            out[col] = pd.to_datetime(
                {
                    "year": 2000,
                    "month": 1,
                    "day": 1,
                    "hour": hour,
                    "minute": min,
                    "second": sec,
                }
            )

        return out


class DatetimeTransformer(RefTransformer):
    name = "datetime"
    in_type = ("time", "datetime")
    out_type = "ordinal"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

    def __init__(self, span="year.halfhour", **_):
        date_span, time_span = span.split(".")
        self.dt = DateTransformer(date_span)
        self.tt = TimeTransformer(time_span)

    def fit(self, data: pd.DataFrame, ref: pd.Series | None = None):
        self.types = {}
        for col, vals in data.items():
            self.types[col] = vals.dtype

        self.dt.fit(data, ref)
        self.tt.fit(data)

    def transform(
        self, data: pd.DataFrame, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        date_enc = self.dt.transform(data, ref)
        time_enc = self.tt.transform(data)
        return pd.concat([date_enc, time_enc], axis=1)

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        date_dec = self.dt.reverse(data, ref)
        time_dec = self.tt.reverse(data)

        out = pd.DataFrame()
        for col in self.types:
            out[col] = pd.to_datetime(
                {
                    "year": date_dec[col].dt.year,
                    "month": date_dec[col].dt.month,
                    "day": date_dec[col].dt.day,
                    "hour": time_dec[col].dt.hour,
                    "minute": time_dec[col].dt.minute,
                    "second": time_dec[col].dt.second,
                }
            )

        return out


TRANSFORMERS = lambda: find_subclasses(Transformer)
