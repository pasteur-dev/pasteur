import logging

import numpy as np
import pandas as pd

from ..utils import find_subclasses

logger = logging.getLogger(__name__)

"""Package with base transformers. 

Contains transformers that convert raw data into 4 types (with name suffix):
    - categorical: integer values from 0-N with columns with no relations
    - ordinal: integer values from 0-N where `k` val is closer to `k + 1` than other vals.
    - numerical: floating point values, assumed mean 0 and std 1.
    - hierarchical: Transformer returns a set of columns, some of which contain information
                    only in the context of other ones.

Assume columns c0 and c1 with a hierarchical relationship (ex. Hour, minute).
When sampling for parent relationships, c0 will be initially checked and
the pair c0, c1 will only be checked if c0 conveys enough mutual information on its own
(c1 only has meaning in the context of c0).

However, all cN will be sampled, since it is assumed that if they were included
the user expects them in the output data.
"""


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

    def get_hierarchy(self, **_) -> dict[str, list[str]]:
        return {}


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
    def from_dict(transformer_names: list[str] | str, args: dict):
        if isinstance(transformer_names, str):
            transformer_names = [transformer_names]

        tdict = find_subclasses(Transformer)
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
        self, transformers: list[Transformer], nullable=None, na_val=0, **_
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

    def get_hierarchy(self, **_) -> dict[str, list[str]]:
        out = {}
        for t in self.transformers:
            hier = t.get_hierarchy()
            new_hier = hier.copy()

            # Slice in the columns from the new transformer into the old one
            # If the new transformer creates new colums.
            # Example:
            # Old: a0: {c0, c1, c2}
            # New: c0: {b0, b1}
            # Combined: a0: {b0, b1, c1, c2}
            for attr in out:
                for new_attr, new_cols in hier.items():
                    if new_attr in out[attr]:
                        idx = out[attr].index(new_attr)
                        out[attr] = (
                            out[attr][:idx]
                            + new_cols
                            + (out[attr][idx + 1 :] if len(out[attr]) > idx else [])
                        )
                        
                        # Delete attribute that was merged
                        del new_hier[new_attr]

            out.update(new_hier)
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

    def __init__(self, unknown_value=-1, is_sortable=True, **_):
        self.unknown_value = unknown_value

        # If a categorical attribute is sortable it can become ordinal
        # otherwise switch output type to categorical.
        if not is_sortable:
            self.out_type = "categorical"

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

    def get_hierarchy(self, **_) -> dict[str, list[str]]:
        out = {}
        span = self.span

        for col in self.types:
            hier = [f"{col}_hour"]
            if span == "halfhour":
                hier += [f"{col}_halfhour"]
            if span in ("minute", "halfminute", "second"):
                hier += [f"{col}_min"]
            if span == "halfminute":
                hier += [f"{col}_halfmin"]
            if span == "second":
                hier += [f"{col}_sec"]

            out[col] = hier

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

    def get_hierarchy(self, **kwargs) -> dict[str, list[str]]:
        return self.tt.get_hierarchy(**kwargs)