from glob import escape
from typing import Dict, List
import pandas as pd
import numpy as np
import math


class Transformer:
    name = "base"
    in_type = None
    out_type = None

    deterministic = True
    lossless = True
    stateful = False

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


class ChainTransformer(Transformer):
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

    @staticmethod
    def from_dict(data: Dict):
        transformer_names = data["transformers"]

        tdict = TRANSFORMERS()
        transformers = [tdict[name](**data) for name in transformer_names]

        return ChainTransformer(transformers=transformers, **data)

    def __init__(
        self, transformers: List[Transformer], nullable=None, na_val=0, **_
    ) -> None:
        self.transformers = transformers
        self.in_type = transformers[0].in_type
        self.out_type = transformers[-1].out_type
        self.nullable = nullable
        self.na_val = na_val

        self.deterministic = all(t.deterministic for t in transformers)
        self.lossless = all(t.lossless for t in transformers)

    def fit(self, data: pd.DataFrame):
        if self.nullable:
            assert (
                len(data.columns) == 1
            ), "Can only handle one column when checking for NA"
            self.na_col = f"{data.columns[0]}_na"
            na_col = np.any(data.isna(), axis=1)
            data = data[~na_col].infer_objects()

        for t in self.transformers:
            data = t.fit_transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.nullable:
            na_col = np.any(data.isna(), axis=1)
            data = data.where(~na_col, other=self.na_val)

        for t in self.transformers:
            data = t.transform(data)

        if self.nullable:
            data[self.na_col] = na_col
        return data

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.nullable:
            na_col = data[self.na_col]
            data = data.drop(columns=[self.na_col])

        for t in reversed(self.transformers):
            data = t.reverse(data)

        if self.nullable:
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
    out_type = "basen"

    deterministic = True
    lossless = True
    stateful = True

    def __init__(self, unknown_value=-1, **_):
        self.unknown_value = unknown_value

    def fit(self, data: pd.DataFrame):
        self.vals = {}
        self.mapping = {}
        self.types = {}

        for col in data:
            vals = data[col].unique()

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

        for col in data:
            out[col] = (
                data[col]
                .map(self.mapping[col])
                .fillna(len(self.mapping[col]))
                .astype("int16")
            )

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, vals in self.vals.items():
            out[col] = data[col].map(vals).astype(self.types[col])

        return out


class OneHotTransformer(Transformer):
    """Transforms a categorical array of any type (fixed num of values) into a set of one hot encoded arrays (suffixed with _i)

    The array with idx len(vals.unique()) becomes True when the transform encounters a value that's not in the fit data.
    This value becomes None/NAN after the reverse transform."""

    name = "onehot"
    in_type = "categorical"
    out_type = "bin"

    deterministic = True
    lossless = True
    stateful = True

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
            l = len(data[f"{col}_0"])
            out_col = np.empty((l), dtype=self.types[col])

            for i in range(len(vals)):
                out_col[data[f"{col}_{i}"]] = vals[i]

            out_col[data[f"{col}_{len(vals)}"]] = self.unknown_value
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
                out[f"{col}_{i}"] = (gray & (1 << i)) != 0

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

            out[col] = n

        return out


class BaseNTransformer(Transformer):
    """Converts an ordinal integer based value into a fixed base-n encoding."""

    name = "basen"
    in_type = "ordinal"
    out_type = "basen"

    deterministic = True
    lossless = True
    stateful = True

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
    in_type = ("numerical", "basen")
    out_type = "numerical"

    deterministic = True
    lossless = False
    stateful = True

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
    in_type = "numerical"
    out_type = "numerical"

    deterministic = True
    lossless = True
    stateful = True

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


TRANSFORMERS = lambda: {t.name: t for t in Transformer.__subclasses__()}
