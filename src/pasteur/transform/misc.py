import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

from .base import Transformer

logger = logging.getLogger(__name__)


class DiscretizationTransformer(Transformer):
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

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        constraints = constraints or {}

        self.bins = {}
        for col in data:
            if col in constraints:
                c = constraints[col]
                assert c["type"] == "numerical"
                self.bins[col] = np.histogram_bin_edges(
                    data[col], bins=self.n_bins, range=(c["min"], c["max"])
                )
            else:
                self.bins[col] = np.histogram_bin_edges(data[col], bins=self.n_bins)

        constraints = {}
        for col in data:
            constraints[col] = {"type": "ordinal", "dom": self.n_bins}
        return constraints

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            digits = np.digitize(data[col], bins=self.bins[col]) - 1
            out[col] = pd.Series(digits, index=data.index)

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        for col, bin in self.bins.items():
            out[col] = bin[data[col].clip(0, len(bin) - 1)]

        return out


class BinTransformer(Transformer):
    """Converts an ordinal variable into a hierarchical binary encoding
    (standard integer to bin conversion)."""

    name = "bin"
    in_type = "ordinal"
    out_type = "bin"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False
    variable_domain = True

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        constraints = constraints or {}
        self.digits = {}
        self.domain = {}

        for col in data:
            if col in constraints:
                assert constraints[col]["type"] in ("ordinal", "categorical")
                domain = constraints[col]["dom"]
            else:
                domain = np.max(data[col]) + 1

            self.digits[col] = math.ceil(math.log2(domain))
            self.domain[col] = domain

        constraints = {}
        for col, digits in self.digits.items():
            for i in range(digits):
                constraints[f"{col}_{i}"] = {"type": "ordinal", "dom": 2}
        return constraints

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col].to_numpy()

            for i in range(self.digits[col]):
                bin_col = (n & (1 << i)) != 0
                out[f"{col}_{i}"] = pd.Series(bin_col, index=data.index)

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col, digits in self.digits.items():
            l = len(data[f"{col}_0"])
            n = np.zeros((l), dtype=np.int32)

            for i in range(digits):
                n |= data[f"{col}_{i}"].to_numpy() << i

            n = n.clip(0, self.domain[col] - 1)
            out[col] = pd.Series(n, index=data.index)

        return out

    def get_hierarchy(self, **_) -> dict[str, list[str]]:
        out = {}

        for col, digits in self.digits.items():
            out[col] = [f"{col}_{i}" for i in reversed(range(digits))]

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

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        self.vals = {}
        self.types = {}

        for col in data:
            self.vals[col] = data[col].unique()
            self.types[col] = data[col].dtype

        constraints = {}
        for col, vals in self.vals.items():
            for i in range(len(vals) + 1):
                constraints[f"{col}_{i}"] = {"type": "ordinal", "dom": 2}
        return constraints

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

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        constraints = constraints or {}
        self.digits = {}
        self.domain = {}

        for col in data:
            if col in constraints:
                assert constraints[col]["type"] in ("ordinal", "categorical")
                domain = constraints[col]["dom"]
            else:
                domain = np.max(data[col]) + 1

            self.digits[col] = math.ceil(math.log2(domain))
            self.domain[col] = domain

        constraints = {}
        for col, digits in self.digits.items():
            for i in range(digits):
                constraints[f"{col}_{i}"] = {"type": "ordinal", "dom": 2}
        return constraints

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

            n = n.clip(0, self.domain[col] - 1)

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
    variable_domain = True

    def __init__(self, base: int = 2, **_) -> None:
        self.base = base
        self.out_type = f"b{base}" if base != 2 else "bin"

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        self.digits = {}
        self.domain = {}
        self.types = {}
        constraints = constraints or {}

        for col in data:
            if col in constraints:
                assert constraints[col]["type"] in ("ordinal", "categorical")
                domain = constraints[col]["dom"]
            else:
                domain = np.max(data[col]) + 1

            self.domain[col] = domain
            self.digits[col] = math.ceil(math.log(domain) / math.log(self.base))
            self.types[col] = data[col].dtype

        constraints = {}
        for col, digits in self.digits.items():
            for i in range(digits):
                constraints[f"{col}_{i}"] = {"type": "ordinal", "dom": self.base}
        return constraints

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

            out[col] = out_col.clip(0, self.domain[col])

        return out


class FixedValueTransformer(Transformer):
    """The transform function of this transformer returns an empty dataframe and
    when reversing it returns the columns with a fixed value.

    Used for the anchoring date of a table."""

    name = "fix"
    in_type = ("ordinal", "categorical", "numerical")
    out_type = "ordinal"

    deterministic = True
    lossless = True
    stateful = True
    handles_na = False

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

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        assert constraints is None
        self.cols = list(data.keys())

        return {}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[[]]

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        for col in self.cols:
            out = out.assign(**{col: self.value})

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

    def fit(self, data: pd.DataFrame, constraints: dict[str, dict] | None = None):
        constraints = constraints or {}
        self.min = {}
        self.max = {}
        self.types = {}

        for col in data:
            if col in constraints:
                c = constraints[col]
                match c["type"]:
                    case "ordinal":
                        self.min[col] = 0
                        self.max[col] = c["dom"] - 1
                    case "numerical":
                        self.min[col] = c["min"]
                        self.max[col] = c["max"]
                    case other:
                        assert (
                            False
                        ), f"Type {other} of {col} not supported in normalize transformer."
            else:
                logger.warning(
                    f"Infering min, max values for column {col}. This violates DP."
                )
                self.min[col] = data[col].min(axis=0)
                self.max[col] = data[col].max(axis=0)

            self.types[col] = data[col].dtype

        return {
            name: {"type": "numerical", "min": 0, "max": 1}
            for name in self.types.keys()
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        for col in data:
            n = data[col]
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

    def __init__(self, max_std=10, **_):
        self.max_std = max_std

    def fit(
        self,
        data: pd.DataFrame,
        constraints: dict[str, dict] | None = None,
    ) -> dict[str, dict] | None:
        self.std = {}
        self.mean = {}
        self.types = {}

        for col in data:
            self.std[col] = data[col].mean()
            self.mean[col] = data[col].std()
            self.types[col] = data[col].dtype

        return {
            name: {"type": "numerical", "min": -self.max_std, "max": self.max_std}
            for name in self.types.keys()
        }

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        for col in data:
            n = data[col]
            std = self.std[col]
            mean = self.mean[col]

            out[col] = (n - mean) / std

        return out

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        for col in self.std:
            n = data[col].to_numpy().clip(-self.max_std, self.max_std)
            std = self.std[col]
            mean = self.mean[col]

            out[col] = (n * std + mean).astype(self.types[col])

        return out
