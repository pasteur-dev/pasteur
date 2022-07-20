import pandas as pd
import numpy as np


class Transformer:
    in_type = None
    out_type = None

    deterministic = True
    lossless = True

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"


class BinTransformer(Transformer):
    """Splits a DataFrame of numerical data (float/int) into bins and outputs idx integers.

    Reversed output has a step effect due to discretization, but is deterministic."""

    in_type = "numerical"
    out_type = "ordinal"

    deterministic = True
    lossless = False
    stateful = True

    def __init__(self, bins):
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


class OneHotTransformer(Transformer):
    """Transforms a categorical array of any type (fixed num of values) into a set of one hot encoded arrays (suffixed with _i)

    The array with idx len(vals.unique()) becomes True when the transform encounters a value that's not in the fit data.
    This value becomes None/NAN after the reverse transform."""

    in_type = "categorical"
    out_type = "bin"

    deterministic = True
    lossless = True
    stateful = True

    def __init__(self, unknown_value):
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
