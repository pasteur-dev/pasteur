import pandas as pd
import numpy as np
from copy import copy

from .attribute import (
    Attributes,
    Attribute,
    Column as ColumnAttr,
    OrdAttribute,
    NumColumn,
    OrdColumn,
)


"""This package contains transformers that take the output from the `base` 
transformers and adjust it to fit certain models.

One example is a one-hot, numerical transformer that adjusts the data to be
suitable for a regression model.

Model specific transformers have their own hyper-parameters and may be considered
part of the model."""


class EncodingTransformer:
    """Receives tables that have been encoded by the base transformers and have
    attributes, and reformats them to fit a specific model."""

    name: str = None
    attrs: Attributes = None

    def fit(self, attrs: Attributes, data: pd.DataFrame) -> Attributes:
        assert 0, "Unimplemented"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"


class ColumnTransformer:
    """Encapsulates a special way to encode a column."""

    attr: ColumnAttr

    def fit(self, attr: ColumnAttr, data: pd.Series) -> Attribute:
        assert 0, "Unimplemented"

    def transform(self, data: pd.Series) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, enc: pd.DataFrame) -> pd.Series:
        assert 0, "Unimplemented"


class DiscretizationColumnTransformer:
    def fit(self, attr: NumColumn, data: pd.Series) -> Attribute:
        self.in_attr = attr
        self.col = data.name

        self.edges = np.histogram_bin_edges(data, attr.bins, (attr.min, attr.max))
        self.vals = (self.edges[:-1] + self.edges[1:]) / 2

        self.attr = OrdColumn(self.vals, attr.na)
        return self.attr

    def transform(self, data: pd.Series) -> pd.DataFrame:
        ofs = 1 if self.in_attr.na else 0
        midx = len(self.vals) - 1  # clip digitize out of bounds values
        digits = (np.digitize(data, bins=self.edges) - 1).clip(0, midx) + ofs
        if self.attr.na:
            digits[pd.isna(data)] = 0
        return pd.Series(digits, index=data.index)

    def reverse(self, enc: pd.DataFrame) -> pd.Series:
        ofs = 1 if self.in_attr.na else 0
        v = self.vals[(enc - ofs).clip(0, len(self.vals) - 1)]
        if self.attr.na:
            v[enc == 0] = np.nan
        return pd.Series(v, index=enc.index)


class IdentityColumnTransformer:
    def fit(self, attr: NumColumn, data: pd.Series) -> Attribute:
        self.attr = attr
        return self.attr

    def transform(self, data: pd.Series) -> pd.DataFrame:
        return data

    def reverse(self, enc: pd.DataFrame) -> pd.Series:
        return enc


class IdxEncodingTransformer(EncodingTransformer):
    def fit(self, attrs: Attributes, data: pd.DataFrame) -> Attributes:
        self.in_attrs = attrs
        self.attrs = {}
        self.transformers = {}

        for an, attr in attrs.items():
            cols = {}
            for cn, col in attr.cols.items():
                if isinstance(col, NumColumn):
                    t = DiscretizationColumnTransformer()
                else:
                    t = IdentityColumnTransformer()

                cattr = t.fit(col, data[cn])
                self.transformers[cn] = t
                cols[cn] = cattr
            nattr = copy(attr)
            nattr.cols = cols
            self.attrs[an] = nattr
