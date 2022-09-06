import pandas as pd
import numpy as np
from copy import copy

from ..utils import find_subclasses
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


class AttributeTransformer:
    """Encapsulates a special way to encode a column."""

    name: str
    attr: Attribute

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        assert 0, "Unimplemented"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def reverse(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"


class EncodingTransformer:
    """Receives tables that have been encoded by the base transformers and have
    attributes, and reformats them to fit a specific model."""

    transformers: dict[str, AttributeTransformer]

    def __init__(self, transformer: str, **kwargs) -> None:
        self.kwargs = kwargs
        self.trn_cls = find_subclasses(AttributeTransformer)[transformer]
        self.transformers = {}

    def fit(self, attrs: Attributes, data: pd.DataFrame | None = None) -> Attributes:
        self.transformers = {}

        for n, a in attrs.items():
            t = self.trn_cls(**self.kwargs)
            t.fit(a, data)
            self.transformers[n] = t

        return self.attrs

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = []

        for t in self.transformers.values():
            cols.append(t.transform(data))

        return pd.concat(cols, axis=1)

    def reverse(self, enc: pd.DataFrame) -> pd.DataFrame:
        cols = []

        for t in self.transformers.values():
            cols.append(t.reverse(enc))

        return pd.concat(cols, axis=1)

    @property
    def attrs(self):
        return {a.name: a for a in [t.attr for t in self.transformers.values()]}


class DiscretizationColumnTransformer:
    """Converts a numerical column into an ordinal one using histograms."""

    def fit(self, attr: NumColumn, data: pd.Series) -> OrdColumn:
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
        v = self.vals[(enc[self.col] - ofs).clip(0, len(self.vals) - 1)]
        if self.attr.na:
            v[enc[self.col] == 0] = np.nan
        return pd.Series(v, index=enc.index, name=self.col)


class IdxAttributeTransformer(AttributeTransformer):
    name = "idx"

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        self.transformers: dict[str, DiscretizationColumnTransformer] = {}

        cols = {}
        for name, col_attr in attr.cols.items():
            if isinstance(col_attr, NumColumn):
                t = DiscretizationColumnTransformer()
                new_attr = t.fit(col_attr, data[name])
                cols.update(new_attr)
                self.transformers[name] = t
            else:
                cols[name] = col_attr

        self.attr = copy(attr)
        self.attr.update_cols(cols)
        return self.attr

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        out_cols = []
        for name, col in data.items():
            t = self.transformers.get(name, None)
            if t:
                out_cols.append(t.transform(col))
            else:
                out_cols.append(col)

        return pd.concat(out_cols, axis=1)

    def reverse(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = enc.copy()
        for n, t in self.transformers.items():
            dec[n] = t.reverse(enc)

        return dec


class MeasureIdxTransformer(IdxAttributeTransformer):
    pass


class MeasureNumTransformer(EncodingTransformer):
    pass
