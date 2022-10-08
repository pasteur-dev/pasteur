import pandas as pd
import numpy as np
from copy import copy

from ..utils import find_subclasses
from .attribute import (
    Attributes,
    Attribute,
    IdxColumn,
    Level,
    LevelColumn,
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

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
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

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = []

        for t in self.transformers.values():
            cols.append(t.encode(data))

        return pd.concat(cols, axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        cols = []

        for t in self.transformers.values():
            cols.append(t.decode(enc))

        return pd.concat(cols, axis=1)

    @property
    def attrs(self):
        return {a.name: a for a in [t.attr for t in self.transformers.values()]}

    def get_attributes(self):
        return self.attrs


class DiscretizationColumnTransformer:
    """Converts a numerical column into an ordinal one using histograms."""

    def fit(self, attr: NumColumn, data: pd.Series) -> IdxColumn:
        self.in_attr = attr
        self.col = data.name

        rng = (
            (attr.min, attr.max)
            if attr.min is not None and attr.max is not None
            else None
        )
        self.edges = np.histogram_bin_edges(data[~pd.isna(data)], attr.bins, rng)
        self.vals = (self.edges[:-1] + self.edges[1:]) / 2

        if attr.common <= 1:
            self.attr = OrdColumn(self.vals, na=attr.common == 1)
        else:
            assert (
                False
            ), "Discretizing multi-column attributes which contain multiple common values and numerical attributes is not supported for now"
            # The problem is that when there's one common value, ie NA, there's a floating point presentation for it
            # but for more there's not...
            # self.attr = LevelColumn(Level("ord", self.vals), attr.common)
        return self.attr

    def encode(self, data: pd.Series) -> pd.DataFrame:
        ofs = self.in_attr.common
        midx = len(self.vals) - 1  # clip digitize out of bounds values
        digits = (np.digitize(data, bins=self.edges) - 1).clip(0, midx) + ofs
        if ofs:
            digits[pd.isna(data)] = 0
        return pd.Series(digits, index=data.index, name=self.col)

    def decode(self, enc: pd.DataFrame) -> pd.Series:
        ofs = self.in_attr.common
        v = self.vals[(enc[self.col] - ofs).clip(0, len(self.vals) - 1)]
        if ofs:
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

                if isinstance(new_attr, dict):
                    cols.update(new_attr)
                else:
                    cols[name] = new_attr

                self.transformers[name] = t
            else:
                cols[name] = col_attr

        self.attr = copy(attr)
        self.attr.update_cols(cols)
        return self.attr

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(self.attr.cols) == 0:
            return pd.DataFrame(index=data.index)

        out_cols = []
        for name, col in self.attr.cols.items():
            t = self.transformers.get(name, None)
            if t:
                out_cols.append(t.encode(data[name]))
            else:
                out_cols.append(data[name])

        return pd.concat(out_cols, axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = pd.DataFrame(index=enc.index)
        for n in self.attr.cols.keys():
            t = self.transformers.get(n, None)
            if t:
                dec[n] = t.decode(enc)
            else:
                dec[n] = enc[n]
        return dec


class NumAttributeTransformer(AttributeTransformer):
    name = "num"

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        self.in_attr = attr

        cols = {}

        common = attr.common
        for i in range(common):
            cols[f"{attr.name}_cmn_{i}"] = NumColumn()

        for name, col in attr.cols.items():
            if isinstance(col, NumColumn):
                cols[name] = col
            if isinstance(col, IdxColumn):
                if col.is_ordinal():
                    cols[name] = NumColumn()
                else:
                    assert col.common == common
                    for i in range(col.get_domain(0) - col.common):
                        cols[f"{name}_{i}"] = NumColumn()

        self.attr = copy(attr)
        self.attr.update_cols(cols)
        return self.attr

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        a = self.in_attr
        if len(a.cols) == 0:
            return pd.DataFrame(index=data.index)
        cols = []

        only_has_na = a.common == 1 and a.na

        # Handle common values
        for i in range(a.common):
            cmn_col = pd.Series(False, index=data.index, name=f"{a.name}_cmn_{i}")

            for name, col in a.cols.items():
                if isinstance(col, IdxColumn):
                    cmn_col |= data[name] == i
                if isinstance(col, NumColumn) and only_has_na:
                    # Numerical values are expected to be NA for all common values
                    # so they are only used to set the common values when:
                    # `common == 1 and a.na`, meaning the only common value is NA.``
                    cmn_col |= pd.isna(data[name])

        # Add other columns
        for name, col in a.cols.items():
            if isinstance(col, NumColumn):
                cols.append(data[name])
            elif isinstance(col, IdxColumn):
                # TODO add proper encodings other than one hot

                # Handle ordinal values
                if col.is_ordinal():
                    cols.append(data[name] - col.common)
                else:
                    # One hot encode everything else
                    for i in range(col.get_domain(0) - col.common):
                        cols.append(
                            (data[name] == i + col.common).rename(f"{name}_{i}")
                        )

        return pd.concat(cols, axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert False, "Not Implemented"
