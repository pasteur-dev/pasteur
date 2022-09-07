import pandas as pd
import numpy as np
from copy import copy

from ..utils import find_subclasses
from .attribute import (
    Attributes,
    Attribute,
    Column,
    IdxColumn,
    NodeLevel,
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

    def fit(self, attr: NumColumn, data: pd.Series) -> OrdColumn:
        self.in_attr = attr
        self.col = data.name

        rng = (
            (attr.min, attr.max)
            if attr.min is not None and attr.max is not None
            else None
        )
        self.edges = np.histogram_bin_edges(data, attr.bins, rng)
        self.vals = (self.edges[:-1] + self.edges[1:]) / 2

        self.attr = OrdColumn(self.vals, attr.na)
        return self.attr

    def encode(self, data: pd.Series) -> pd.DataFrame:
        ofs = 1 if self.in_attr.na else 0
        midx = len(self.vals) - 1  # clip digitize out of bounds values
        digits = (np.digitize(data, bins=self.edges) - 1).clip(0, midx) + ofs
        if self.in_attr.na:
            digits[pd.isna(data)] = 0
        return pd.Series(digits, index=data.index, name=self.col)

    def decode(self, enc: pd.DataFrame) -> pd.Series:
        ofs = 1 if self.in_attr.na else 0
        v = self.vals[(enc[self.col] - ofs).clip(0, len(self.vals) - 1)]
        if self.in_attr.na:
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
        out_cols = []
        for name, col in self.attr.cols.items():
            t = self.transformers.get(name, None)
            if t:
                out_cols.append(t.encode(data[name]))
            else:
                out_cols.append(data[name])

        return pd.concat(out_cols, axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = pd.DataFrame()
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
        ofs = 0
        if attr.na:
            cols[f"{attr.name}_na"] = NumColumn()
            ofs += 1
        if attr.ukn_val:
            cols[f"{attr.name}_ukn"] = NumColumn()
            ofs += 1

        for name, col in attr.cols.items():
            if col.type == "num":
                cols[name] = col
            if col.type == "idx":
                col: IdxColumn
                if col.lvl.type == "ord" and col.lvl.size == len(col.lvl):
                    cols[name] = NumColumn()
                elif (
                    col.lvl.type == "cat"
                    and len(col.lvl) == ofs + 1
                    and col.lvl[ofs].type == "ord"
                ):
                    cols[name] = NumColumn()
                else:
                    for i in range(col.lvl.size - ofs):
                        cols[f"{name}_{i}"] = NumColumn()

        self.attr = copy(attr)
        self.attr.update_cols(cols)
        return self.attr

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        a = self.in_attr
        cols = []

        # Add NA column
        if a.na:
            na_col = pd.Series(False, index=data.index, name=f"{a.name}_na")
            for name, col in a.cols.items():
                if col.type == "idx":
                    na_col |= data[name] == 0
                if col.type == "num":
                    na_col |= pd.isna(data[name])
            cols.append(na_col)

        # Add ukn val column
        if a.ukn_val:
            ukn_col = pd.Series(False, index=data.index, name=f"{a.name}_ukn")
            for name, col in a.cols.items():
                if col.type == "idx":
                    ukn_col |= data[name] == 1
            cols.append(ukn_col)

        # Add other columns
        for name, col in a.cols.items():
            if col.type == "num":
                cols.append(data[name])
            if col.type == "idx":
                # TODO add proper encodings other than one hot
                col: IdxColumn
                ofs = 0
                if a.na:
                    ofs += 1
                if a.ukn_val:
                    ofs += 1

                # Handle ordinal values
                encoded = False
                if len(col.lvl) == ofs + 1:
                    lvl = col.lvl[ofs]
                    # Level has to be ordinal and only include leafs (infered by size being equal to length)
                    if (
                        isinstance(lvl, NodeLevel)
                        and lvl.type == "ord"
                        and len(lvl) == lvl.size
                    ):
                        cols.append(data[name] - ofs)
                        encoded = True
                elif col.lvl.type == "ord" and col.lvl.size == len(col.lvl):
                    cols.append(data[name])
                    encoded = True

                # One hot encode everything else
                if not encoded:
                    for i in range(col.lvl.size - ofs):
                        cols.append((data[name] == i + ofs).rename(f"{name}_{i}"))

        return pd.concat(cols, axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert False, "Not Implemented"
