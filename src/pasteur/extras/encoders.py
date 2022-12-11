from copy import copy

import numpy as np
import pandas as pd

from ..attribute import Attribute, IdxValue, NumValue, OrdValue, get_dtype
from ..encode import Encoder


class DiscretizationColumnTransformer:
    """Converts a numerical column into an ordinal one using histograms."""

    def fit(self, attr: NumValue, data: pd.Series) -> IdxValue:
        self.in_attr = attr
        assert data.name
        self.col = data.name

        rng = (
            (attr.min, attr.max)
            if attr.min is not None and attr.max is not None
            else None
        )
        assert attr.bins is not None
        # FIXME: is not out of core
        self.edges = np.histogram_bin_edges(data[~pd.isna(data)], attr.bins, rng)
        self.vals = ((self.edges[:-1] + self.edges[1:]) / 2).astype(np.float32)

        if attr.common <= 1:
            self.attr = OrdValue(self.vals, na=attr.common == 1)
        else:
            assert (
                False
            ), "Discretizing multi-column attributes which contain multiple common values and numerical attributes is not supported for now"
            # The problem is that when there's one common value, ie NA, there's a floating point presentation for it
            # but for more there's not...
            # self.attr = LevelColumn(Level("ord", self.vals), attr.common)
        return self.attr

    def encode(self, data: pd.Series) -> pd.DataFrame | pd.Series:
        ofs = self.in_attr.common
        dtype = get_dtype(len(self.vals))
        midx = len(self.vals) - 1  # clip digitize out of bounds values
        digits = (np.digitize(data, bins=self.edges).astype(dtype) - 1).clip(0, midx) + ofs
        if ofs:
            digits[pd.isna(data)] = 0
        return pd.Series(digits, index=data.index, name=self.col)

    def decode(self, enc: pd.DataFrame) -> pd.Series:
        ofs = self.in_attr.common
        v = self.vals[(enc[self.col] - ofs).clip(0, len(self.vals) - 1)]
        if ofs:
            v[enc[self.col] == 0] = np.nan
        return pd.Series(v, index=enc.index, name=self.col)


class IdxEncoder(Encoder):
    name = "idx"

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        self.transformers: dict[str, DiscretizationColumnTransformer] = {}

        # FIXME: not out-of-core
        cols = {}
        for name, col_attr in attr.vals.items():
            if isinstance(col_attr, NumValue):
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
        self.attr.update_vals(cols)
        return self.attr

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(self.attr.vals) == 0:
            return pd.DataFrame(index=data.index)

        out_cols = []
        for name, col in self.attr.vals.items():
            t = self.transformers.get(name, None)
            if t:
                out_cols.append(t.encode(data[name]))
            else:
                out_cols.append(data[name])

        return pd.concat(out_cols, axis=1, copy=False, join='inner')

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = pd.DataFrame(index=enc.index)
        for n in self.attr.vals.keys():
            t = self.transformers.get(n, None)
            if t:
                dec[n] = t.decode(enc)
            else:
                dec[n] = enc[n]
        return dec


class NumEncoder(Encoder):
    name = "num"

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        self.in_attr = attr

        cols = {}

        common = attr.common

        skip_common = False
        if len(attr.vals) == 1:
            v = next(iter(attr.vals.values()))
            if isinstance(v, IdxValue) and v.is_ordinal:
                skip_common = True

        if not skip_common:
            for i in range(common):
                cols[f"{attr.name}_cmn_{i}"] = NumValue()

        for name, col in attr.vals.items():
            if isinstance(col, NumValue):
                cols[name] = col
            elif isinstance(col, IdxValue):
                if col.is_ordinal():
                    cols[name] = NumValue()
                else:
                    assert col.common == common
                    for i in range(col.get_domain(0) - col.common):
                        cols[f"{name}_{i}"] = NumValue()

        self.attr = copy(attr)
        self.attr.update_vals(cols)
        return self.attr

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        a = self.in_attr
        if len(a.vals) == 0:
            return pd.DataFrame(index=data.index)
        cols = []
        only_has_na = a.common == 1 and a.na

        # Handle common values
        skip_common = False
        if len(a.vals) == 1:
            v = next(iter(a.vals.values()))
            if isinstance(v, IdxValue) and v.is_ordinal:
                skip_common = True

        for i in range(a.common) if not skip_common else []:
            cmn_col = pd.Series(False, index=data.index, name=f"{a.name}_cmn_{i}", dtype=np.float32)

            for name, col in a.vals.items():
                if isinstance(col, IdxValue):
                    cmn_col += data[name] == i
                elif isinstance(col, NumValue) and only_has_na:
                    # Numerical values are expected to be NA for all common values
                    # so they are only used to set the common values when:
                    # `common == 1 and a.na`, meaning the only common value is NA.``
                    cmn_col += pd.isna(data[name])
            cols.append(cmn_col.clip(0, 1, inplace=False))

        # Add other columns
        for name, col in a.vals.items():
            if isinstance(col, NumValue):
                cols.append(data[name])
            elif isinstance(col, IdxValue):
                # TODO add proper encodings other than one hot

                # Handle ordinal values
                if col.is_ordinal():
                    cols.append(data[name])
                else:
                    # One hot encode everything else
                    for i in range(col.get_domain(0) - col.common):
                        cols.append(
                            (data[name] == i + col.common).rename(f"{name}_{i}")
                        )

        return pd.concat(cols, axis=1, copy=False, join='inner')

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert False, "Not Implemented"
