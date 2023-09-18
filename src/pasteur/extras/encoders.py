from copy import copy
from typing import Any, cast

import numpy as np
import pandas as pd

from pasteur.attribute import Attribute
from pasteur.metadata import Metadata
from pasteur.utils import LazyPartition

from ..attribute import (
    Attribute,
    CatValue,
    NumValue,
    StratifiedNumValue,
    Grouping,
    get_dtype,
)
from ..encode import AttributeEncoder, PostprocessEncoder


class DiscretizationColumnTransformer:
    """Converts a numerical column into an ordinal one using histograms."""

    def fit(self, val: NumValue, data: pd.Series) -> CatValue:
        self.in_val = val
        assert data.name and data.name == val.name
        self.col = val.name
        self.col_cnt = val.name + "_cnt"

        assert val.bins is not None

        self.edges = val.bins
        self.centers = ((self.edges[:-1] + self.edges[1:]) / 2).astype(np.float32)

        b = val.bins
        names = [f"[{b[i]:.2f}, {b[i + 1]:.2f})" for i in range(len(b) - 1)]

        group = Grouping("ord", names)
        null = None
        if val.nullable:
            group = Grouping("cat", [None, group])
            null = [True, *[False for _ in range(len(b) - 1)]]

        self.val = StratifiedNumValue(self.col, self.col_cnt, group, null)
        self.nullable = val.nullable
        return self.val

    def encode(self, data: pd.Series) -> pd.DataFrame | pd.Series:
        ofs = int(self.nullable)
        dtype = get_dtype(len(self.centers) + 1)
        midx = len(self.centers) - 1  # clip digitize out of bounds values
        digits = (
            np.digitize(data, bins=self.edges).astype(dtype).clip(1, midx) - 1 + ofs
        )

        digits = pd.Series(digits, index=data.index, name=self.col)
        if ofs:
            digits[pd.isna(data)] = 0

        clip_min = digits.astype("float32").replace(
            {i + ofs: c for i, c in enumerate(self.edges)}
        )
        clip_max = digits.astype("float32").replace(
            {i + ofs - 1: c for i, c in enumerate(self.edges)}
        )
        c = self.edges
        clip_norm = digits.astype("float32").replace(
            {i + ofs: c[i + 1] - c[i] for i in range(len(self.edges) - 1)}
        )
        cont = ((data.clip(upper=clip_max) - clip_min) / clip_norm).rename(self.col_cnt)
        if self.nullable:
            cont[digits == 0] = np.nan
        else:
            assert not np.any(
                pd.isna(data)
            ), f"Found nullable data in the non-nullable column '{self.col}'"
        return pd.concat([digits, cont], axis=1)

    def decode(self, enc: pd.DataFrame) -> pd.Series:
        ofs = int(self.nullable)

        if self.col_cnt in enc:
            c = self.edges
            clip_norm = (
                enc[self.col]
                .astype("float32")
                .replace({i + ofs: c[i + 1] - c[i] for i in range(len(self.edges) - 1)})
            )
            new_col = (
                enc[self.col]
                .astype("float32")
                .replace({i + ofs: c for i, c in enumerate(self.edges)})
                + enc[self.col_cnt] * clip_norm
            )
            new_col = new_col.rename(self.col)
        else:
            new_col = (
                enc[self.col]
                .astype("float32")
                .replace({i + ofs: c for i, c in enumerate(self.centers)})
                .rename(self.col)
            )

        if self.nullable:
            new_col[pd.isna(enc[self.col])] = np.nan

        return new_col


class IdxEncoder(AttributeEncoder[Attribute]):
    name = "idx"

    def fit(self, attr: Attribute, data: pd.DataFrame):
        self.transformers: dict[str, DiscretizationColumnTransformer] = {}

        # FIXME: not out-of-core
        cols = []
        found_num = False
        for name, col_attr in attr.vals.items():
            if isinstance(col_attr, NumValue):
                found_num = True
                t = DiscretizationColumnTransformer()
                new_attr = t.fit(col_attr, data[name])

                if isinstance(new_attr, dict):
                    cols.extend(new_attr.values())
                else:
                    cols.append(new_attr)

                self.transformers[name] = t
            else:
                cols.append(col_attr)

        self.common_name = attr.common.name if attr.common else None

        assert not (
            found_num and attr.common and attr.common.get_domain(0) > 2
        ), "Only null supported as a common condition for now."

        self.attr = Attribute(attr.name, cols, attr.common)

    def get_metadata(self) -> dict[str | tuple[str, ...], Attribute]:
        return {self.attr.name: self.attr}

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(self.attr.vals) == 0:
            return pd.DataFrame(index=data.index)

        out_cols = []
        for name in self.attr.vals:
            t = self.transformers.get(name, None)
            if t:
                out_cols.append(t.encode(data[name]))
            else:
                out_cols.append(data[name])

        if self.common_name:
            out_cols.append(data[self.common_name])

        return pd.concat(out_cols, axis=1, copy=False, join="inner")

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = pd.DataFrame(index=enc.index)
        for n in self.attr.vals.keys():
            t = self.transformers.get(n, None)
            if t:
                dec[n] = t.decode(enc)
            else:
                dec[n] = enc[n]

        if self.common_name and self.common_name in enc:
            dec[self.common_name] = enc[self.common_name]

        return dec


class NumEncoder(AttributeEncoder[Attribute]):
    name = "num"

    def fit(self, attr: Attribute, data: pd.DataFrame):
        self.in_attr = attr

        cols = []
        common = attr.common

        skip_common = False
        if len(attr.vals) == 1:
            v = next(iter(attr.vals.values()))
            if isinstance(v, CatValue) and v.is_ordinal:
                skip_common = True

        if not skip_common and common:
            for i in range(common.get_domain(common.height)):
                cols.append(NumValue(f"{attr.name}_cmn_{i}", [0, 0.5, 1]))

        for name, col in attr.vals.items():
            if isinstance(col, NumValue):
                cols.append(col)
            elif isinstance(col, CatValue):
                if col.is_ordinal():
                    cols.append(
                        NumValue(name, np.array(list(range(col.get_domain(0)))))
                    )
                else:
                    # TODO: Fix common values
                    for i in range(col.get_domain(0)):
                        cols.append(NumValue(f"{name}_{i}", [0, 0.5, 1]))

        self.attr = Attribute(attr.name, cols)

    def get_metadata(self) -> dict[str | tuple[str, ...], Attribute]:
        return {self.attr.name: self.attr}

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        a = self.in_attr
        if len(a.vals) == 0:
            return pd.DataFrame(index=data.index)
        cols = []
        only_has_na = a.common and a.common.get_domain(a.common.height) == 1

        # Handle common values
        skip_common = False
        if len(a.vals) == 1:
            v = next(iter(a.vals.values()))
            if isinstance(v, CatValue) and v.is_ordinal:
                skip_common = True

        common = a.common
        if not skip_common and common:
            for i in range(common.get_domain(common.height)):
                cmn_col = pd.Series(
                    False, index=data.index, name=f"{a.name}_cmn_{i}", dtype=np.float32
                )

                for name, col in a.vals.items():
                    if isinstance(col, CatValue):
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
            elif isinstance(col, CatValue):
                # TODO add proper encodings other than one hot

                # Handle ordinal values
                if col.is_ordinal():
                    cols.append(data[name])
                else:
                    # One hot encode everything else
                    for i in range(col.get_domain(0)):
                        cols.append((data[name] == i).rename(f"{name}_{i}"))

        return pd.concat(cols, axis=1, copy=False, join="inner")

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert False, "Not Implemented"


class MareEncoder(IdxEncoder, PostprocessEncoder[Attribute]):
    name = "mare"

    def finalize(
        self,
        meta: dict[str, dict[tuple[str, ...] | str, Attribute]],
        tables: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        return super().finalize(meta, tables, ids)

    def undo(
        self,
        meta: dict[str, dict[tuple[str, ...] | str, Attribute]],
        data: dict[str, LazyPartition],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        return super().undo(meta, data)
