import numpy as np
import pandas as pd

from pasteur.attribute import (
    Attribute,
    Attributes,
    CatValue,
    Grouping,
    NumValue,
    StratifiedNumValue,
    get_dtype,
)
from pasteur.metadata import Metadata
from pasteur.utils import (
    LazyDataset,
    LazyFrame,
    LazyPartition,
    LazyChunk,
    to_chunked,
)

from ..encode import AttributeEncoder, PostprocessEncoder, ViewEncoder
from typing import TYPE_CHECKING
from typing import NamedTuple
from pasteur.attribute import CatValue, NumValue, SeqValue, Attributes

if TYPE_CHECKING:
    import pydantic
    from pandas import DataFrame


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

        self.val = StratifiedNumValue(
            self.col, self.col_cnt, group, null, ignore_nan=val.ignore_nan
        )
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
            new_col[enc[self.col] == 0] = np.nan

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

        self.attr = Attribute(
            attr.name,
            cols,
            common=attr.common,
            unroll=attr.unroll,
            along=attr.along,
            partition=attr.partition,
            seq_repeat=attr.seq_repeat,
        )

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

        if self.common_name and self.common_name not in self.attr.vals:
            if self.common_name in data:
                out_cols.append(data[self.common_name])
            else:
                # Reconstruct common value from provided values
                val = next(iter(self.attr.vals.values()))
                for c in out_cols:
                    if isinstance(c, pd.DataFrame) and val.name in c:
                        col = c[val.name]
                    elif isinstance(c, pd.Series) and c.name == val.name:
                        col = c
                    else:
                        continue
                    assert isinstance(val, CatValue)
                    out_cols.append(
                        pd.Series(
                            val.get_mapping(val.height - 1)[col], index=col.index
                        ).rename(self.common_name)
                    )
                    break

        return pd.concat(out_cols, axis=1, copy=False, join="inner")

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        dec = pd.DataFrame(index=enc.index)
        for n in self.attr.vals.keys():
            t = self.transformers.get(n, None)
            if t:
                dec[n] = t.decode(enc)
            else:
                dec[n] = enc[n]

        if (
            self.common_name
            and self.common_name in enc
            and self.common_name not in self.attr.vals
        ):
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


def get_relationships(
    ids: dict[str, LazyFrame],
):
    from collections import defaultdict

    full_relationships = defaultdict(list)

    # Find parents
    for name, table_ids in ids.items():
        for parent in table_ids.sample().columns:
            full_relationships[parent].append(name)

    # Trim leaf tables
    relationships = {}

    for name in list(full_relationships.keys()):
        tmp = list(full_relationships[name])
        for child in full_relationships[name]:
            for k in full_relationships[child]:
                if k in tmp:
                    tmp.remove(k)

        relationships[name] = tmp

    return relationships


def get_top_table(relationships: dict[str, list[str]]):
    seen = set()
    for r in relationships.values():
        seen.update(r)

    # Find table we should start with
    top_table = None
    for name in relationships:
        if name not in seen:
            assert top_table is None, f"Multiple top tables found: {top_table}, {name}"
            top_table = name

    assert top_table is not None, "No top table found"
    return top_table


def create_pydantic_model(
    relationships: dict[str, list[str]],
    attrs: dict[str, Attributes],
    ctx_attrs: dict[str, dict[str, Attributes]],
):
    from pydantic import create_model

    from typing import Literal

    from pasteur.attribute import CatValue, NumValue, SeqValue

    top_table = get_top_table(relationships)

    # Now, recurse and create pydantic model
    def to_pydantic(name):
        fields = {}

        table_attrs = list(attrs[name].items())

        for ctxs in ctx_attrs.values():
            table_attrs.extend(ctxs.get(name, {}).items())

        for k, v in table_attrs:
            afields = {}
            nullable = True
            for kc, c in v.vals.items():
                if kc.startswith(f"{k}_"):
                    kc = kc[len(k) + 1 :]
                if isinstance(c, CatValue):
                    vals = c.get_human_readable()
                    nullable &= str(vals[0]) == "None"
                    afields[kc] = (Literal[*vals], ...)
                elif isinstance(c, NumValue):
                    if c.nullable:
                        afields[kc] = (float | None, ...)
                    else:
                        nullable = False
                        afields[kc] = (float, ...)
                elif isinstance(c, SeqValue):
                    # Sequence values are restored from their order
                    nullable = False
                else:
                    assert False, f"Unknown attribute type: {type(c)}"
            if isinstance(k, tuple):
                k = "_".join(k)
            if len(afields) > 1:
                model = create_model(k, **afields)
                if nullable:
                    model = model | None
                fields[k] = (model, ...)
            elif len(afields) == 1:
                fields[k] = list(afields.values())[0]

        for child in relationships.get(name, []):
            fields[child] = (list[to_pydantic(child)], ...)

        return create_model(
            name,
            **fields,
        )

    return to_pydantic(top_table)


class CatMapping(NamedTuple):
    table: str | None
    name: str
    mapping: list


class NumMapping(NamedTuple):
    table: str | None
    name: str
    nullable: bool


class AttrMapping(NamedTuple):
    table: str | None
    name: str
    nullable: bool
    cols: dict[str, CatMapping | NumMapping]


class ChildMapping(NamedTuple):
    name: str
    seq: str | None
    child: "TableMapping"


TopLevelAttr = NumMapping | CatMapping | AttrMapping | ChildMapping
TableMapping = dict[str, TopLevelAttr]


def _create_table_mapping(
    table: str,
    relationships: dict[str, list[str]],
    attrs: dict[str, Attributes],
    ctx_attrs: dict[str, dict[str, Attributes]],
) -> tuple[TableMapping, str | None]:
    seq = None

    mapping: TableMapping = {}
    adjacent = {}
    adjacent[None] = attrs[table]
    for k, v in ctx_attrs.items():
        adjacent[k] = v.get(table, {})
    for adj, adj_attrs in adjacent.items():
        for attr_name, attr in adj_attrs.items():
            afields: dict[str, CatMapping | NumMapping] = {}
            nullable = True
            for k, c in attr.vals.items():
                kc = k
                if kc.startswith(f"{k}_"):
                    kc = kc[len(k) + 1 :]
                if isinstance(c, CatValue):
                    vals = c.get_human_readable()
                    nullable &= str(vals[0]) == "None"
                    afields[kc] = CatMapping(adj, k, vals)
                elif isinstance(c, NumValue):
                    nullable &= c.nullable
                    afields[kc] = NumMapping(adj, k, c.nullable)
                elif isinstance(c, SeqValue):
                    # Sequence values are restored from their order
                    seq = k
                    nullable = False
                else:
                    assert False, f"Unknown attribute type: {type(c)}"
            if isinstance(attr_name, tuple):
                attr_name = "_".join(attr_name)
            if len(afields) > 1:
                mapping[attr_name] = AttrMapping(adj, attr_name, nullable, afields)
            elif len(afields) == 1:
                mapping[attr_name] = list(afields.values())[0]

    if table in relationships:
        for child in relationships[table]:
            child_mapping, child_seq = _create_table_mapping(
                child, relationships, attrs, ctx_attrs
            )
            mapping[child] = ChildMapping(child, child_seq, child_mapping)

    return mapping, seq


def create_table_mapping(
    table: str,
    relationships: dict[str, list[str]],
    attrs: dict[str, Attributes],
    ctx_attrs: dict[str, dict[str, Attributes]],
) -> TableMapping:
    return _create_table_mapping(table, relationships, attrs, ctx_attrs)[0]


def process_entity(
    table: str,
    cid: int,
    mapping: TableMapping,
    tables: dict[str, "DataFrame"],
    ctx: dict[str, dict[str, "DataFrame"]],
    ids: dict[str, "DataFrame"],
) -> dict:
    result = {}
    stack = [(table, cid, mapping, result)]

    while stack:
        table, cid, mapping, current = stack.pop()
        cached = {}
        cached[None] = tables[table].loc[cid]
        for k, v in ctx.items():
            if table in v and v[table].columns.any():
                cached[k] = v[table].loc[cid]

        for key, value in mapping.items():
            if isinstance(value, NumMapping):
                current[key] = float(cached[value.table][value.name])
            elif isinstance(value, CatMapping):
                current[key] = value.mapping[int(cached[value.table][value.name])]
            elif isinstance(value, AttrMapping):
                current[key] = {}
                for subkey, subvalue in value.cols.items():
                    if isinstance(subvalue, NumMapping):
                        v = cached[subvalue.table][subvalue.name]
                        if value.nullable and pd.isna(v):
                            current[key] = None
                            break
                        current[key][subkey] = float(v)
                    elif isinstance(subvalue, CatMapping):
                        if value.nullable and not v:
                            current[key] = None
                            break
                        current[key][subkey] = subvalue.mapping[
                            int(cached[subvalue.table][subvalue.name])
                        ]
                    else:
                        assert False, f"Unexpected subvalue type: {type(subvalue)}"
            elif isinstance(value, ChildMapping):
                # Get children
                cids = ids[value.name].index[ids[value.name][table] == cid]

                # Sort if required
                if value.seq:
                    cids = tables[value.name][value.seq][cids].sort_values().index

                children = []
                for cid in cids:
                    child = {}
                    stack.append((value.name, cid, value.child, child))
                    children.append(child)
                current[key] = children
            else:
                assert False, f"Unexpected value type: {type(value)}"
    
    return result


class JsonEncoder(ViewEncoder["type[pydantic.BaseModel]"]):
    name = "json"

    def fit(
        self,
        attrs: dict[str, Attributes],
        tables: dict[str, LazyFrame],
        ctx_attrs: dict[str, dict[str, Attributes]],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        self.relationships = get_relationships(ids)
        self.attrs = attrs
        self.ctx_attrs = ctx_attrs

    @to_chunked
    def encode(
        self,
        tables: dict[str, LazyChunk],
        ctx: dict[str, dict[str, LazyChunk]],
        ids: dict[str, LazyChunk],
    ) -> dict[str, LazyPartition[str]]:
        raise NotImplementedError()

    def decode(
        self,
        data: dict[str, LazyDataset[str]],
    ) -> tuple[
        dict[str, LazyFrame],
        dict[str, dict[str, LazyFrame]],
        dict[str, LazyFrame],
    ]:
        raise NotImplementedError()

    def get_metadata(self) -> "type[pydantic.BaseModel]":
        return create_pydantic_model(self.relationships, self.attrs, self.ctx_attrs)


class MareEncoder(IdxEncoder, PostprocessEncoder[Attribute]):
    name = "mare"
