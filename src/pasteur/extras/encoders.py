import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Type, cast

import numpy as np
import pandas as pd

from pasteur.attribute import (
    Attribute,
    Attributes,
    CatValue,
    DatasetAttributes,
    GenAttribute,
    Grouping,
    NumValue,
    SeqAttributes,
    SeqValue,
    StratifiedNumValue,
    get_dtype,
)
from pasteur.encode import AttributeEncoder, PostprocessEncoder, ViewEncoder
from pasteur.marginal.oracle import TableSelector
from pasteur.utils import (
    LazyChunk,
    LazyDataset,
    LazyFrame,
    get_relationships,
)

if TYPE_CHECKING:
    import numpy as np
    import pydantic
    from pandas import DataFrame

logger = logging.getLogger(__name__)

WORKER_MULTIPLIER = 4
MAX_IDX = 1000


class DiscretizationColumnTransformer:
    """Converts a numerical column into an ordinal one using histograms."""

    def fit(self, val: NumValue, data: pd.Series) -> CatValue:
        self.in_val = val
        assert data.name and data.name == val.name
        self.col = val.name
        self.col_cnt = val.name + "_cnt"

        assert val.bins is not None

        self.edges = val.bins
        self.centers = ((self.edges[:-1] + self.edges[1:]) / 2).astype(
            np.int32 if val.is_int else np.float32
        )

        b = val.bins
        names = [
            (
                f"[{b[i]:.0f}, {b[i + 1]:.0f})"
                if val.is_int
                else f"[{b[i]:.2f}, {b[i + 1]:.2f})"
            )
            for i in range(len(b) - 1)
        ]

        group = Grouping("ord", names)
        null = None
        if val.nullable:
            group = Grouping("cat", [None, group])
            null = [True, *[False for _ in range(len(b) - 1)]]

        self.val = StratifiedNumValue(
            self.col,
            self.col_cnt,
            group,
            null,
            ignore_nan=val.ignore_nan,
            is_int=val.is_int,
        )
        self.nullable = val.nullable
        return self.val

    def encode(self, data: pd.Series) -> pd.DataFrame | pd.Series:
        dtype = get_dtype(len(self.centers) + 1)
        if not np.all(pd.isna(data)):
            ofs = int(self.nullable)
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
            cont = ((data.clip(upper=clip_max) - clip_min) / clip_norm).rename(
                self.col_cnt
            )
            if self.nullable:
                cont[digits == 0] = np.nan
            else:
                assert not np.any(
                    pd.isna(data)
                ), f"Found nullable data in the non-nullable column '{self.col}'"
        else:
            assert (
                self.nullable
            ), f"Column {data.name} needs to be nullable to encode nullable columns with all NaN values"
            digits = pd.Series(0, index=data.index, name=self.col, dtype=dtype)
            cont = pd.Series(np.nan, index=data.index, name=self.col_cnt)

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


def get_top_table(relationships: dict[str, list[str]], ids: dict[str, Any] | None = None):
    if ids and len(ids) == 1:
        top_table = next(iter(ids))
        return top_table
    
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
    from typing import Literal

    from pydantic import create_model

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
                        afields[kc] = ((int if c.is_int else float) | None, ...)
                    else:
                        nullable = False
                        afields[kc] = ((int if c.is_int else float), ...)
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
        cached[None] = tables[table].loc[cid] if not tables[table].empty else None
        for k, v in ctx.items():
            if table in v and cid in v[table].index:
                cached[k] = v[table].loc[cid]

        for key, value in mapping.items():
            if isinstance(value, NumMapping):
                v = float(cached[value.table][value.name])
                # clip to .4f
                try:
                    current[key] = (int(v * 10000) / 10000) if v % 1 != 0 else int(v)
                except Exception:
                    current[key] = None
            elif isinstance(value, CatMapping):
                current[key] = value.mapping[int(cached[value.table][value.name])]
            elif isinstance(value, AttrMapping):
                if value.table not in cached:
                    current[key] = None
                    continue
                current[key] = {}
                for subkey, subvalue in value.cols.items():
                    assert subvalue.table in cached
                    if subkey.startswith(f"{key}_"):
                        subkey = subkey[len(key) + 1 :]

                    v = cached[subvalue.table][subvalue.name]
                    if isinstance(subvalue, NumMapping):
                        if value.nullable and pd.isna(v):
                            current[key] = None
                            break
                        current[key][subkey] = float(v) if v % 1 != 0 else int(v)
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


def _get_partition_keys(
    ids: dict[str, LazyChunk],
    relationships: dict[str, list[str]],
):
    top_table = get_top_table(relationships, ids)
    return sorted(set(ids[top_table]().index))


def _json_encode(
    tables: dict[str, LazyChunk],
    ctx: dict[str, dict[str, LazyChunk]],
    ids: dict[str, LazyChunk],
    relationships: dict[str, list[str]],
    attrs: dict[str, Attributes],
    ctx_attrs: dict[str, dict[str, Attributes]],
    nrange: "list[int] | np.ndarray",
):
    import pandas as pd

    from pasteur.utils.progress import piter

    l_tables = {k: v() for k, v in tables.items()}
    l_ctx = {k: {kk: vv() for kk, vv in v.items()} for k, v in ctx.items()}
    l_ids = {k: v() for k, v in ids.items()}

    top_table = get_top_table(relationships, ids)

    mapping = create_table_mapping(top_table, relationships, attrs, ctx_attrs)
    out = []

    for id in nrange:
        out.append(
            process_entity(
                top_table,
                id,
                mapping,
                l_tables,
                l_ctx,
                l_ids,
            )
        )

    if not len(nrange):
        return {}

    # We have to save the ids to be able to do look ups for the json
    pid = f"part_{nrange[0]}_{nrange[-1]}"
    return {
        "ids": {pid: pd.DataFrame(nrange)},
        "data": {pid: pd.DataFrame(map(str, out), index=nrange)},
    }


def _json_decode_entity(
    table: str,
    cid: int,
    obj: dict,
    mapping: TableMapping,
    out: dict[str | tuple[str, str], dict[int, dict]],
    out_ids: dict[str, dict[int, dict[str, int]]],
):
    stack = [(table, cid, mapping, obj)]

    while stack:
        table, cid, mapping, current = stack.pop()

        for key, v in mapping.items():
            if isinstance(v, NumMapping):
                lt = (table, v.table) if v.table else table
                out[lt][cid][v.name] = float(current[v.name] if current[key] is not None else 0)
            elif isinstance(v, CatMapping):
                lt = (table, v.table) if v.table else table
                out[lt][cid][v.name] = int(v.mapping.index(current[key]))
            elif isinstance(v, AttrMapping):
                for subkey, subvalue in v.cols.items():
                    if subkey.startswith(f"{v.name}_"):
                        subkey = subkey[len(v.name) + 1 :]

                    null = current[v.name] is None
                    if isinstance(subvalue, NumMapping):
                        lt = (table, subvalue.table) if subvalue.table else table
                        if null:
                            out[lt][cid][subvalue.name] = None
                        else:
                            # FIXME: figure out why gemma likes to make this None
                            rv = current[v.name][subkey]
                            out[lt][cid][subvalue.name] = float(
                                rv if rv is not None else 0
                            )
                    elif isinstance(subvalue, CatMapping):
                        lt = (table, subvalue.table) if subvalue.table else table
                        if null:
                            out[lt][cid][subvalue.name] = 0
                        else:
                            k = current[v.name][subkey]
                            out[lt][cid][subvalue.name] = (
                                int(subvalue.mapping.index(k)) if k is not None else 0
                            )
                    else:
                        assert False, f"Unexpected subvalue type: {type(subvalue)}"
            elif isinstance(v, ChildMapping):
                for i, chld in enumerate(current[v.name]):
                    ccid = cid * MAX_IDX + i
                    out_ids[v.name][ccid] = {
                        **out_ids[table].get(cid, {}),
                        table: int(cid),
                    }

                    # Add seq if required
                    if v.seq:
                        out[v.name][ccid][v.seq] = i

                    stack.append((v.name, ccid, v.child, chld))
            else:
                assert False, f"Unexpected value type: {type(v)}"


def _json_decode(
    ids: LazyChunk,
    data: LazyChunk,
    relationships: dict[str, list[str]],
    attrs: dict[str, Attributes],
    ctx_attrs: dict[str, dict[str, Attributes]],
):
    cids = ids()
    cdata = data()
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    out_ids = defaultdict(lambda: defaultdict(dict))

    model = create_pydantic_model(relationships, attrs, ctx_attrs)
    top_table = get_top_table(relationships)
    mapping = create_table_mapping(top_table, relationships, attrs, ctx_attrs)

    for cid, d in zip(cids.iterrows(), cdata.iterrows()):
        obj = eval(d[1][0])
        try:
            # Verify this is a valid object with pydantic
            model.parse_obj(obj)
            _json_decode_entity(top_table, int(cid[1][0]), obj, mapping, out, out_ids)  # type: ignore
        except Exception as e:
            logger.error(f"Error decoding entity {cid[1][0]}", exc_info=True)
            # Try to continue...

    # Convert to dataframes
    cctx = defaultdict(dict)
    cdata = {}

    for k, v in out.items():
        if isinstance(k, tuple):
            cctx[k[1]][k[0]] = pd.DataFrame.from_dict(v, orient="index")
        else:
            cdata[k] = pd.DataFrame.from_dict(v, orient="index")

    # Convert IDs to dataframes
    cids = {}
    table = get_top_table(relationships)

    stack = [table]
    while stack:
        t = stack.pop()
        for child in relationships.get(t, []):
            stack.append(child)

        if not out_ids.get(t, {}):
            cids[t] = pd.DataFrame(index=cdata[t].index)
        else:
            cids[t] = pd.DataFrame.from_dict(out_ids[t], orient="index")

    return cdata, dict(cctx), cids


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

    def encode(
        self,
        tables: dict[str, LazyFrame],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        import numpy as np

        from pasteur.utils import gen_closure
        from pasteur.utils.progress import get_max_workers, process, process_in_parallel

        if LazyDataset.are_partitioned(tables, ctx, ids):
            pids = []
            per_call = []

            for pid, vals in LazyDataset.zip(
                ids,
            ).items():  # type: ignore
                pids.append(pid)
                per_call.append({"ids": vals})

            key_sets = process_in_parallel(
                _get_partition_keys,
                per_call,
                base_args={"relationships": self.relationships},
                desc="Finding partition keys",
            )
            keys = {k: v for k, v in zip(pids, key_sets)}

            n_per_partition = get_max_workers() * WORKER_MULTIPLIER // len(pids)

            base_args = {
                "relationships": self.relationships,
                "attrs": self.attrs,
                "ctx_attrs": self.ctx_attrs,
            }
            per_call = []

            for pid, key_set in keys.items():
                key_list = sorted(key_set)
                pids = np.array_split(key_list, max(1, int(n_per_partition)))
                for p in pids:
                    per_call.append(
                        {
                            **LazyFrame.get_by_pid(pid,tables=tables,ctx=ctx,ids=ids),
                            "nrange": p,
                        }
                    )
        else:
            keys = process(
                _get_partition_keys,
                ids,  # type: ignore
                self.relationships,
            )

            # Partition into MAX_WORKERS*WORKER_MULTIPLIER chunks
            pids = list(
                np.array_split(sorted(keys), get_max_workers() * WORKER_MULTIPLIER)
            )

            logger.info(
                f"Processing {len(keys)} entities to JSON in {len(pids)} chunks"
            )
            base_args = {
                "tables": tables,
                "ctx": ctx,
                "ids": ids,
                "relationships": self.relationships,
                "attrs": self.attrs,
                "ctx_attrs": self.ctx_attrs,
            }
            per_call = []
            for pid_set in pids:
                per_call.append(
                    {
                        "nrange": pid_set,
                    }
                )

        return {gen_closure(_json_encode, **base_args, **p) for p in per_call}

    def decode(
        self,
        data: dict[str, LazyDataset[str]],
    ):
        from pasteur.utils import gen_closure

        return {
            gen_closure(
                _json_decode,
                d["ids"],
                d["data"],
                self.relationships,
                self.attrs,
                self.ctx_attrs,
            )
            for d in LazyDataset.zip_values(data)
        }

    def get_metadata(self):
        return {
            "attrs": self.attrs,
            "ctx_attrs": self.ctx_attrs,
            "relationships": self.relationships,
            "top_table": get_top_table(self.relationships),
        }


SEQ_MAX = 20


def _flatten_load(
    meta: Any,
    ids: Mapping[str, "DataFrame"],
    tables: Mapping[str, "DataFrame"],
    ctx: Mapping[str, Mapping[str, "DataFrame"]],
) -> dict[str, "DataFrame"]:
    top_table = meta["top_table"]
    parents = meta["parents"]
    out = tables[top_table].add_prefix(f"{top_table}_")

    for t in tables:
        if t == top_table:
            continue
        t_ids = ids[t][[top_table]]
        t_data = t_ids.join(tables[t]).groupby(top_table).first()
        if t not in parents or parents[t] == top_table:
            t_data["count"] = t_ids.groupby([top_table]).size().clip(0, SEQ_MAX - 1)
        else:
            # Keep the count of the first event only
            t_ids = ids[t][[top_table, parents[t]]]
            t_data["count"] = (
                t_ids.groupby([top_table, parents[t]])
                .size()
                .clip(0, SEQ_MAX - 1)
                .groupby([top_table])
                .first()
            )
        out = out.join(t_data.add_prefix(f"{t}_"), how="inner")

    for creator, ctx_tables in ctx.items():
        for t, table in ctx_tables.items():
            if t == top_table:
                t_data = table.add_prefix(f"{top_table}#{creator}_")
                out = out.join(t_data, how="inner")
            else:
                t_ids = ids[t][[top_table]]
                t_data = t_ids.join(table).groupby(top_table).first()
                out = out.join(t_data.add_prefix(f"{t}#{creator}_"), how="inner")

    return {"table": out}


def _flatten_meta(
    relationships: Mapping[str, list[str]],
    attrs: Mapping[tuple[str, ...] | str, Attribute],
    ctx_attrs: Mapping[str, Mapping[tuple[str, ...] | str, Attribute]],
):
    out = {}

    parents = {}

    for k, v in relationships.items():
        for child in v:
            parents[child] = k

    top_table = None
    for k in relationships:
        if k not in parents:
            top_table = k

    blacklist = []

    for table, table_attrs in attrs.items():
        is_top_table = True

        tmp = {}
        for name, attr in table_attrs.items():
            new_vals = [
                v.prefix_rename(f"{table}_")
                for v in attr.vals.values()
                if not isinstance(v, SeqValue)
            ]

            new_common = attr.common.prefix_rename(f"{table}_") if attr.common else None

            tmp[f"{table}_{name}"] = Attribute(f"{table}_{name}", new_vals, new_common)

            is_top_table = table not in parents

        if not is_top_table:
            out[f"{table}_count"] = GenAttribute(f"{table}_count", SEQ_MAX)

        # Add attrs after count to be clearer
        out.update(tmp)

        if is_top_table:
            assert (
                top_table is None or top_table == table
            ), f"Multiple top tables found: {top_table}, {table}"
            top_table = table

        for ctx, ctx_table_attrs in ctx_attrs.items():
            table_attrs = ctx_table_attrs.get(table, {})

            for name, attr in table_attrs.items():
                new_vals = [
                    v.prefix_rename(f"{table}#{ctx}_") for k, v in attr.vals.items()
                ]

                new_common = (
                    attr.common.prefix_rename(f"{table}#{ctx}_")
                    if attr.common
                    else None
                )

                out[f"{table}#{ctx}_{name}"] = Attribute(
                    f"{table}#{ctx}_{name}",
                    new_vals,
                    new_common,
                    attr.unroll,
                    attr.along,
                )

                blacklist.append(f"{ctx}_{name}")

    # Do not add the child values for context attrs
    # This removes the model's confusion if the first entry is always null
    for b in blacklist:
        if b in out:
            del out[b]

    return {
        "meta": out,
        "parents": parents,
        "top_table": top_table,
    }


class FlatEncoder(IdxEncoder, PostprocessEncoder[Attribute, Attributes]):
    name = "flat"

    def finalize(self, meta, ids, tables, ctx):
        return _flatten_load(meta, ids, tables, ctx)

    def undo(self, meta, data):
        raise NotImplementedError('"flat" is a one way transformation.')

    def get_post_metadata(self, relationships, attrs, ctx_attrs):
        return _flatten_meta(relationships, attrs, ctx_attrs)
