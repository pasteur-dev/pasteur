""" Contains the logic for handling multiple tables, and holding transformers and
encoders.

The functionality is achieved through a class named `ReferenceManager`, which
is used to generate the ID tables, and the `TransformHolder`, which holds
everything required to encode and transform a table.

The `TransformHolder` holds a `TableTransformer` which hosts the Table's transformers
and multiple `TableEncoder`s, which can be accesed with array syntax (ex. `['idx']`),
one for each supported encoding.

Once the TransformHolder is fit, it can be loaded and used to transform, encode,
reverse, and decode table partitions."""

import logging
from collections import defaultdict
from typing import Any, Callable, Generic, Mapping, TypeVar, cast

import pandas as pd

from .attribute import Attribute, Attributes
from .encode import (
    AttributeEncoder,
    AttributeEncoderFactory,
    EncoderFactory,
    ViewEncoder,
)
from .metadata import ColumnRef, Metadata
from .module import Module, get_module_dict
from .transform import RefTransformer, SeqTransformer, Transformer, TransformerFactory
from .utils import LazyChunk, LazyFrame, LazyPartition, lazy_load_tables, to_chunked
from .utils.progress import process_in_parallel, reduce

logger = logging.getLogger(__file__)

A = TypeVar("A", bound="Any")
META = TypeVar("META")
_IDKEY = "__ids_lkjhasndsfnewr"


def _reduce_inner(
    a: dict[str | tuple[str], A],
    b: dict[str | tuple[str], A],
):
    for key in a.keys():
        a[key].reduce(b[key])
    return a


class ReferenceManager:
    """Manages the foreign relationships of a table."""

    def __init__(
        self,
        meta: Metadata,
        name: str,
    ) -> None:
        self.name = name
        self.meta = meta

    def find_parents(self, table: str) -> list[tuple[str, str, str]]:
        """Finds the reference cols that link a table to its parents and
        returns tuples with the name of that column, the name of the parent table
        and the name of the parent key column (usually the primary key)."""

        res = []

        meta = self.meta[table]
        for name, col in meta.cols.items():
            if col.is_id():
                assert isinstance(name, str), "ids only require one column"
                if col.ref:
                    assert not isinstance(col.ref, list), "ids only have 1 reference"
                    ref_table = col.ref.table
                    assert ref_table
                    ref_col = col.ref.col
                    if not ref_col:
                        ref_col = self.meta[ref_table].primary_key
                    res.append((name, ref_table, ref_col))
        return res

    def get_id_cols(self, name: str, ref: bool = False):
        """Returns the id column names of the provided table. If ref is set to True,
        only the ids with a reference are returned."""
        meta = self.meta[name]
        return [
            n
            for n, col in meta.cols.items()
            if col.is_id() and n != meta.primary_key and (not ref or meta[n].ref)
        ]

    def get_table_data(self, name: str, table: pd.DataFrame):
        """Returns the data columns of a table."""
        return table.drop(columns=self.get_id_cols(name, False))

    def get_foreign_keys(self, name: str, table: pd.DataFrame):
        """Returns the id columns of a table with a foreign reference."""
        return table[self.get_id_cols(name, True)]

    def find_foreign_ids(self, name: str, get_table: Callable[[str], pd.DataFrame]):
        """Creates an id lookup table for the provided table.

        The id lookup table is composed of columns that have the foreign table name and
        contain its index key to join on."""
        ids = self.get_foreign_keys(name, get_table(name))

        for col, table, foreign_col in self.find_parents(name):
            assert (
                foreign_col == self.meta[table].primary_key
            ), "Only referencing primary keys supported for now."

            ids = ids.rename(columns={col: table})
            foreign_ids = self.find_foreign_ids(table, get_table)
            ids = ids[~pd.isna(ids[table])].join(foreign_ids, on=table, how="inner")

        return ids

    def table_has_reference(self):
        """Checks whether the table has a column that depends on another table's
        column for transformation."""
        name = self.name
        meta = self.meta[name]

        for col in meta.cols.values():
            if not col.is_id() and col.ref is not None:
                assert not isinstance(col.ref, list), "ids can only have one reference"
                assert (
                    col.ref.table is not None
                ), "ids with a reference should have one on a foreign table"
                return True
        return False


def _calc_joined_refs(
    name: str,
    get_table: Callable[[str], pd.DataFrame],
    ids: pd.DataFrame | None,
    cref: list[ColumnRef] | ColumnRef | None,
):
    """Returns a dataframe where for each row in the original data,
    reference values are provided matching the ones in cref.

    In the case of one reference, a series is returned.
    If no references are provided, returns None."""
    table = get_table(name)
    ref_col = None

    if cref and isinstance(cref, list):
        table_cols: dict[str | None, list[str]] = defaultdict(list)

        for ref in cref:
            assert ref.col is not None
            table_cols[ref.table].append(ref.col)

        dfs = []
        for rtable, refs in table_cols.items():
            if rtable:
                assert ids is not None
                df = ids.join(get_table(rtable)[refs], on=rtable)[refs].add_prefix(
                    f"{rtable}."
                )
                dfs.append(df)
            else:
                dfs.append(table[refs])

        ref_col = pd.concat(dfs)
    elif cref:
        ref = cast(ColumnRef, cref)
        f_table, f_col = ref.table, ref.col
        assert f_col
        if f_table:
            # Foreign column from another table
            assert ids is not None
            ref_col = ids.join(get_table(f_table)[f_col], on=f_table)[f_col]
        else:
            # Local column, duplicate and rename
            ref_col = table[f_col]

    return ref_col


def _calc_unjoined_refs(
    name: str,
    get_table: Callable[[str], pd.DataFrame],
    cref: list[ColumnRef] | ColumnRef | None,
):
    """Returns a dictionary containing columns from all upstream parents,
    as required by `cref`.

    If `cref` is None, None is returned."""
    table_cols: dict[str, list[str]] = defaultdict(list)

    if cref and isinstance(cref, list):
        for ref in cref:
            assert ref.col is not None
            table_cols[ref.table or name].append(ref.col)
    elif cref:
        ref = cast(ColumnRef, cref)
        assert ref.col is not None
        table_cols[ref.table or name].append(ref.col)

    return {k: get_table(name)[v] for k, v in table_cols.items()} or None


class TableTransformer:
    def __init__(
        self,
        meta: Metadata,
        name: str,
        modules: list[Module],
    ) -> None:
        self.name = name
        self.meta = meta

        self.transformer_cls = get_module_dict(TransformerFactory, modules)
        self.ref = ReferenceManager(meta, name)

        self.transformers: dict[str | tuple[str], Transformer] = {}
        self.fitted = False

    def fit(
        self,
        tables: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
    ):
        if ids is not None:
            tables = {_IDKEY: ids, **tables}

        per_call = []
        for _, chunks in LazyFrame.zip(tables).items():
            ids_chunk = chunks.pop(_IDKEY, None)
            per_call.append({"ids": ids_chunk, "tables": chunks})

        transformer_chunks: list[
            dict[str | tuple[str], Transformer]
        ] = process_in_parallel(
            self.fit_chunk,
            per_call,
            desc=f"Fitting transformers for table '{self.name}'",
        )

        self.transformers = reduce(_reduce_inner, transformer_chunks)
        self.fitted = True

    def fit_chunk(
        self,
        tables: dict[str, LazyChunk],
        ids: LazyChunk | None = None,
    ):
        get_table = lazy_load_tables(tables)  # type: ignore
        loaded_ids = self._load_ids(ids, get_table)
        meta = self.meta[self.name]
        table = get_table(self.name)

        transformers = {}

        if not meta.primary_key == table.index.name:
            assert (
                False
            ), "Properly formatted datasets should have their primary key as their index column"
            # table.reindex(meta.primary_key)

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            assert (
                col.type in self.transformer_cls
            ), f"Column type {col.type} not in transformers:\n{list(self.transformer_cls.keys())}"

            # Fit transformer
            if "main_param" in col.args:
                t = self.transformer_cls[col.type].build(
                    col.args["main_param"], **col.args
                )
            else:
                t = self.transformer_cls[col.type].build(**col.args)

            if isinstance(t, SeqTransformer):
                # Add foreign column if required
                ref_cols = _calc_unjoined_refs(self.name, get_table, col.ref)

                assert loaded_ids is not None
                t.fit(table[name], ref_cols, loaded_ids)
            elif isinstance(t, RefTransformer):
                # Add foreign column if required
                assert loaded_ids is not None
                ref_cols = _calc_joined_refs(self.name, get_table, loaded_ids, col.ref)
                t.fit(table[name], ref_cols)
            else:
                t.fit(table[name])
            transformers[name] = t

        return transformers

    def _load_ids(
        self,
        ids: LazyChunk | pd.DataFrame | None,
        get_table: Callable[[str], pd.DataFrame],
    ):
        """Loads ids only if required. If `ids` is None, it calculates them anew using
        `get_table` and the reference manager."""
        if not self.ref.table_has_reference():
            return None

        if callable(ids):
            return ids()

        return self.ref.find_foreign_ids(self.name, get_table)

    def transform_chunk(
        self,
        tables: dict[str, LazyChunk],
        ids: LazyChunk | None = None,
    ):
        assert self.fitted

        get_table = lazy_load_tables(tables)  # type: ignore
        loaded_ids = self._load_ids(ids, get_table)
        meta = self.meta[self.name]
        table = get_table(self.name)
        tts = []
        ctxs = defaultdict(list)

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            trn = self.transformers[name]
            if isinstance(trn, SeqTransformer):
                # Add foreign column if required
                ref_cols = _calc_unjoined_refs(self.name, get_table, col.ref)
                assert loaded_ids is not None
                tt, ctx = trn.transform(table[name], ref_cols, loaded_ids)

                for n, c in ctx.items():
                    ctxs[n].append(c)
            elif isinstance(trn, RefTransformer):
                # Add foreign column if required
                ref_cols = _calc_joined_refs(self.name, get_table, loaded_ids, col.ref)
                tt = trn.transform(table[name], ref_cols)
            else:
                tt = trn.transform(table[name])

            tts.append(tt)

        return pd.concat(tts, axis=1, copy=False, join="inner"), {
            n: pd.concat(c, axis=1, copy=False, join="inner") for n, c in ctxs.items()
        }, loaded_ids

    @to_chunked
    def _transform_chunk(
        self,
        tables: dict[str, LazyChunk],
        ids: LazyChunk | None = None,
    ):
        return self.transform_chunk(tables, ids)

    def transform(
        self,
        tables: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
    ):
        return self._transform_chunk(tables, ids)  # type: ignore

    @to_chunked
    def _reverse_chunk(
        self,
        data: LazyChunk,
        ctx: dict[str, LazyChunk],
        ids: LazyChunk | None = None,
        parent_tables: dict[str, LazyChunk] | None = None,
    ):
        # If there are no ids that reference a foreign table, the ids and
        # parent_table parameters can be set to None (ex. tabular data).
        assert self.fitted

        cached_table = data()
        cached_ids = ids() if ids is not None else pd.DataFrame()
        cached_ctx = {n: c() for n, c in ctx.items()}

        get_parent = (
            lazy_load_tables(parent_tables)
            if parent_tables
            else lambda _: pd.DataFrame()
        )

        meta = self.meta[self.name]

        # Add ids
        # If an id references another table it will be merged from the ids
        # dataframe. Otherwise, it will be set to 0 (irrelevant to data synthesis).
        tts = {}
        for name, col in meta.cols.items():
            if not col.is_id() or name == meta.primary_key:
                continue

            if col.ref is not None:
                assert (
                    not isinstance(col.ref, list)
                    and col.ref.table
                    and cached_ids is not None
                )
                tts[name] = cached_ids[col.ref.table]
            else:
                tts[name] = 0

        # Process columns using a for loop based topological sort
        completed_cols = set()
        processed_col = True
        while processed_col:
            processed_col = False

            for name, col in meta.cols.items():
                # Skip already processed columns
                if name in completed_cols:
                    continue
                # Skip ids
                if col.is_id():
                    continue

                # Check column requirements
                cref = col.ref
                if cref and isinstance(cref, list):
                    # Check ref requirements met
                    unmet_requirements = False
                    for ref in cref:
                        if not ref.table and not ref.col in tts:
                            unmet_requirements = True
                            break
                    if unmet_requirements:
                        continue
                elif cref:
                    ref = cast(ColumnRef, cref)
                    # Check ref requirements met
                    if not ref.table and ref.col not in completed_cols:
                        continue

                t = self.transformers[name]
                if isinstance(t, SeqTransformer):
                    ref_col = _calc_unjoined_refs(self.name, get_parent, cref)
                    tt = t.reverse(cached_table, cached_ctx, ref_col, cached_ids)
                elif isinstance(t, RefTransformer):
                    ref_col = _calc_joined_refs(self.name, get_parent, cached_ids, cref)
                    tt = t.reverse(cached_table, ref_col)
                else:
                    tt = t.reverse(cached_table)

                processed_col = True
                completed_cols.add(name)
                if isinstance(name, str):
                    tts[name] = tt
                else:
                    for n in name:
                        tts[n] = tt[n]

        decoded_cols = sum(
            len(n) if isinstance(n, tuple) else 1
            for n, c in meta.cols.items()
            if not c.is_id()
        )
        assert (
            len(tts) == decoded_cols
        ), f"Did not process column in this loop. There are columns with cyclical dependencies."

        # Create decoded table
        del cached_table, cached_ids, get_parent
        dec_table = pd.concat(tts.values(), axis=1, copy=False, join="inner")
        del tts
        # Re-order columns to metadata based order
        cols = [key for key in meta.cols.keys() if key != meta.primary_key]
        dec_table = dec_table[cols]

        return dec_table

    def reverse(
        self,
        data: LazyFrame,
        ctx: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
        parent_tables: dict[str, LazyFrame] | None = None,
    ):
        return self._reverse_chunk(data, ctx, ids, parent_tables)  # type: ignore

    def get_attributes(self) -> tuple[Attributes, dict[str, Attributes]]:
        """Returns information about the transformed columns and their hierarchical attributes."""
        assert self.fitted

        attrs = {}
        ctx_attrs = defaultdict(dict)
        for t in self.transformers.values():
            if isinstance(t, SeqTransformer):
                t_attrs, t_ctx_attrs = t.get_attributes()
                for n, c in t_ctx_attrs.items():
                    ctx_attrs[n].update(c)
            else:
                t_attrs = t.get_attributes()
            attrs.update(t_attrs)

        return attrs, dict(ctx_attrs)


def _fit_encoders_for_table(
    factory: EncoderFactory[AttributeEncoder], attrs: Attributes, data: LazyChunk
):
    table = data()
    encs = {}

    for name, attr in attrs.items():
        enc = factory.build()
        enc.fit(attr, table)
        encs[name] = enc

    return encs


@to_chunked
def _return_df(name: str, df: LazyChunk):
    return {name: df()}


@to_chunked
def _return_ids(name: str, df: LazyChunk):
    return {}, {}, {name: df()}


class AttributeEncoderHolder(
    ViewEncoder[dict[str, dict[str | tuple[str], META]]], Generic[META]
):
    """Receives tables that have been encoded by the base transformers and have
    attributes, and reformats them to fit a specific model."""

    table_encoders: dict[str, dict[str | tuple[str], AttributeEncoder[META]]]
    ctx_encoders: dict[str, dict[str, dict[str | tuple[str], AttributeEncoder[META]]]]

    def __init__(self, encoder: AttributeEncoderFactory, **kwargs) -> None:
        self.kwargs = kwargs
        self._encoder_fr = encoder
        self.table_encoders = {}
        self.ctx_encoders = {}

    def fit(
        self,
        attrs: dict[str, Attributes],
        tables: dict[str, LazyFrame],
        ctx_attrs: dict[str, dict[str, Attributes]],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        self.encoders = {}

        base_args = {"factory": self._encoder_fr}
        per_call = []
        per_call_meta = []

        # Create granular tasks for all table partitions
        for name in tables:
            for pid, part in tables[name].items():
                per_call.append({"attrs": attrs[name], "data": part})
                per_call_meta.append({"ctx": False, "table": name, "pid": pid})

        for creator, ctxs in ctx.items():
            for name in ctxs:
                for pid, part in ctxs[name].items():
                    per_call.append({"attrs": ctx_attrs[creator][name], "data": part})
                    per_call_meta.append(
                        {"ctx": True, "creator": creator, "table": name, "pid": pid}
                    )

        # Process them
        out = process_in_parallel(
            _fit_encoders_for_table, per_call, base_args, desc="Fitting encoders"
        )

        # Entangle output
        table_enc = defaultdict(list)
        ctx_enc = defaultdict(lambda: defaultdict(list))
        for enc, meta in zip(out, per_call_meta):
            if meta["ctx"]:
                ctx_enc[meta["creator"]][meta["table"]].append(enc)
            else:
                table_enc[meta["table"]].append(enc)

        # Reduce resulting encoders
        self.table_encoders = {}
        self.ctx_encoders = {}

        for name, encs in table_enc.items():
            self.table_encoders[name] = reduce(_reduce_inner, encs)

        for creator, ctx_encs in ctx_enc.items():
            for name, encs in ctx_encs.items():
                self.ctx_encoders[creator][name] = reduce(_reduce_inner, encs)

        self.fitted = True

    @to_chunked
    def encode_chunk(self, name: str, table: LazyChunk, ctx: dict[str, LazyChunk]):
        tts = []

        cached_table = table()
        cached_ctx = {n: c() for n, c in ctx.items()}

        for enc in self.table_encoders[name].values():
            tts.append(enc.encode(cached_table))

        for creator, ctx_encs in self.ctx_encoders.items():
            if name not in ctx_encs:
                continue

            for enc in ctx_encs[name].values():
                tts.append(enc.encode(cached_ctx[creator]))

        return {name: pd.concat(tts, axis=1, copy=False, join="inner")}

    def encode(
        self,
        tables: dict[str, LazyFrame],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        lazies = set()
        for name in tables:
            table_ctx = {}

            for creator, child_ctx in ctx.items():
                if name in child_ctx:
                    table_ctx[creator] = child_ctx[name]

            lazies |= self.encode_chunk(name, tables[name], table_ctx)

        # Passthrough ids
        for name, tid in ids.items():
            lazies |= _return_df(name + "_ids", tid)

        return lazies

    @to_chunked
    def decode_chunk(
        self, name: str, data: LazyChunk
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, dict[str, pd.DataFrame]],
        dict[str, pd.DataFrame],
    ]:
        table = data()

        # Decode main table
        tts = []
        for enc in self.table_encoders[name].values():
            tts.append(enc.decode(table))
        tables = {name: pd.concat(tts, axis=1, copy=False, join="inner")}

        # Decode context tables
        ctx_tts: dict[str, list] = defaultdict(list)
        for creator, ctx_encs in self.ctx_encoders.items():
            if name not in ctx_encs:
                continue

            for ctx_enc in ctx_encs[name].values():
                ctx_tts[creator].append(ctx_enc.decode(table))

        ctx = {
            creator: {name: pd.concat(tts, axis=1, copy=False, join="inner")}
            for creator, tts in ctx_tts.items()
        }

        return tables, ctx, {}

    def decode(
        self,
        data: dict[str, LazyFrame],
    ):
        lazies = set()

        for name, table in data.items():
            lazies |= self.decode_chunk(name, table)

        # Passthrough ids
        for name, tid in data.items():
            if name.endswith("_ids"):
                lazies |= _return_ids(name.replace("_ids", ""), tid)

        return lazies

    def get_metadata(self) -> dict[str, dict[str | tuple[str], META]]:
        out = defaultdict(dict)

        for table, encs in self.table_encoders.items():
            for name, enc in encs.items():
                out[table][name] = enc.get_metadata()

        for cencs in self.ctx_encoders.values():
            for table, encs in cencs.items():
                for enc in encs.values():
                    out[table].update(enc.get_metadata())

        return out
