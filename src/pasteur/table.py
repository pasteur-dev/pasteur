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
from typing import Any, Callable, cast

import pandas as pd

from .attribute import Attribute, Attributes
from .encode import AttributeEncoder, EncoderFactory, ViewEncoder
from .metadata import ColumnRef, Metadata
from .module import Module, get_module_dict
from .transform import RefTransformer, SeqTransformer, Transformer, TransformerFactory
from .utils import LazyChunk, LazyFrame, LazyPartition, to_chunked
from .utils.progress import process_in_parallel, reduce

logger = logging.getLogger(__file__)


_IDKEY = "__ids_lkjhasndsfnewr"

def _reduce_inner(
    a: dict[str | tuple[str], Transformer],
    b: dict[str | tuple[str], Transformer],
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

def _lazy_load_tables(tables: dict[str, LazyChunk | pd.DataFrame]):
    """ Lazy loads partitions and keeps them in-memory in a closure.
    
    Once the functions go out of scope, the partitions are released."""
    cached_tables = {}

    def _get_table(name: str):
        if name in cached_tables:
            return cached_tables[name]
        
        candidate = tables[name]
        if callable(candidate):
            table = candidate()
        else:
            table = candidate

        cached_tables[name] = table
        return table
    
    return _get_table


def _calc_joined_refs(
    name: str,
    get_table: Callable[[str], pd.DataFrame],
    ids: pd.DataFrame | None,
    cref: list[ColumnRef] | ColumnRef | None,
):
    """ Returns a dataframe where for each row in the original data,
    reference values are provided matching the ones in cref.
    
    In the case of one reference, a series is returned.
    If no references are provided, returns None. """
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
    """ Returns a dictionary containing columns from all upstream parents,
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
        ref: ReferenceManager,
        meta: Metadata,
        name: str,
        transformers: dict[str, TransformerFactory],
    ) -> None:
        self.ref = ref
        self.name = name
        self.meta = meta

        self.transformers: dict[str | tuple[str], Transformer] = {}
        self.transformer_cls = transformers
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
        get_table = _lazy_load_tables(tables) # type: ignore
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

    def _load_ids(self, ids: LazyChunk | pd.DataFrame | None, get_table: Callable[[str], pd.DataFrame]):
        """ Loads ids only if required. If `ids` is None, it calculates them anew using
        `get_table` and the reference manager. """
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

        get_table = _lazy_load_tables(tables) # type: ignore
        loaded_ids = self._load_ids(ids, get_table)
        meta = self.meta[self.name]
        table = get_table(self.name)
        tts = []

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            trn = self.transformers[name]
            if isinstance(trn, SeqTransformer):
                # Add foreign column if required
                ref_cols = _calc_unjoined_refs(self.name, get_table, col.ref)
                assert loaded_ids is not None
                tt = trn.transform(table[name], ref_cols, loaded_ids)
            elif isinstance(trn, RefTransformer):
                # Add foreign column if required
                ref_cols = _calc_joined_refs(self.name, get_table, loaded_ids, col.ref)
                tt = trn.transform(table[name], ref_cols)
            else:
                tt = trn.transform(table[name])

            tts.append(tt)

        return pd.concat(tts, axis=1, copy=False, join="inner")

    def transform(
        self,
        tables: dict[str, LazyFrame],
        ids: LazyFrame | None = None,
    ):
        return to_chunked(self.transform_chunk)(tables, ids) #type: ignore

    def reverse(
        self,
        table: LazyChunk | pd.DataFrame,
        ids: pd.DataFrame | LazyChunk | None = None,
        parent_tables: dict[str, LazyChunk] | dict[str, pd.DataFrame] | None = None,
    ):
        # If there are no ids that reference a foreign table, the ids and
        # parent_table parameters can be set to None (ex. tabular data).
        assert self.fitted
        transformers = self.transformers

        cached_table = table() if callable(table) else table
        cached_ids = (ids() if callable(ids) else ids) if ids is not None else None
        if parent_tables:
            cached_parents = {
                name: part() if callable(part) else part
                for name, part in parent_tables.items()
            }
        else:
            cached_parents = {}

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

                ref_col = None
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

                    table_cols: dict[str | None, list[str]] = defaultdict(list)

                    for ref in cref:
                        assert ref.col is not None
                        table_cols[ref.table].append(ref.col)

                    dfs = []
                    for rtable, refs in table_cols.items():
                        if rtable:
                            assert cached_ids and rtable in cached_parents

                            df = cached_ids.join(
                                cached_parents[rtable][refs], on=rtable
                            )[refs].add_prefix(f"{rtable}.")
                            dfs.append(df)
                        else:
                            dfs.append(tts[refs])

                    ref_col = pd.concat(dfs)
                elif cref:
                    ref = cast(ColumnRef, cref)
                    # Check ref requirements met
                    if not ref.table and ref.col not in completed_cols:
                        continue

                    f_table, f_col = ref.table, ref.col
                    assert cached_ids and cached_parents and f_col and f_table
                    assert (
                        f_table in cached_parents
                    ), f"Attempted to reverse table {self.name} before reversing {f_table}, which is required by it."
                    ref_col = cached_ids.join(
                        cached_parents[f_table][f_col], on=f_table
                    )[f_col]

                t = transformers[name]
                if isinstance(t, RefTransformer) and ref_col is not None:
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
        del cached_table, cached_parents
        dec_table = pd.concat(tts.values(), axis=1, copy=False, join="inner")
        del tts
        # Re-order columns to metadata based order
        cols = [key for key in meta.cols.keys() if key != meta.primary_key]
        dec_table = dec_table[cols]

        return dec_table

    def get_attributes(self) -> dict[str, Attribute]:
        """Returns information about the transformed columns and their hierarchical attributes."""
        assert self.fitted

        attrs = {}
        for t in self.transformers.values():
            attrs[t.attr.name] = t.attr
        return attrs

    def find_ids(self, tables: dict[str, pd.DataFrame]):
        return self.ref.find_foreign_ids(self.name, tables)


class AttributeEncoderHolder(TableEncoder):
    """Receives tables that have been encoded by the base transformers and have
    attributes, and reformats them to fit a specific model."""

    encoders: dict[str, AttributeEncoder]

    def __init__(self, encoder: EncoderFactory, **kwargs) -> None:
        self.kwargs = kwargs
        self._encoder_fr = encoder
        self.encoders = {}

    def fit(
        self,
        attrs: dict[str, Attributes],
        tables: dict[str, LazyFrame],
        ctx: dict[str, LazyFrame],
        ids: LazyFrame,
    ) -> Attributes:
        self.encoders = {}

        for n, a in attrs.items():
            t = self._encoder_fr.build(**self.kwargs)
            t.fit(a, data)
            self.encoders[n] = t

        return self.attrs

    def encode(
        self, tables: dict[str, LazyFrame], ctx: dict[str, LazyFrame], ids: LazyFrame
    ) -> pd.DataFrame:
        cols = []
        table = data() if callable(data) else data

        for t in self.encoders.values():
            cols.append(t.encode(table))

        del table
        return pd.concat(cols, axis=1, copy=False, join="inner")

    def decode(
        self,
        data: dict[str, LazyPartition[Any]],
    ) -> tuple[
        LazyFrame | pd.DataFrame,
        dict[str, LazyFrame | pd.DataFrame],
        LazyFrame | pd.DataFrame,
    ]:
        cols = []
        table = enc() if callable(enc) else enc

        for t in self.encoders.values():
            cols.append(t.decode(table))

        del table
        return pd.concat(cols, axis=1, copy=False, join="inner")

    @property
    def attrs(self):
        return {a.name: a for a in [t.attr for t in self.encoders.values()]}

    def get_attributes(self):
        return self.attrs


class TransformHolder:
    """Handles transforming and encoding a table."""

    def __init__(
        self,
        meta: Metadata,
        name: str,
        transformers: dict[str, TransformerFactory] | None = None,
        encoders: dict[str, EncoderFactory] | None = None,
        modules: list[Module] | None = None,
    ) -> None:
        self.name = name
        self.meta = meta

        if modules:
            transformers = get_module_dict(TransformerFactory, modules)
            encoders = get_module_dict(EncoderFactory, modules)
        else:
            assert (
                transformers and encoders
            ), "Either modules or transformers and encoders should be provided."

        self.ref = ReferenceManager(meta, name)
        self.transformer = TableTransformer(self.ref, meta, name, transformers)
        self.fitted = False

        self.encoders = {
            name: ColumnEncoderHolder(encoder) for name, encoder in encoders.items()
        }

    def find_ids(self, tables: dict[str, pd.DataFrame]):
        return self.ref.find_foreign_ids(self.name, tables)

    def transform(
        self,
        tables: dict[str, pd.DataFrame] | dict[str, LazyChunk],
        ids: pd.DataFrame | LazyChunk | None = None,
    ):
        return self.transformer.transform(tables, ids)

    def reverse(
        self,
        table: pd.DataFrame | LazyChunk,
        ids: pd.DataFrame | LazyChunk | None = None,
        parent_tables: dict[str, LazyChunk] | dict[str, pd.DataFrame] | None = None,
    ):
        return self.transformer.reverse(table, ids, parent_tables)

    def fit_transform(
        self,
        tables: dict[str, pd.DataFrame] | dict[str, LazyChunk],
        ids: pd.DataFrame | LazyChunk | None = None,
    ):
        """Fits the base transformer and the encoding transformers.

        The data has to be encoded using the base transformer to make this possible.
        So it's returned by the function, and there's no `fit` only function."""
        cached_tables = {
            name: table() if callable(table) else table
            for name, table in tables.items()
        }
        cached_ids = (ids() if callable(ids) else ids) if ids is not None else None

        if self.ref.table_has_reference() and ids is None:
            cached_ids = self.ref.find_foreign_ids(self.name, cached_tables)

        self.transformer.fit(cached_tables, cached_ids)

        attrs = self.transformer.get_attributes()
        table = self.transformer.transform(cached_tables, cached_ids)

        for encoder in self.encoders.values():
            encoder.fit(attrs, table)

        self.fitted = True
        return table, ids

    def get_attributes(self) -> dict[str, Attribute]:
        return self.transformer.get_attributes()

    def __getitem__(self, type):
        assert self.fitted
        return self.encoders[type]
