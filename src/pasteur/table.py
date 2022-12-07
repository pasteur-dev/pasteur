import logging

import pandas as pd

from .attribute import Attribute, Attributes
from .encode import Encoder, EncoderFactory
from .metadata import Metadata
from .module import Module, get_module_dict
from .transform import RefTransformer, Transformer, TransformerFactory
from .utils import LazyChunk, LazyFrame

logger = logging.getLogger(__file__)


class ReferenceManager:
    """Manages the foreign relationships of a table"""

    def __init__(
        self,
        meta: Metadata,
        name: str,
    ) -> None:
        self.name = name
        self.meta = meta

    def find_parents(self, table: str):
        """Finds the reference cols that link a table to its parents and
        returns tuples with the name of that column, the name of the parent table
        and the name of the parent key column (usually the primary key)."""

        res = []

        meta = self.meta[table]
        for name, col in meta.cols.items():
            if col.is_id():
                if col.ref:
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

    def find_foreign_ids(self, name: str, tables: dict[str, pd.DataFrame]):
        """Creates an id lookup table for the provided table.

        The id lookup table is composed of columns that have the foreign table name and
        contain its index key to join on."""
        ids = self.get_foreign_keys(name, tables[name])

        for col, table, foreign_col in self.find_parents(name):
            assert (
                foreign_col == self.meta[table].primary_key
            ), "Only referencing primary keys supported for now."

            ids = ids.rename(columns={col: table})
            foreign_ids = self.find_foreign_ids(table, tables)
            ids = ids[~pd.isna(ids[table])].join(foreign_ids, on=table, how="inner")

        return ids

    def table_has_reference(self):
        """Checks whether the table has a column that depends on another table's
        column for transformation."""
        name = self.name
        meta = self.meta[name]

        for col in meta.cols.values():
            if not col.is_id() and col.ref is not None and col.ref.table is not None:
                return True
        return False

    def find_ids(self, tables: dict[str, pd.DataFrame]):
        return self.find_foreign_ids(self.name, tables)


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

        self.transformers: dict[str, Transformer] = {}
        self.transformer_cls = transformers
        self.fitted = False

    def fit(
        self,
        tables: dict[str, pd.DataFrame],
        ids: pd.DataFrame | None = None,
    ):
        meta = self.meta[self.name]
        if isinstance(tables, dict):
            table = tables[self.name]
        else:
            table = tables
        transformers = self.transformer_cls

        if self.ref.table_has_reference():
            if ids is None:
                ids = self.ref.find_foreign_ids(self.name, tables)

            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(ids)} rows without ids on table {self.name}. Dropping before fitting..."
                )
                table = table.loc[ids.index]

        if not meta.primary_key == table.index.name:
            assert (
                False
            ), "Properly formatted datasets should have their primary key as their index column"
            # table.reindex(meta.primary_key)

        for name, col in meta.cols.items():
            if col.is_id():
                continue

            # Add foreign column if required
            ref_col = None
            if col.ref:
                f_table, f_col = col.ref.table, col.ref.col
                assert f_col
                if f_table:
                    assert ids is not None
                    # Foreign column from another table
                    ref_col = ids.join(tables[f_table][f_col], on=f_table)[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            assert (
                col.type in transformers
            ), f"Column type {col.type} not in transformers:\n{list(transformers.keys())}"

            # Fit transformer
            if "main_param" in col.args:
                t = transformers[col.type].build(col.args["main_param"], **col.args)
            else:
                t = transformers[col.type].build(**col.args)

            if isinstance(t, RefTransformer) and ref_col is not None:
                t.fit(table[name], ref_col)
            else:
                t.fit(table[name])
            self.transformers[name] = t

        self.fitted = True

    def transform(
        self,
        tables: dict[str, pd.DataFrame] | dict[str, LazyChunk],
        ids: pd.DataFrame | LazyChunk | None = None,
    ):
        assert self.fitted

        meta = self.meta[self.name]
        if isinstance(tables, dict):
            cached_tables = {
                name: part() if callable(part) else part
                for name, part in tables.items()
            }
            table = cached_tables[self.name]
        else:
            cached_tables = {self.name: tables() if callable(tables) else tables}
            table = cached_tables[self.name]

        # Only load foreign keys if required by column
        cached_ids = None
        if self.ref.table_has_reference():
            if ids is None:
                cached_ids = self.ref.find_foreign_ids(self.name, cached_tables)
            else:
                cached_ids = ids() if callable(ids) else ids
            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(cached_ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(cached_ids)} rows without ids on table {self.name}. Dropping before transforming..."
                )
                table = table.loc[cached_ids.index]

        tts = []
        for name, col in meta.cols.items():
            if col.is_id():
                continue

            # Add foreign column if required
            ref_col = None
            if col.ref:
                f_table, f_col = col.ref.table, col.ref.col
                assert f_col
                if f_table:
                    assert cached_ids is not None
                    # Foreign column from another table
                    ref_col = cached_ids.join(
                        cached_tables[f_table][f_col], on=f_table
                    )[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            t = self.transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.transform(table[name].copy(), ref_col)
            else:
                tt = t.transform(table[name].copy())
            tts.append(tt)
            del ref_col, tt

        del cached_ids, cached_tables, table
        return pd.concat(tts, axis=1, copy=False, join="inner")

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

        # Process columns with no intra-table dependencies first
        tts = {}
        for name, col in meta.cols.items():
            if col.is_id():
                continue
            if col.ref is not None and col.ref.table is None:
                # Intra-table dependencies
                continue

            ref_col = None
            if col.ref is not None:
                f_table, f_col = col.ref.table, col.ref.col
                assert cached_ids and cached_parents and f_col and f_table
                assert (
                    f_table in cached_parents
                ), f"Attempted to reverse table {self.name} before reversing {f_table}, which is required by it."
                ref_col = cached_ids.join(cached_parents[f_table][f_col], on=f_table)[
                    f_col
                ]

            t = transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.reverse(cached_table, ref_col)
            else:
                tt = t.reverse(cached_table)
            tts[name] = tt

        # Process columns with intra-table dependencies afterwards.
        # fix-me: assumes no nested dependencies within the same table
        for name, col in meta.cols.items():
            if col.is_id() or col.ref is None:
                # Columns with no dependencies have been processed
                continue
            if col.ref is not None and col.ref.table is not None:
                # Columns with inter-table dependencies have been processed
                continue

            assert col.ref.col
            ref_col = tts[col.ref.col]
            t = transformers[name]
            if isinstance(t, RefTransformer):
                tt = t.reverse(cached_table, ref_col)
            else:
                tt = t.reverse(cached_table)
            tts[name] = tt

        # Re-add ids
        # If an id references another table it will be merged from the ids
        # dataframe. Otherwise, it will be set to 0 (irrelevant to data synthesis).
        del cached_table, cached_parents
        dec_table = pd.concat(tts.values(), axis=1, copy=False, join="inner")
        del tts
        for name, col in meta.cols.items():
            if not col.is_id() or name == meta.primary_key:
                continue

            if col.ref is not None:
                assert col.ref.table and cached_ids is not None
                dec_table[name] = cached_ids[col.ref.table]
            else:
                dec_table[name] = 0

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


class TableEncoder:
    """Receives tables that have been encoded by the base transformers and have
    attributes, and reformats them to fit a specific model."""

    encoders: dict[str, Encoder]

    def __init__(self, encoder: EncoderFactory, **kwargs) -> None:
        self.kwargs = kwargs
        self._encoder_fr = encoder
        self.encoders = {}

    def fit(self, attrs: Attributes, data: pd.DataFrame | None = None) -> Attributes:
        self.encoders = {}

        for n, a in attrs.items():
            t = self._encoder_fr.build(**self.kwargs)
            t.fit(a, data)
            self.encoders[n] = t

        return self.attrs

    def encode(self, data: LazyChunk | pd.DataFrame) -> pd.DataFrame:
        cols = []
        table = data() if callable(data) else data

        for t in self.encoders.values():
            cols.append(t.encode(table))

        del table
        return pd.concat(cols, axis=1, copy=False, join="inner")

    def decode(self, enc: LazyChunk | pd.DataFrame) -> pd.DataFrame:
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
            name: TableEncoder(encoder) for name, encoder in encoders.items()
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
