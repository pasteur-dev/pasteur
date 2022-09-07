import logging
from typing import Collection, Dict, Optional

import pandas as pd

from pasteur.transform.encoding import EncodingTransformer

from ..metadata import Metadata
from .attribute import Attribute, Attributes
from .base import RefTransformer, Transformer
from ..utils import find_subclasses

logger = logging.getLogger(__name__)


def _get_transformers():
    return find_subclasses(Transformer)


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

    def find_foreign_ids(self, name: str, tables: Dict[str, pd.DataFrame]):
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

    def find_ids(self, tables: Dict[str, pd.DataFrame]):
        return self.find_foreign_ids(self.name, tables)


class BaseTableTransformer:
    def __init__(
        self,
        ref: ReferenceManager,
        meta: Metadata,
        name: str,
    ) -> None:
        self.ref = ref
        self.name = name
        self.meta = meta

        self.transformers: Dict[str, Transformer] = {}
        self.fitted = False

    def fit(
        self,
        tables: Dict[str, pd.DataFrame] | pd.DataFrame,
        ids: pd.DataFrame | None = None,
    ):
        # Only load foreign keys if required by column
        ids = None

        meta = self.meta[self.name]
        if isinstance(tables, dict):
            table = tables[self.name]
        else:
            table = tables
        transformers = _get_transformers()

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
                if f_table:
                    # Foreign column from another table
                    ref_col = ids.join(tables[f_table][f_col], on=f_table)[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            # Fit transformer
            t = transformers[col.type](**col.args)
            if isinstance(t, RefTransformer) and ref_col is not None:
                t.fit(table[name], ref_col)
            else:
                t.fit(table[name])
            self.transformers[name] = t

        self.fitted = True

    def transform(
        self,
        tables: Dict[str, pd.DataFrame] | pd.DataFrame,
        ids: pd.DataFrame | None = None,
    ):
        assert self.fitted

        meta = self.meta[self.name]
        if isinstance(tables, dict):
            table = tables[self.name]
        else:
            table = tables

        # Only load foreign keys if required by column
        ids = None
        if self.ref.table_has_reference():
            if ids is None:
                ids = self.ref.find_foreign_ids(self.name, tables)
            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(ids)} rows without ids on table {self.name}. Dropping before transforming..."
                )
                table = table.loc[ids.index]

        tts = []
        for name, col in meta.cols.items():
            if col.is_id():
                continue

            # Add foreign column if required
            ref_col = None
            if col.ref:
                f_table, f_col = col.ref.table, col.ref.col
                if f_table:
                    # Foreign column from another table
                    ref_col = ids.join(tables[f_table][f_col], on=f_table)[f_col]
                else:
                    # Local column, duplicate and rename
                    ref_col = table[f_col]

            t = self.transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.transform(table[name], ref_col)
            else:
                tt = t.transform(table[name])
            tts.append(tt)

        return pd.concat(tts, axis=1)

    def reverse(
        self,
        table: pd.DataFrame,
        ids: Optional[pd.DataFrame] = None,
        parent_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        # If there are no ids that reference a foreign table, the ids and
        # parent_table parameters can be set to None (ex. tabular data).
        assert self.fitted
        transformers = self.transformers

        meta = self.meta[self.name]

        # Process columns with no intra-table dependencies first
        tts = []
        for name, col in meta.cols.items():
            if col.is_id():
                continue
            if col.ref is not None and col.ref.table is None:
                # Intra-table dependencies
                continue

            ref_col = None
            if col.ref is not None:
                f_table, f_col = col.ref.table, col.ref.col
                assert (
                    f_table in parent_tables
                ), f"Attempted to reverse table {self.name} before reversing {f_table}, which is required by it."
                ref_col = ids.join(parent_tables[f_table][f_col], on=f_table)[f_col]

            t = transformers[name]
            if isinstance(t, RefTransformer) and ref_col is not None:
                tt = t.reverse(table, ref_col)
            else:
                tt = t.reverse(table)
            tts.append(tt)

        # Process columns with intra-table dependencies afterwards.
        # fix-me: assumes no nested dependencies within the same table
        parent_cols = pd.concat(tts, axis=1)
        for name, col in meta.cols.items():
            if col.is_id() or col.ref is None:
                # Columns with no dependencies have been processed
                continue
            if col.ref is not None and col.ref.table is not None:
                # Columns with inter-table dependencies have been processed
                continue

            ref_col = parent_cols[col.ref.col]
            t = transformers[name]
            if isinstance(t, RefTransformer):
                tt = t.reverse(table, ref_col)
            else:
                tt = t.reverse(table)

            # todo: make sure this solves inter-table dependencies
            parent_cols[name] = tt
            tts.append(tt)

        # Re-add ids
        # If an id references another table it will be merged from the ids
        # dataframe. Otherwise, it will be set to 0 (irrelevant to data synthesis).
        dec_table = pd.concat(tts, axis=1)
        for name, col in meta.cols.items():
            if not col.is_id() or name == meta.primary_key:
                continue

            if col.ref is not None:
                dec_table[name] = ids[col.ref.table]
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

    def find_ids(self, tables: Dict[str, pd.DataFrame]):
        return self.ref.find_foreign_ids(self.name, tables)


class TableTransformer:
    """Holds the transformer dictionary for this table and manages the foreign relationships of the table."""

    def __init__(
        self,
        meta: Metadata,
        name: str,
        types: str | Collection[str],
    ) -> None:
        self.name = name
        self.meta = meta

        self.ref = ReferenceManager(meta, name)
        self.base = BaseTableTransformer(self.ref, meta, name)
        self.transformers: Dict[str, EncodingTransformer] = {}
        self.fitted = False

        if isinstance(types, str):
            types = [types]
        self.types = types

    def find_ids(self, tables: Dict[str, pd.DataFrame]):
        return self.ref.find_foreign_ids(self.name, tables)

    def fit_transform(
        self, tables: Dict[str, pd.DataFrame], ids: pd.DataFrame | None = None
    ):
        """Fits the base transformer and the encoding transformers.

        The data has to be encoded using the base transformer to make this possible.
        So it's returned by the function, and there's no `fit` only function."""
        if self.ref.table_has_reference() and ids is None:
            ids = self.ref.find_foreign_ids(self.name, tables)

        self.base.fit(tables, ids)

        attrs = self.base.get_attributes()
        table = self.base.transform(tables, ids)

        for type in self.types:
            transformer = EncodingTransformer(type)
            transformer.fit(attrs, table)
            self.transformers[type] = transformer

        self.fitted = True
        return table, ids

    def __getitem__(self, type):
        assert self.fitted
        return self.transformers[type]
