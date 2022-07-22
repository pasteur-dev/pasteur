from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import math

from .metadata import Metadata
from .transform import ChainTransformer, Transformer


class TableTransformer:
    """Holds the transformer dictionary for this table and manages the foreign relationships of the table."""

    def __init__(self, meta: Metadata, name: str, type: str) -> None:
        self.name = name
        self.meta = meta
        self.transformers: Dict[str, Transformer] = {}

        assert type in ("idx", "num", "bin")
        self.type = type

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
            ids = ids.join(foreign_ids, on=table)

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

    def fit(self, tables: Dict[str, pd.DataFrame]):
        # Only load foreign keys if required by column
        ids = None
        if self.table_has_reference():
            ids = self.find_foreign_ids(self.name, tables)

        meta = self.meta[self.name]
        table = tables[self.name]

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

            # Fit transformer with proper chain
            chain = col.chains[self.type]
            args = col.args.copy()
            t = ChainTransformer.from_dict(chain, args)
            t.fit(table[[name]], ref_col)

            self.transformers[name] = t

    def transform(self, tables: Dict[str, pd.DataFrame]):
        ids = self.find_foreign_ids(self.name, tables)

        meta = self.meta[self.name]
        table = tables[self.name]

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

            tt = self.transformers[name].transform(table[[name]], ref_col)
            tts.append(tt)

        return pd.concat(tts, axis=1), ids

    def reverse(
        self,
        table: pd.DataFrame,
        ids: Optional[pd.DataFrame],
        parent_tables: Dict[str, pd.DataFrame],
    ):
        # If there are no ids that reference a foreign table, the ids parameter
        # can be set to None (ex. tabular data).

        meta = self.meta[self.name]

        # Process columns with no dependencies first
        tts = []
        for name, col in meta.cols.items():
            if col.is_id() or col.ref is not None:
                continue

            tt = self.transformers[name].reverse(table[[name]])
            tts.append(tt)

        # Process columns with dependencies afterwards.
        # fix-me: assumes no nested dependencies within the same table
        parent_cols = pd.concat(tts, axis=1)
        for name, col in meta.cols.items():
            if col.is_id() or col.ref is None:
                continue

            ref_col = None
            f_table, f_col = col.ref.table, col.ref.col
            if f_table:
                # Foreign column from another table
                assert (
                    f_table in parent_tables
                ), f"Attempted to reverse table {self.name} before reversing {f_table}, which is required by it."
                ref_col = ids.join(parent_tables[f_table][f_col], on=f_table)[f_col]
            else:
                # Local column, duplicate and rename
                ref_col = parent_cols[f_col]

            tt = self.transformers[name].reverse(table[[name]], ref_col)
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

        return dec_table
