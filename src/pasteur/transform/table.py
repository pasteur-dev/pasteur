from typing import Collection, Dict, Optional
import pandas as pd
import logging

from ..metadata import Metadata, DEFAULT_TRANSFORMERS
from .base import ChainTransformer, Transformer

logger = logging.getLogger(__name__)

DEFAULT_TYPES = list(DEFAULT_TRANSFORMERS.keys())


class TableTransformer:
    """Holds the transformer dictionary for this table and manages the foreign relationships of the table."""

    def __init__(
        self,
        meta: Metadata,
        name: str,
        types: str | Collection[str] = DEFAULT_TYPES,
    ) -> None:
        self.name = name
        self.meta = meta
        self.transformers: Dict[str, Dict[str, Transformer]] = {}
        self.fitted = False

        if isinstance(types, str):
            types = [types]
        self.types = types

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

    def fit(self, tables: Dict[str, pd.DataFrame], ids: pd.DataFrame | None = None):
        # Only load foreign keys if required by column
        ids = None
        if self.table_has_reference() and ids is None:
            ids = self.find_foreign_ids(self.name, tables)

        meta = self.meta[self.name]
        table = tables[self.name]

        if not meta.primary_key == table.index.name:
            assert (
                False
            ), "Properly formatted datasets should have their primary key as their index column"
            # table.reindex(meta.primary_key)

        transformers = {}
        for type in self.types:
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
                chain = col.chains[type]
                args = col.args.copy()
                t = ChainTransformer.from_dict(chain, args)
                t.fit(table[[name]], ref_col)

                transformers[name] = t

            self.transformers[type] = transformers
        self.fitted = True

    def transform(
        self,
        type: str,
        tables: Dict[str, pd.DataFrame],
        ids: pd.DataFrame | None = None,
    ):
        assert type in self.types and self.fitted

        meta = self.meta[self.name]
        table = tables[self.name]

        # Only load foreign keys if required by column
        ids = None
        if self.table_has_reference():
            if ids is None:
                ids = self.find_foreign_ids(self.name, tables)
            # If we do have foreign relationships drop all rows that don't have
            # parents and warn
            if len(ids) < len(table):
                logger.warning(
                    f"Found {len(table) - len(ids)} rows without ids on table {self.name}. Dropping before fitting..."
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

            tt = self.transformers[type][name].transform(table[[name]], ref_col)
            tts.append(tt)

        return pd.concat(tts, axis=1)

    def reverse(
        self,
        type: str,
        table: pd.DataFrame,
        ids: Optional[pd.DataFrame] = None,
        parent_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        # If there are no ids that reference a foreign table, the ids and
        # parent_table parameters can be set to None (ex. tabular data).
        assert type in self.types and self.fitted
        transformers = self.transformers[type]

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

            tt = transformers[name].reverse(table, ref_col)
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
            tt = transformers[name].reverse(table, ref_col)
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

    def get_attributes(
        self, type: str, table: pd.DataFrame | None = None
    ) -> dict[str, list[str]]:
        """Collates the table columns into ordered sets named attributes (sourced from transformer hierarchy).

        If there's an attribute a0 with columns c0, c1, c2,
        we only take c1 into consideration for a parent relationship with c0,
        and c2 with both c1, c0.

        Lowers computational complexity for bayesian networks from
        `O(nChooseK(c_t, k*c_n))` to `O(nChooseK(a_n, k)*c_n)`, where the latter is
        orders of magnitude smaller (combinations of marginals).
        c_t: total columns, c_n: columns per attribute, k: columns selected for parents,
        a_n: number of attributes

        Example (binary encoding; 50 bits in total):
        a_n=10, c_t=50, c_n=5
        From O(...) = 2118760 for k=1
        To O(...) = 1000 for k=4
        """
        assert type in self.types and self.fitted
        transformers = self.transformers[type].values()

        # Add actual attributes
        attrs = {}
        for t in transformers:
            attrs.update(t.get_hierarchy())

        if table is None:
            return attrs

        # Add fake attributes for cols with no hierarchical relationship
        cols_in_attr = set()
        for a in attrs.values():
            cols_in_attr.update(a)

        cols = set(table.columns)
        attrs.update({c: [c] for c in cols - cols_in_attr})
        return attrs