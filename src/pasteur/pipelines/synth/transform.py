from copy import deepcopy
from typing import Collection

import pandas as pd

from ...transform import TableTransformer
from ...metadata import Metadata


def fit_table_closure(name: str, types: Collection[str]):
    def fit_table(metadata: dict, params: dict, **tables: dict[str, pd.DataFrame]):
        meta = Metadata(metadata, tables, params)
        t = TableTransformer(meta, name, types)
        t.fit(tables)
        return t

    return fit_table


def find_ids(transformer: TableTransformer, **tables: dict[str, pd.DataFrame]):
    return transformer.find_ids(tables)


def transform_table_closure(type: str):
    def transform_table(
        transformer: TableTransformer,
        ids: pd.DataFrame,
        **tables: dict[str, pd.DataFrame]
    ):
        return transformer.transform(type, tables, ids)

    return transform_table


def reverse_table_closure(type: str):
    def reverse_table(
        transformer: TableTransformer,
        ids: pd.DataFrame,
        table: pd.DataFrame,
        **parents: dict[str, pd.DataFrame]
    ):
        return transformer.reverse(type, table, ids, parents)

    return reverse_table


def transform_table_tab(
    transformer: TableTransformer, **tables: dict[str, pd.DataFrame]
):
    table = transformer.transform(tables)
    return table


def reverse_table_tab(
    transformer: TableTransformer,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame]
):
    return transformer.reverse(table, None, parents)
