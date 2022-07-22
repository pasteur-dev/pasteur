from copy import deepcopy
from typing import Dict

import pandas as pd

from ...dataset_transform import TableTransformer
from ...metadata import Metadata


def fit_table_closure(name: str, type: str):
    def fit_table(metadata: Dict, **tables: Dict[str, pd.DataFrame]):
        meta = Metadata(metadata, tables)
        t = TableTransformer(meta, name, type)
        t.fit(tables)
        return t

    return fit_table


def transform_table(transformer: TableTransformer, **tables: Dict[str, pd.DataFrame]):
    return transformer.transform(tables)


def reverse_table(
    transformer: TableTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: Dict[str, pd.DataFrame]
):
    return transformer.reverse(ids, table, parents)


def transform_table_tab(
    transformer: TableTransformer, **tables: Dict[str, pd.DataFrame]
):
    table = transformer.transform(tables)
    return table


def reverse_table_tab(
    transformer: TableTransformer,
    table: pd.DataFrame,
    **parents: Dict[str, pd.DataFrame]
):
    return transformer.reverse(None, table, parents)
