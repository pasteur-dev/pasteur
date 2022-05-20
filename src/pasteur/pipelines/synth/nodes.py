from __future__ import absolute_import
from typing import Dict
from sdv import sdv
from sdv.metadata import Metadata, Table

import pandas as pd


def fit_table(table: pd.DataFrame, metadata: Dict):
    transformer = Table.from_dict(metadata)
    transformer._dtype_transformers.update({"O": "categorical_fuzzy"})
    transformer.fit(table.reset_index())
    return transformer


def transform_table(table: pd.DataFrame, transformer: Table):
    return transformer.transform(table.reset_index())


def reverse_transform_table(table: pd.DataFrame, transformer: Table):
    return transformer.reverse_transform(table)
