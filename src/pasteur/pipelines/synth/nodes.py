from __future__ import absolute_import
from typing import Dict

from rdt import HyperTransformer

import pandas as pd

from .transformers import IdHyperTransformer


def fit_table(table: pd.DataFrame, metadata: Dict):
    transformer = IdHyperTransformer(metadata=metadata)
    transformer.fit(table)
    return transformer


def transform_table(table: pd.DataFrame, transformer: HyperTransformer):
    return transformer.transform(table)


def reverse_transform_table(table: pd.DataFrame, transformer: HyperTransformer):
    return transformer.reverse_transform(table)
