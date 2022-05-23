from __future__ import absolute_import
from curses import meta
from importlib.metadata import metadata
from statistics import median
from typing import Dict

from rdt import HyperTransformer
from sdv.relational import HMA1
from sdv.relational.base import BaseRelationalModel
from sdv.metadata import Metadata

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


def sdv_fit(metadata: Dict, **kwargs: pd.DataFrame):
    tables = kwargs
    metadata = Metadata(metadata)

    model = HMA1(metadata)
    model.fit(tables)

    return model


def sdv_sample(model: BaseRelationalModel):
    return model.sample()
