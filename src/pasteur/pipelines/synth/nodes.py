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


def synth_fit(alg: str, metadata: Dict, **kwargs: pd.DataFrame):
    assert alg.lower() == "hma1"

    tables = kwargs
    metadata = Metadata(metadata)

    model = HMA1(metadata)
    model.fit(tables)

    return model


def synth_fit_closure(alg: str):
    fun = lambda *args, **kwargs: synth_fit(alg, *args, **kwargs)
    fun.__name__ = "fit_%s_model" % alg
    return fun


def synth_sample(alg: str, model: BaseRelationalModel):
    assert alg.lower() == "hma1"
    return model.sample()


def synth_sample_closure(alg: str):
    fun = lambda *args, **kwargs: synth_sample(alg, *args, **kwargs)
    fun.__name__ = "sample_with_%s" % alg
    return fun
