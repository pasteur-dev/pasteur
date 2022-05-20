from __future__ import absolute_import
from typing import Dict
from sdv import sdv
from sdv.metadata import Metadata, Table

import pandas as pd


def transform_tables(metadata: Dict, **tables: pd.DataFrame):
    metadata = Metadata(metadata)

    transformers = {}
    encoded = {}

    for name, table in tables.items():
        transformer = Table.from_dict(metadata.get_table_meta(name))
        transformer.fit(table.reset_index())

        transformers[name] = transformer
        encoded[name] = transformer.transform(table)

    encoded = {
        name: metadata.get_table_meta(name).fit(table.reset_index())
        for name, table in tables.items()
    }
    return {"transformers": transformers, **encoded}


def reverse_transform_tables(transformers: Dict[str, Table], **encoded: pd.DataFrame):
    return {
        name: transformers[name].reverse_transform(table)
        for name, table in encoded.items()
    }


def create_sdv_model(metadata: Metadata, **encoded: pd.DataFrame):
    pass
