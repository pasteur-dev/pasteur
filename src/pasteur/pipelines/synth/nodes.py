from __future__ import absolute_import
from typing import Dict
from sdv import sdv
from sdv.metadata import Metadata

import pandas as pd


def transform_tables(metadata: Dict, **tables: pd.DataFrame):
    metadata = Metadata(metadata)
    encoded = {name: metadata.transform(table) for name, table in tables.items()}
    return {"metadata": metadata, **encoded}


def reverse_transform_tables(metadata: Metadata, **encoded: pd.DataFrame):
    return {name: metadata.transform(table) for name, table in encoded.items()}
