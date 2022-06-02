from __future__ import absolute_import
from copy import deepcopy
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


def hma1_fit(metadata: Dict, **kwargs: pd.DataFrame):
    from sdv.relational import HMA1
    from sdv.metadata import Metadata

    # Reset primary key index since sdv doesn't support indexes
    tables = kwargs
    tables = {n: t.reset_index(drop=not t.index.name) for n, t in tables.items()}

    # Create new metadata for transformed data
    new_meta = deepcopy(metadata)

    # Keep only IDs from previous dictionary
    for name, old_dict in metadata["tables"].items():
        new_dict = new_meta["tables"][name]

        new_dict["fields"] = {
            n: t for n, t in old_dict["fields"].items() if t["type"] == "id"
        }

    # Add all other inputs (already encoded) as float
    for name, table in tables.items():
        for col in table.keys():
            if col not in new_meta["tables"][name]["fields"]:
                new_meta["tables"][name]["fields"][col] = {
                    "type": "numerical",
                    "subtype": "float",
                }

    metadata = Metadata(new_meta)

    model = HMA1(metadata)
    model.fit(tables)

    return model


def synth_alg_get_fit(alg: str):
    assert alg.lower() == "hma1"
    return hma1_fit


def hma1_sample(model):
    # Patch finalize to add stray keys not included by default
    # FIXME: remove me when SDV is fixed
    old_finalize = model._finalize
    metadata = model.metadata

    def new_finalize(sampled_data):
        for name, table in sampled_data.items():
            for field, data in metadata.get_fields(name).items():
                if field == metadata.get_primary_key(name) or field in table:
                    continue
                if data["type"] == "id" and "ref" not in data:
                    table[field] = range(len(table))

        return old_finalize(sampled_data)

    model._finalize = new_finalize

    return model.sample()


def synth_alg_get_sample(alg: str):
    assert alg.lower() == "hma1"
    return hma1_sample
