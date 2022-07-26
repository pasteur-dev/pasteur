from copy import deepcopy
from typing import Dict

from itertools import chain
import pandas as pd


class Synth:
    name = None
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False

    @staticmethod
    def fit(metadata: Dict, **kwargs: pd.DataFrame):
        assert False, "Not implemented"

    @staticmethod
    def sample(model):
        assert False, "Not implemented"


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = "ident_idx"
    type = "idx"
    tabular = True
    multimodal = True
    timeseries = True

    @staticmethod
    def fit(metadata: Dict, **kwargs: pd.DataFrame):
        return kwargs

    @staticmethod
    def sample(model):
        return model


class NumIdentSynth(IdentSynth):
    name = "ident_num"
    type = "num"


class BinIdentSynth(IdentSynth):
    name = "ident_bin"
    type = "bin"


class Hma1Synth(Synth):
    name = "hma1"
    type = "num"
    tabular = True

    @staticmethod
    def fit(metadata: Dict, **kwargs: pd.DataFrame):
        from sdv.relational import HMA1
        from sdv.metadata import Metadata as SdvMeta

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

        metadata = SdvMeta(new_meta)

        model = HMA1(metadata)
        model.fit(tables)

        return model

    @staticmethod
    def sample(model):
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


def get_algs():
    return {
        t.name: t for t in chain(Synth.__subclasses__(), IdentSynth.__subclasses__())
    }
