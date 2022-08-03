from copy import deepcopy
from typing import Dict

import pandas as pd

from .base import Synth

# FIXME: Dead code


class Hma1Synth(Synth):
    name = "hma1"
    type = "num"
    tabular = True

    def fit(self, metadata: Dict, data: dict[str, pd.DataFrame]):
        from sdv.metadata import Metadata as SdvMeta
        from sdv.relational import HMA1

        # Reset primary key index since sdv doesn't support indexes
        tables = data
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

        self._model = model

    def sample(self):
        model = self._model

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
