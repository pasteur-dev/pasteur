"""Hypertransformer with support for IDs."""

from typing import Callable
import pandas as pd
from rdt import HyperTransformer


class IdHyperTransformer(HyperTransformer):
    def __init__(self, *args, **kwargs):
        self.meta = kwargs.pop("metadata")
        super().__init__(*args, **kwargs)

        if self.meta:
            self.primary_key = self.meta.get("primary_key")

            fields = self.meta.get("fields")
            field_data = {n: f.get("type") for n, f in fields.items()}

            self.field_data_types = {
                n: f for n, f in field_data.items() if f and f != "id"
            }
            self.ids = {n for n, f in field_data.items() if f == "id"}
        else:
            self.primary_key = None
            self.ids = None

    def fit(self, data: pd.DataFrame):
        super().fit(data.reset_index(drop=not data.index.name).drop(columns=self.ids))

    def _call_without_ids(
        self, data: pd.DataFrame, callback: Callable[[pd.DataFrame], pd.DataFrame]
    ):
        is_idx_key = self.primary_key and data.index.name == self.primary_key

        data_new_idx = data.reset_index(drop=not data.index.name)
        data_ids = data_new_idx[list(self.ids)]
        data_cols = data_new_idx.drop(columns=list(self.ids))

        mutated: pd.DataFrame = data_ids.join(callback(data_cols))
        if is_idx_key:
            mutated = mutated.set_index(self.primary_key)
        return mutated

    def transform(self, data: pd.DataFrame):
        return self._call_without_ids(
            data, lambda x: super(IdHyperTransformer, self).transform(x)
        )

    def reverse_transform(self, data):
        return self._call_without_ids(
            data, lambda x: super(IdHyperTransformer, self).reverse_transform(x)
        )
