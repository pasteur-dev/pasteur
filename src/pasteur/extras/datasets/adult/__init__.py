from __future__ import annotations

from typing import TYPE_CHECKING

from ....dataset import TabularDataset
from ....utils import get_relative_fn

if TYPE_CHECKING:
    import pandas as pd


class AdultDataset(TabularDataset):
    name = "adult"
    deps = {"table": ["train", "test"]}
    key_deps = ["train", "test"]

    folder_name = "adult"
    catalog = get_relative_fn("catalog.yml")

    def ingest(self, name, **tables: pd.DataFrame):
        df = super().ingest(name, **tables)
        df["income"] = df["income"].str.replace(".", "")  # test set lines end with . ?
        return df
