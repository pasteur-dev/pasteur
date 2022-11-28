from __future__ import annotations

from typing import TYPE_CHECKING

from ....dataset import TabularDataset
from ....utils import get_relative_fn

if TYPE_CHECKING:
    import pandas as pd


class AdultDataset(TabularDataset):
    name = "adult"
    deps = {"table": ["train", "test"]}

    folder_name = "adult"
    catalog = get_relative_fn("catalog.yml")

    def _process_chunk(self, table: pd.DataFrame):
        df = table
        df["income"] = df["income"].str.replace(".", "")  # test set lines end with . ?
        return df
