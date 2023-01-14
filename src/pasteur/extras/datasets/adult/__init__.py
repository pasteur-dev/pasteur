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

    def _process_chunk(self, tables: dict[str, pd.DataFrame]):
        import pandas as pd
        train = tables["train"]
        test = tables["test"].assign(
            income=tables["test"]["income"].cat.rename_categories(
                {"<=50K.": "<=50K", ">50K.": ">50K"}
            )
        )
        return pd.concat([train, test]).astype({'native-country': 'category'})
