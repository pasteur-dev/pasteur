from __future__ import annotations

from typing import TYPE_CHECKING

from ....dataset import TabularDataset
from ....utils import get_relative_fn, RawSource

import logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class AdultDataset(TabularDataset):
    name = "adult"
    deps = {"table": ["train", "test"]}

    folder_name = "adult"
    catalog = get_relative_fn("catalog.yml")

    raw_sources = RawSource(["https://archive.ics.uci.edu/static/public/2/adult.zip"])

    def bootstrap(self, raw: str, dst: str):
        from zipfile import ZipFile
        import os

        os.makedirs(dst, exist_ok=True)

        with ZipFile(os.path.join(raw, "adult.zip"), "r") as zf:
            logger.info(f"Extracting adult.zip...")
            zf.extractall(dst)

    def _process_chunk(self, tables: dict[str, pd.DataFrame]):
        import pandas as pd

        train = tables["train"]
        test = tables["test"].assign(
            income=tables["test"]["income"].cat.rename_categories(
                {"<=50K.": "<=50K", ">50K.": ">50K"}
            )
        )
        return pd.concat([train, test]).astype({"native-country": "category"})
