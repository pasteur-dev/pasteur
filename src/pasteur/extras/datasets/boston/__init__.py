from __future__ import annotations

from typing import TYPE_CHECKING

from ....dataset import TabularDataset
from ....utils import get_relative_fn, RawSource

import logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DSFN = "boston-housing-dataset"


class BostonDataset(TabularDataset):
    name = "boston"
    deps = {"table": ["table"]}

    folder_name = "boston"
    catalog = get_relative_fn("catalog.yml")

    raw_sources = RawSource(
        [
            "https://www.kaggle.com/api/v1/datasets/download/altavish/boston-housing-dataset",
        ]
    )

    def bootstrap(self, raw: str, dst: str):
        from zipfile import ZipFile
        import os

        os.makedirs(dst, exist_ok=True)

        with ZipFile(os.path.join(raw, DSFN), "r") as zf:
            logger.info(f"Extracting {DSFN}...")
            zf.extractall(dst)
