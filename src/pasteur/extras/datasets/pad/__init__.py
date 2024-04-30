from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pandas as pd

from ....dataset import Dataset
from ....utils import (
    LazyChunk,
    LazyFrame,
    gen_closure,
    get_relative_fn,
    to_chunked,
)

import logging

logger = logging.getLogger(__name__)


class PadDataset(Dataset):
    def __init__(self, n_partitions: int = 5, **_) -> None:
        super().__init__(**_)
        self._n_partitions = n_partitions

    name = "pad"
    deps = {}
    key_deps = ["pad1"]

    folder_name = "pad"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        import os

        os.makedirs(bootstrap, exist_ok=True)

        logger.info("Using pyreadr to extract dataframes.")
        import pyreadr

        dfs = pyreadr.read_r(os.path.join(location, "pad.rda"))
        for k, v in dfs.items():
            out_fn = os.path.join(bootstrap, f"{k}.csv")
            logger.info(f"Writing '{k}' to '{out_fn}'")
            v.to_csv(out_fn, index=False)

    def ingest(self, name, **tables: LazyFrame | Callable[[],]):
        pass

    @to_chunked
    def keys(self, **tables: LazyChunk):
        return tables["pad1"]()[[]]
