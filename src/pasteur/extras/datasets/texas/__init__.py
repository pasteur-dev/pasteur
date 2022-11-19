import logging

import pandas as pd

from ....dataset import Dataset, split_keys
from ....utils import get_relative_fn
from ....utils.progress import piter

logger = logging.getLogger(__name__)

class TexasDataset(Dataset):
    name = "texas"
    deps = {"table": ["pudf_1q2006"]}  # {t: [t] for t in []}
    key_deps = ["pudf_1q2006"]

    folder_name = "texas"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        import os
        from zipfile import ZipFile

        os.makedirs(bootstrap, exist_ok=True)
        for fn in piter(os.listdir(location), leave=False):
            if not (fn.endswith("_tab") or fn.endswith("-tab-delimited")):
                continue

            with ZipFile(os.path.join(location, fn), "r") as zf:
                logger.info(f"Extracting {fn}...")
                zf.extractall(bootstrap)

    def ingest(self, name, **tables: pd.DataFrame):
        return tables[name]

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables[""], ratios, random_state)
