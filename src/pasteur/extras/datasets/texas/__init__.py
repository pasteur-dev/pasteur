import pandas as pd

from ....dataset import Dataset, split_keys
from ....utils import get_relative_fn


class TexasDataset(Dataset):
    name = "texas"
    deps = {"table": ["pudf_1q2006"]}  # {t: [t] for t in []}
    key_deps = ["pudf_1q2006"]

    folder_name = "texas"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        print(location, bootstrap)
        assert False

    def ingest(self, name, **tables: pd.DataFrame):
        return tables[name]

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables[""], ratios, random_state)
