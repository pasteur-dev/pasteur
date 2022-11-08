def _gen_sdgym_tables(dir: str):
    import os
    from zipfile import ZipFile

    files = os.listdir(dir)
    names = {
        fn.split("_v")[0].split(".zip")[0].lower(): fn
        for fn in files
        if fn.endswith(".zip")
    }

    datasets = {}
    for ds_name, ds_fn in names.items():
        with ZipFile(os.path.join(dir, ds_fn), "r") as zip:
            ds_tables = []
            for file in zip.filelist:
                fn = file.filename
                if not fn.endswith(".csv"):
                    continue

                name = fn.split("/")[-1][:-4].lower()
                ds_tables.append(name)

            if len(ds_tables) == 1:
                ds_tables = [f"table"]

            datasets[f"sdgym_{ds_name}"] = ds_tables
    return datasets


def _reload_sdgym_tables(dir: str = "raw/sdgym", out_fn: str = "_sdgym.json"):
    import json 
    
    assert out_fn
    datasets = _gen_sdgym_tables(dir)
    with open(out_fn, "w") as f:
        json.dump(datasets, f)

if __name__ == "__main__":
    """If this script is ran as as a module, it updates the sdgym constants based on
    the currently downloaded datasets.
    
    The path to the sdgyms zips can be passed as a parameter.
    
    Kedro requires pipeline structures to be predefined, so tables and views have 
    to be hardcoded."""
    import sys
    import os

    if len(sys.argv) < 2:
        path = "raw/sdgym"
    else:
        path = sys.argv[1]
    
    curr_path = os.path.dirname(os.path.realpath(__file__))
    _reload_sdgym_tables(path, os.path.join(curr_path, "sdgym.json"))

import pandas as pd

from .base import Dataset


class SDGymDataset(Dataset):
    _tables = []

    name = None
    deps = {t: [] for t in _tables}
    key_deps = ["core_patients"]

    def ingest(self, name, **tables: pd.DataFrame):
        return tables[name]

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        from .utils import split_keys

        return split_keys(tables["core_patients"], ratios, random_state)
