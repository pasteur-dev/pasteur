from functools import reduce
from math import floor

import pandas as pd
import numpy as np


def split_keys(
    keys: pd.DataFrame, split: dict[str, any], random_state: int | None = None
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits keys according to the split dictionary.

    Example: split = {"dev": 0.3, "wrk": 0.3}
    Returns {"dev": 0 col Dataframe, "wrk" 0 col Dataframe}
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Sort to ensure consistent split every time
    # Dataframe should consist of one column that is the index
    if keys.keys().empty:
        # If DataFrame is empty assume index is key
        assert keys.index.name, "No index column available"
        idx_name = None
        idx = keys.index
    elif keys.index.name:
        # If index has a name, assume it is the key and drop other columns
        idx_name = keys.index.name
        idx = keys.index
    else:
        # Otherwise, pick first column as index and drop the others
        idx_name = keys.columns[0]
        idx = keys[idx_name]

    assert sum(split.values()) <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    ns = {name: floor(ratio * n_all) for name, ratio in split.items()}
    assert sum(ns.values()) <= n_all, "Sizes exceed dataset size"

    # Sort and shuffle array for a consistent split every time
    keys = np.sort(idx)
    np.random.shuffle(keys)

    # Split array into the required chunks
    splits = {}
    i = 0
    for name, n in ns.items():
        split_keys = keys[i : i + n]
        i += n
        splits[name] = pd.DataFrame(index=split_keys)
        if idx_name is not None:
            splits[name].index.name = idx_name

    return splits


class Dataset:
    """A class for a Dataset named <name> that creates a set of tables based on the
    provided dependencies.

    The set of tables is `deps.keys()`. It will be based on raw tables `set(deps.values())`.

    If used with kedro, the pipeline will look for the following raw tables:
    `<name>.raw@<raw_table>` (using transcoding to make the raw dataset appear as one
    node in `kedro viz`).

    Then, it will produce tables in the following format: `<name>.<table>`.

    @Warning: having a table named raw is not allowed. Both dependencies and name
    have to be statically defined class variables."""

    name = None
    deps: dict[str, list[str]] = None
    key_deps: list[str] = None

    def __init__(self, **_) -> None:
        pass

    @property
    def raw_tables(self):
        return list(
            dict.fromkeys(reduce(lambda a, b: a + b, self.dependencies.values(), []))
        )

    @property
    def tables(self):
        return list(self.deps.keys())

    def ingest(self, name, **tables: pd.DataFrame):
        """Creates the table <name> using the tables provided based on the dependencies."""
        assert False, "Unimplemented"

    def ingest_closure(self, name):
        """Wraps ingest function to include the table name."""
        fun = lambda **tables: self.ingest(name, **tables)
        fun.__name__ = f"ingest_{name}"
        return fun

    def keys(
        self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        assert False, "Unimplemented"

    def keys_closure(self, req_splits: list[str]):
        def gen_keys(
            ratios: dict[str, float], random_state: int, **tables: pd.DataFrame
        ):
            splits = self.keys(ratios, random_state, **tables)
            return {name: split for name, split in splits.items() if name in req_splits}

        return gen_keys


class TabularDataset(Dataset):
    deps = {"table": ["table"]}
    key_deps = ["table"]

    def ingest(self, name, **tables: pd.DataFrame):
        assert name == "table"
        df = tables["table"].copy()
        df.index.name = "id"
        return df

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        df = tables["table"].copy()
        df.index.name = "id"
        return split_keys(df, ratios, random_state)
