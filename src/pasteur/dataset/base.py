from functools import reduce
from math import floor

import pandas as pd


def split_keys(
    keys: pd.DataFrame, split: dict[str, any], random_state: int | None = None
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits keys according to the split dictionary.

    Example: split = {"dev": 0.3, "wrk": 0.3}
    Returns {"dev": 0 col Dataframe, "wrk" 0 col Dataframe}
    """

    # Sort to ensure consistent split every time
    # Dataframe should consist of one column that is the index
    if keys.keys().empty:
        # If DataFrame is empty assume index is key
        assert keys.index.name, "No index column available"
    elif keys.index.name:
        # If index has a name, assume it is the key and drop other columns
        keys = keys[[]]
    else:
        # Otherwise, pick first column as index and drop the others
        keys.set_index(keys.columns[0])[[]]

    keys = keys.sort_values(by=keys.index.name)

    assert sum(split.values()) <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    ns = {name: floor(ratio * n_all) for name, ratio in split.items()}
    assert sum(ns.values()) <= n_all, "Sizes exceed dataset size"

    # TODO: check if using the same random state is valid.
    return {name: keys.sample(n=n, random_state=random_state) for name, n in ns.items()}


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
        def keys_fun(
            ratios: dict[str, float], random_state: int, **tables: pd.DataFrame
        ):
            splits = self.keys(ratios, random_state, **tables)
            return {name: split for name, split in splits.items() if name in req_splits}

        return keys_fun


class TabularDataset(Dataset):
    deps = {"table": ["table"]}
    key_deps = ["table"]

    def ingest(self, name, **tables: pd.DataFrame):
        assert name == "table"
        df = tables["table"].copy()
        df.index.name = "id"
        return {"table": df}

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables["table"], ratios, random_state)
