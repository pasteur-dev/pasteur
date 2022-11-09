from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import pandas as pd



class Dataset:
    """A class for a Dataset named <name> that creates a set of tables based on the
    provided dependencies.

    The set of tables is `deps.keys()`. It will be based on raw tables `set(deps.values())`.

    If used with kedro, the pipeline will look for the following raw tables:
    `<name>.raw@<raw_table>` (using transcoding to make the raw dataset appear as one
    node in `kedro viz`).

    Then, it will produce tables in the following format: `<name>.<table>`.

    If `folder_name` is provided, Pasteur will then check that 
    `#{<folder_name>_location}` has been provided and exists as a path.
    If not, it will check that `${raw_location}/<folder_name>` exists.
    If neither are the case, then the dataset will not be loaded and a warning will be logged.

    If `catalog_fn` is provided, it will be loaded with the additional
    parameter `#{<folder_name>_location}` (if `folder_name` has been provided
    that can be used to substitute the relative paths.
    Use `utils.get_relative_fn()` to specify configs. Useful for packaging.

    @Warning: having a table named raw is not allowed."""

    name = None
    deps: dict[str, list[str]] = None
    key_deps: list[str] = None

    folder_name: str | None = None
    catalog_fn: str | None = None

    def __init__(self, **_) -> None:
        pass

    @property
    def raw_tables(self):
        from functools import reduce

        return list(
            dict.fromkeys(reduce(lambda a, b: a + b, self.dependencies.values(), []))
        )

    @property
    def tables(self):
        return list(self.deps.keys())

    def ingest(self, name, **tables: pd.DataFrame):
        """Creates the table <name> using the tables provided based on the dependencies."""
        raise NotImplemented()

    def keys(
        self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        raise NotImplemented()

    def keys_filtered(
        self,
        req_splits: list[str],
        ratios: dict[str, float],
        random_state: int,
        **tables: pd.DataFrame,
    ):
        splits = self.keys(ratios, random_state, **tables)
        return {name: split for name, split in splits.items() if name in req_splits}

    def __str__(self) -> str:
        return self.name


class TabularDataset(Dataset):
    deps = {"table": ["table"]}
    key_deps = ["table"]

    def ingest(self, name, **tables: pd.DataFrame):
        import pandas as pd

        assert name == "table"
        df = pd.concat([tables[name] for name in self.deps["table"]])
        df.index.name = "id"
        return df

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        import pandas as pd

        from .utils import split_keys

        df = pd.concat([tables[name] for name in self.deps["table"]])
        df.index.name = "id"
        return split_keys(df, ratios, random_state)


def split_keys(
    keys: pd.DataFrame, split: dict[str, any], random_state: int | None = None
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits keys according to the split dictionary.

    Example: split = {"dev": 0.3, "wrk": 0.3}
    Returns {"dev": 0 col Dataframe, "wrk" 0 col Dataframe}
    """
    from math import floor

    import numpy as np
    import pandas as pd

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


__all__ = [
    "Dataset",
    "TabularDataset",
    "split_keys",
    "get_relative_fn",
]
