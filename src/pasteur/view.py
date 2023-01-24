from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from .module import Module
from .utils import LazyFrame, LazyChunk, to_chunked
from .utils.progress import process_in_parallel

if TYPE_CHECKING:
    import pandas as pd


@to_chunked
def split_keys(
    key_chunk: LazyChunk,
    req_splits: list[str] | None,
    splits: dict[str, Any],
    random_state: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Splits keys according to the split dictionary.

    Example: split = {"dev": 0.3, "wrk": 0.3}
    Returns {"dev": 0 col Dataframe, "wrk" 0 col Dataframe}
    """
    from math import floor

    import numpy as np
    import pandas as pd

    keys = key_chunk()

    if random_state is not None:
        np.random.seed(random_state)

    # Sort to ensure consistent split every time
    # Dataframe should consist of one column that is the index
    if keys.keys().empty:
        # If DataFrame is empty assume index is key
        assert keys.index.name, "No index column available"
        idx_name = keys.index.name
        idx = keys.index
    elif keys.index.name:
        # If index has a name, assume it is the key and drop other columns
        idx_name = keys.index.name
        idx = keys.index
    else:
        # Otherwise, pick first column as index and drop the others
        idx_name = keys.columns[0]
        idx = keys[idx_name]

    assert sum(splits.values()) <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    ns = {name: floor(ratio * n_all) for name, ratio in splits.items()}
    assert sum(ns.values()) <= n_all, "Sizes exceed dataset size"

    # Sort and shuffle array for a consistent split every time
    np_keys = np.sort(idx)
    np.random.shuffle(np_keys)

    # Split array into the required chunks
    splits = {}
    i = 0
    for name, n in ns.items():
        keys_split = np_keys[i : i + n]
        i += n
        splits[name] = pd.DataFrame(index=keys_split)
        if idx_name is not None:
            splits[name].index.name = idx_name

    if req_splits:
        return {name: splits[name] for name in req_splits}
    return splits


@to_chunked
def filter_by_keys(key_chunk: LazyChunk, table_chunk: LazyChunk, drop_index: bool = False) -> pd.DataFrame:
    # Sort to ensure consistent split every time
    # Dataframe should consist of up to 1 column (which is the key) or an index

    keys = key_chunk()
    table = table_chunk()

    assert keys.keys().empty, "Keys df should only have an index (0 columns)"

    col = keys.index.name
    idx = table.index.name
    if idx == col:
        # Assume if index of table is from keys we can index it
        return table.loc[keys.index]
    else:
        mask = table[col].isin(keys.index)
        del keys
        if drop_index:
            table = table.drop(columns=[col])
        return table.loc[mask]


def _runner(func):
    return func()


def filter_by_keys_merged(keys: LazyFrame, table: LazyFrame, reset_index: bool = False, drop_index: bool = False):
    import pandas as pd

    tasks = filter_by_keys(keys, table, drop_index=drop_index)

    res = process_in_parallel(_runner, [{"func": task} for task in tasks], desc="Filtering and merging...")  # type: ignore

    # Sort to ensure determinism
    res_dict = {}
    for d in res:
        for n, v in d.items():
            res_dict[n] = v
    data = pd.concat([res_dict[n] for n in sorted(res_dict)])  # type: ignore

    if reset_index:
        data = data.reset_index(drop=True).rename_axis("id")
    
    return data


class View(Module):
    """A class for a View named <name> based on dataset <dataset> that creates
    a set of tables based on the provided dependencies, where here they are
    tables in the dataset provided.

    The set of tables is `deps.keys()`. It will be based on tables `set(deps.values())`.

    If used with kedro, the pipeline will look for the following dataset tables:
    `<dataset>@<table>`.

    Then, it will produce tables in the following format: `<name>.<table>`.

    For decoding a particular view, it may be required to decode the tables in
    a particular order. `trn_deps` defines that order. It needs to be static,
    so it can't be placed in `parameters.yml`

    `parameters_fn`, if provided, will be used to load a parameters file with
    defaults for the view (such as metadata). Useful for packaging.
    Use `utils.get_relative_fn()` from datasets."""

    dataset: str
    deps: dict[str, list[str]] = {}
    trn_deps: dict[str, list[str]] = {}
    parameters: dict[str, Any] | str | None = None
    tabular: bool = False

    def __init__(self, **_) -> None:
        pass

    @property
    def dataset_tables(self):
        from functools import reduce

        return list(dict.fromkeys(reduce(lambda a, b: a + b, self.deps.values(), [])))

    @property
    def tables(self):
        return list(self.deps.keys())

    def ingest(self, name, **tables: LazyFrame):
        """Creates the table <name> using the tables provided based on the dependencies."""
        raise NotImplementedError()

    def split_keys(
        self,
        keys: LazyFrame,
        req_splits: list[str] | None,
        splits: dict[str, Any],
        random_state: int,
    ):
        """Takes the key frame and splits it into the portions specified by `splits`. Then, return
        the split with names in `req_splits`.

        Should produce the same results each run regardless of the value of `split`,
        because it will be ran once per split."""
        ...
        return split_keys(keys, req_splits, splits, random_state)

    def filter_table(self, name: str, keys: LazyFrame, **tables: LazyFrame):
        """Filters the table using the keys provided."""
        return filter_by_keys(keys, tables[name])

    def __str__(self) -> str:
        return self.name


class TabularView(View):
    deps = {"table": ["table"]}
    tabular: bool = True

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        assert name == "table"
        return tables["table"]()


__all__ = ["View", "TabularView", "filter_by_keys"]
