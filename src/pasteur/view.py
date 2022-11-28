from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .module import Module
from .utils import (
    LazyFrame,
    LazyChunk,
    gen_closure,
    is_partitioned,
    are_partitioned,
    is_lazy,
    get_data,
)

if TYPE_CHECKING:
    import pandas as pd


def split_keys(
    keys: pd.DataFrame, split: dict[str, Any], random_state: int | None = None
) -> dict[str, pd.DataFrame]:
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

    assert sum(split.values()) <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    ns = {name: floor(ratio * n_all) for name, ratio in split.items()}
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

    return splits


def filter_by_keys(keys: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    # Sort to ensure consistent split every time
    # Dataframe should consist of up to 1 column (which is the key) or an index
    if keys.keys().empty:
        col = keys.index.name
    else:
        assert False, "Keys df should only have an index (0 columns)"
        # assert len.keys() == 1, "Keys df should only have one column"
        # col = keys.keys()[0]

    idx = table.index.name
    # new_table = table.reset_index(drop=not idx).merge(keys, on=col, how="inner")
    new_table = table.reset_index(drop=not idx).join(keys, on=col, how="inner")

    if idx:
        new_table = new_table.set_index(idx)

    return new_table


def filter_by_keys_lazy(keys: pd.DataFrame, chunk: LazyChunk):
    return filter_by_keys(keys, get_data(chunk))


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

    def ingest(self, name, **tables: pd.DataFrame):
        """Creates the table <name> using the tables provided based on the dependencies."""
        raise NotImplementedError()

    def split_keys(
        self, keys: pd.DataFrame, split: dict[str, Any], random_state: int | None = None
    ):
        return split_keys(keys, split, random_state)

    def filter_tables(self, keys: pd.DataFrame, **tables: LazyFrame):
        new_tables = {}
        for name, table in tables.items():
            if is_partitioned(table):
                new_tables[name] = {
                    pid: gen_closure(filter_by_keys_lazy, keys, fun)
                    for pid, fun in table.items()
                }
            else:
                new_tables[name] = filter_by_keys(keys, table)
        return new_tables

    def __str__(self) -> str:
        return self.name


class TabularView(View):
    deps = {"table": ["table"]}
    tabular: bool = True

    def ingest(self, name, **tables: LazyFrame):
        assert name == "table"
        return tables["table"]


__all__ = ["View", "TabularView", "filter_by_keys"]
