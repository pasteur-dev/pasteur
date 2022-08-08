from functools import reduce

import pandas as pd


def filter_by_keys(
    keys: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    # Sort to ensure consistent split every time
    # Dataframe should consist of up to 1 column (which is the key) or an index
    if keys.keys().empty:
        col = keys.index.name
    else:
        assert False, "Keys df should only have an index (0 columns)"
        # assert len.keys() == 1, "Keys df should only have one column"
        # col = keys.keys()[0]

    out = []
    for name, table in tables.items():
        idx = table.index.name
        new_table = table.reset_index(drop=not idx).merge(keys, on=col)
        if idx:
            new_table = new_table.set_index(idx)

        out[name] = table

    return out


class View:
    """A class for a View named <name> based on dataset <dataset> that creates
    a set of tables based on the provided dependencies, where here they are
    tables in the dataset provided.

    The set of tables is `deps.keys()`. It will be based on tables `set(deps.values())`.

    If used with kedro, the pipeline will look for the following dataset tables:
    `<dataset>@<table>`.

    Then, it will produce tables in the following format: `<name>.<table>`.

    For decoding a particular view, it may be required to decode the tables in
    a particular order. `trn_deps` defines that order. It needs to be static,
    so it can't be placed in `parameters.yml`"""

    name: str = None
    dataset: str = None
    deps: dict[str, list[str]] = None
    trn_deps: dict[str, list[str]] | None = None
    tabular: bool = False

    def __init__(self, **_) -> None:
        pass

    @property
    def dataset_tables(self):
        return list(dict.fromkeys(reduce(lambda a, b: a + b, self.deps.values(), [])))

    @property
    def tables(self):
        return list(self.deps.keys())

    def ingest(self, name, **tables: pd.DataFrame):
        """Creates the table <name> using the tables provided based on the dependencies."""
        assert False, "Unimplemented"

    def ingest_closure(self, name):
        """Wraps ingest function to include the table name."""
        return lambda **tables: self.ingest(name, **tables)

    def filter(self, keys: pd.DataFrame, **tables):
        return filter_by_keys(keys, tables)


class TabularView(View):
    deps = {"table": ["table"]}
    tabular: bool = True

    def ingest(self, name, **tables: pd.DataFrame):
        assert name == "table"
        return tables["table"]
