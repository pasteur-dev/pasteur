from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .module import Module
from .utils import LazyChunk, LazyFrame, to_chunked

if TYPE_CHECKING:
    import pandas as pd


class Dataset(Module):
    """A class for a Dataset named `name` that creates a set of tables based on the
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

    If `catalog` is provided and is a path, it will be loaded with replacing
    `#{location}` with the location of the dataset (if `folder_name` has been provided
    that can be used to substitute the relative paths.
    Use `utils.get_relative_fn()` to specify the path. Useful for packaging.
    If it's a dict, it will be loaded straight away.
    `#{bootstrap}` will also be provided for defining intermediary datasets not
    present in the original download.

    For compressed datasets or ones that require preprocessing, `bootstrap` can be
    implemented as a callable which receives 2 strings, one for the raw location
    of the dataset (based on `folder_name`), and one for a reserved intermediary
    directory for this dataset (`#{bootstrap}`).

    @Warning: having a table named raw is not allowed."""

    deps: dict[str, list[str]] = {}
    key_deps: list[str] = []

    folder_name: str | None = None
    catalog: dict[str, Any] | str | None = None
    bootstrap: Callable[[str, str], None] | None = None

    def __init__(self, **_) -> None:
        pass

    @property
    def raw_tables(self):
        from functools import reduce

        return list(dict.fromkeys(reduce(lambda a, b: a + b, self.deps.values(), [])))

    @property
    def tables(self):
        return list(self.deps.keys())

    def ingest(self, name, **tables: Any) -> LazyFrame:
        """Creates the table <name> using the tables provided based on the dependencies.

        The dependencies may be any and should be defined in the catalog.
        The raw tables of a dataset are the only kedro datasets explicitly
        defined by the user.

        Can return a dataframe, callable which produces a dataframe, or dict of callables, dataframes.
        If it's a dict, the table will be partitioned using the dict keys.

        @warning: all partitioned tables should have the same partitions.
        Some tables may not be partitioned."""
        raise NotImplemented()

    def keys(self, **tables: LazyFrame) -> pd.DataFrame:
        """Returns a set of keys which split the current dataset (or partition).

        Keys do not need to be unique per partition, since splitting will also
        be partition based.

        Gets a set of table partitions based on `key_deps`. All tables are the
        same partition. If a table is not partitioned, it's the whole DataFrame.

        Shouldn't return a callable."""
        raise NotImplemented()

    def __str__(self) -> str:
        return self.name


class TypedDataset(Dataset):
    """Extend from to create an intermediary step in ingestion, where the table
    is loaded from `<dataset>.raw@<table>` to a parquet one `<dataset>.typed.<table>.

    Useful for multiple reads to raw tables. You can also override the `type()` function to make
    minor changes to the dataset. By default it's the identity.

    Since parquet files don't support chunked loading it's unused."""

    def type(self, table: Any):
        return table


class TabularDataset(Dataset):
    """Boilerplate for a tabular dataset. Assumes the dataset contains one table
    named `table`, the index of which is the keys.

    By default, assumes one raw table `raw@table`, exists. If the data is concatenated
    from multiple sources, `deps` can be modified to reflect this."""

    deps = {"table": ["table"]}
    key_deps = ["table"]

    def _process_chunk(self, tables: dict[str, pd.DataFrame]):
        return pd.concat(list(tables.values()))

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        assert name == "table"
        df = self._process_chunk(
            {name: table() for name, table in tables.items()}
        ).reset_index(drop=True)
        df.index = df.index.astype("int64").rename("id")
        return df

    @to_chunked
    def keys(self, **tables: LazyChunk) -> pd.DataFrame:
        """Returns a DataFrame containing only the index column of table "table"."""
        assert len(tables) == 1 and "table" in tables

        return tables["table"]()


__all__ = ["Dataset", "TabularDataset"]
