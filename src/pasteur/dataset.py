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
    """ Defines the Tables of the dataset and their dependencies, ex.:
    
    ```python
    {"table1": ["raw1", "raw2"], "table2": ["raw3", "raw4"]}
    ```
    """

    key_deps: list[str] = []
    """ Provides the table dependencies (Table, not raw) that are used to create 
    the keys of the dataset. """

    folder_name: str | None = None
    """ Specifies the name of the folder in the raw directory that will be used
    for the dataset's raw sources. If the folder does not exist, the dataset
    is disabled (used for packaging)."""
    catalog: dict[str, Any] | str | None = None
    """ A kedro catalog that represents the dataset's sources. Can be provided
    as a dictionary to be used as is, or as a filepath, in which case
    the path will be loaded and processed, by replacing the paths with appropriate
    ones based on the raw directory and folder name."""

    bootstrap: Callable[[str, str], None] | None = None
    """ An optional function that is used for one-time tasks (such as extraction).
    Can be run with `pasteur bootstrap <dataset_name>`. 
    
    Is provided with 2 paths: the raw directory of the dataset and another 
    directory dedicated to the dataset named bootstrap.
    If the dataset has any archives, extract them from the raw directory to 
    bootstrap and then use the bootstrap directory as a base in the catalog."""

    def __init__(self, **_) -> None:
        pass

    @property
    def raw_tables(self):
        """Returns the raw dependency names of the dataset."""
        from functools import reduce

        return list(dict.fromkeys(reduce(lambda a, b: a + b, self.deps.values(), [])))

    @property
    def tables(self):
        """Returns the table names of the dataset."""
        return list(self.deps.keys())

    def ingest(self, name, **tables: Any) -> LazyFrame:
        """Creates the table <name> using the tables provided based on the dependencies.

        The dependencies may be anything and should be defined in the catalog.
        The raw tables of a dataset are the only kedro datasets explicitly
        defined by the user.

        Can return a dataframe, callable which produces a dataframe, or dict of callables, dataframes.
        If it's a dict, the table will be partitioned using the dict keys.

        @warning: all partitioned tables should have the same partitions.
        Some tables may not be partitioned.
        
        Tip: use a `match` statement to fork based on table name to per-table functions."""
        raise NotImplemented()

    def keys(self, **tables: LazyFrame) -> pd.DataFrame:
        """Returns a set of keys which split the current dataset.

        Keys do not need to be unique per partition, since splitting will also
        be partition based.
        Gets a set of table partitions based on `key_deps`.

        Use the `to_chunked` operator to handle partitions."""
        raise NotImplemented()

    def __str__(self) -> str:
        return self.name


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
