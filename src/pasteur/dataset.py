from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .module import Module
from .utils import LazyFrame

if TYPE_CHECKING:
    import pandas as pd


class Dataset(Module):
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

    If `catalog` is provided and is a path, it will be loaded with replacing
    `#{location}` with the location of the dataset (if `folder_name` has been provided
    that can be used to substitute the relative paths.
    Use `utils.get_relative_fn()` to specify the path. Useful for packaging.
    If it's a dict, it will be loaded straight away.
    `#{bootstrap}` will also be provided for defining intermediary datasets not
    present in the original download.

    For compressed datasets or ones that require preprocessing, `bootstrap` can be
    implemented as a callable which receives 2 strings, one for the base location
    of the dataset (based on folder name), and one for a reserved intermediary
    directory.

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

    def ingest(
        self, name, **tables: pd.DataFrame | LazyFrame
    ) -> pd.DataFrame | LazyFrame:
        """Creates the table <name> using the tables provided based on the dependencies."""
        raise NotImplemented()

    def keys(self, **tables: pd.DataFrame | LazyFrame) -> pd.DataFrame:
        raise NotImplemented()

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

    def keys(self, **tables: pd.DataFrame):
        import pandas as pd

        df = pd.concat([tables[name] for name in self.key_deps]).reset_index(drop=True)
        df.index.name = "id"
        return df


__all__ = [
    "Dataset",
    "TabularDataset",
]
