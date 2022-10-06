from __future__ import annotations
from typing import TYPE_CHECKING


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

    @Warning: having a table named raw is not allowed. Both dependencies and name
    have to be statically defined class variables."""

    name = None
    deps: dict[str, list[str]] = None
    key_deps: list[str] = None

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
        assert False, "Unimplemented"

    def keys(
        self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        assert False, "Unimplemented"

    def keys_filtered(
        self,
        req_splits: list[str],
        ratios: dict[str, float],
        random_state: int,
        **tables: pd.DataFrame,
    ):
        splits = self.keys(ratios, random_state, **tables)
        return {name: split for name, split in splits.items() if name in req_splits}


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
        from .utils import split_keys
        import pandas as pd

        df = pd.concat([tables[name] for name in self.deps["table"]])
        df.index.name = "id"
        return split_keys(df, ratios, random_state)
