#
# Datasets from
# https://relational.fel.cvut.cz/
#

import logging
from typing import TYPE_CHECKING

from ....dataset import Dataset
from ....utils import LazyChunk, RawSource, to_chunked

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

rfel = "See https://relational.fel.cvut.cz/about for citation and license info."


class RfelDataset(Dataset):

    def __init__(
        self,
        short_name: str,
        name: str,
        db: str,
        tables: dict[str, list[str]],
        keys: None | str | dict[str, str | list[str]] = None,
        **kwargs,
    ) -> None:
        from itertools import chain

        self.name = f"rfel_{short_name}"
        self.key_deps = [next(iter(tables))]
        self.deps = tables
        self._keys = keys

        self.folder_name = "rfel/" + name
        self.catalog = {
            t: {
                "type": "pasteur.kedro.dataset.AutoDataset",
                "filepath": "${location}/" + t + ".pq",
            }
            for t in chain.from_iterable(tables.values())
        }

        self.raw_sources = RawSource(
            f"relational.fel:{db}", self.folder_name, False, rfel
        )
        super().__init__(**kwargs)

    def _process_chunk(self, name, tables: dict[str, "pd.DataFrame"]):
        assert len(tables) == 1
        df = next(iter(tables.values()))

        if self._keys is not None:
            if isinstance(self._keys, str):
                key = self._keys
            elif name in self._keys:
                key = self._keys[name]
            else:
                key = None
            if key is not None:
                df = df.set_index(key)

        # Make all columns lower case
        df.columns = df.columns.str.lower()
        if df.index is not None and df.index.names is not None:
            df.index.names = [
                n.lower() if n is not None else None for n in df.index.names
            ]

        return df

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        return self._process_chunk(name, {n: table() for n, table in tables.items()})

    def keys(self, **tables: LazyChunk) -> "pd.DataFrame":
        """Returns a DataFrame containing only the index column of the first table."""
        assert len(tables) == 1

        return next(iter(tables.values()))().index.to_frame()


class ConsumerExpendituresDataset(RfelDataset):
    def __init__(self, **kwargs) -> None:
        tables = {
            k.lower(): [k]
            for k in [
                "HOUSEHOLDS",
                "HOUSEHOLD_MEMBERS",
                "EXPENDITURES",
            ]
        }
        keys = {
            "households": "HOUSEHOLD_ID",
            "expenditures": "EXPENDITURE_ID",
        }
        super().__init__(
            short_name="ce",
            name="consumer_expenditures",
            db="ConsumerExpenditures",
            keys=keys,
            tables=tables,
            **kwargs,
        )

    def keys(self, **tables: LazyChunk) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(
            index=(
                tables["households"]()
                .index
                .unique()
                .astype(pd.Int64Dtype())
            )
        )


class StudentLoanDataset(RfelDataset):
    def __init__(self, **kwargs) -> None:
        all_tables = [
            "person",
            "bool",
            "disabled",
            "enlist",
            "enrolled",
            "filed_for_bankrupcy",
            "longest_absense_from_school",
            "male",
            "no_payment_due",
            "unemployed",
        ]

        tables = {k.lower(): [k] for k in all_tables}

        keys = {
            **{t.lower(): "name" for t in all_tables},
            "enlist": None,
            "enrolled": None,
        }

        super().__init__(
            short_name="sl",
            name="student_loan",
            db="Student_loan",
            tables=tables,
            keys=keys,
            **kwargs,
        )

    def keys(self, **tables: LazyChunk) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(
            index=(
                tables["person"]()
                .index
                .str.replace("student", "")
                .unique()
                .astype(pd.Int64Dtype())
            )
        )
