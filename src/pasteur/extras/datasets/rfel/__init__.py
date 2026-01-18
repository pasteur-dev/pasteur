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
        self, short_name: str, name: str, db: str, tables: dict[str, list[str]], **kwargs
    ) -> None:
        from itertools import chain

        self.name = f"rfel_{short_name}"
        self.key_deps = [next(iter(tables))]
        self.deps = tables

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

    def _process_chunk(self, tables: dict[str, "pd.DataFrame"]):
        assert len(tables) == 1
        return next(iter(tables.values()))

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        return self._process_chunk({name: table() for name, table in tables.items()})
    
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
        super().__init__(
            short_name="ce",
            name="consumer_expenditures",
            db="ConsumerExpenditures",
            tables=tables,
            **kwargs,
        )


class StudentLoanDataset(RfelDataset):
    def __init__(self, **kwargs) -> None:
        tables = {
            k.lower(): [k]
            for k in [
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
        }
        super().__init__(
            short_name="sl",
            name="student_loan",
            db="Student_loan",
            tables=tables,
            **kwargs,
        )
