#
# Datasets from
# https://relational.fel.cvut.cz/
#

from typing import TYPE_CHECKING

from ....dataset import Dataset
from ....utils import RawSource

import logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

rfel = "See https://relational.fel.cvut.cz/about for citation and license info."


class RfelDataset(Dataset):

    def __init__(
        self, short_name: str, name: str, db: str, tables: dict[str, str], **kwargs
    ) -> None:
        self.name = f"rfel_{short_name}"
        self.key_deps = [next(iter(tables))]
        self.deps = {k: [v] for k, v in tables.items()}

        self.folder_name = "rfel/" + name
        self.catalog = {
            t: {
                "type": "pandas.ParquetDataSet",
                "filepath": "${location}/" + t,
            } for t in tables
        }

        self.raw_sources = RawSource(
            f"relational.fel:{db}", self.folder_name, False, rfel
        )
        super().__init__(**kwargs)


class ConsumerExpendituresDataset(RfelDataset):
    def __init__(self, **kwargs) -> None:
        tables = {
            k.lower(): k
            for k in [
                "EXPENDITURES",
                "HOUSEHOLDS",
                "HOUSEHOLD_MEMBERS",
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
            k.lower(): k
            for k in [
                "bool",
                "disabled",
                "enlist",
                "enrolled",
                "filed_for_bankrupcy",
                "longest_absense_from_school",
                "male",
                "no_payment_due",
                "person",
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
