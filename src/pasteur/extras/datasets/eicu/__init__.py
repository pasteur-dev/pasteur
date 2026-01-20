from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pandas as pd

from ....dataset import Dataset
from ....utils import (
    LazyChunk,
    LazyFrame,
    gen_closure,
    get_relative_fn,
    to_chunked,
)

if TYPE_CHECKING:
    from pandas.io.parsers.readers import TextFileReader


def _split_table(
    name: str, chunksize: int, keys: np.ndarray, table: "Callable[..., TextFileReader]"
):
    key_index = pd.Index(keys)
    del keys

    for chunk in table(chunksize=chunksize):
        if "patientunitstayid" not in chunk.columns:
            continue

        col = chunk["patientunitstayid"]
        try:
            key_index = key_index.astype(col.dtype)
        except (TypeError, ValueError):
            pass

        c = chunk[col.isin(key_index)]

        if c.shape[0] > 0:
            yield c


def _partition_table(
    name: str, table: Callable, patient: LazyFrame, n_partition: int, chunksize: int
):
    # Deterministic loading = all tables have the same split
    keys = patient(["patientunitstayid"]).index.to_numpy()
    partitions = np.array_split(keys, n_partition)

    return {
        str(i): gen_closure(_split_table, name, chunksize, part, table)
        for i, part in enumerate(partitions)
    }


class EicuDataset(Dataset):
    def __init__(self, n_partitions: int = 5, **_) -> None:
        super().__init__(**_)
        self._n_partitions = n_partitions

    _eicu_tables_single = [
        "pasthistory",
        "admissiondrug",
        "respiratorycare",
        "admissiondx",
        "careplancareprovider",
        "careplangoal",
        "apachepatientresult",
        "allergy",
        "apacheapsvar",
        "apachepredvar",
        "microlab",
        "patient",
        "careplaninfectiousdisease",
        "careplaneol",
        "customlab",
        "hospital",
    ]

    _eicu_tables_partitioned = {
        # "nursecharting": 2_000_000,
        # "vitalperiodic": 2_000_000,
        "lab": 2_000_000,
        "vitalaperiodic": 2_000_000,
        # "respiratorycharting": 2_000_000,
        "nurseassessment": 2_000_000,
        "intakeoutput": 1_000_000,
        "medication": 1_000_000,
        "nursecare": 2_000_000,
        "physicalexam": 2_000_000,
        "infusiondrug": 2_000_000,
        "treatment": 2_000_000,
        "careplangeneral": 2_000_000,
        "diagnosis": 2_000_000,
        "note": 2_000_000,
    }
    _n_partitions = 5

    name = "eicu"
    deps = {
        **{t: [t] for t in _eicu_tables_single},
        **{t: [t, "patient"] for t in _eicu_tables_partitioned},
    }
    key_deps = ["patient"]

    folder_name = "eicu_2_0"
    catalog = get_relative_fn("catalog.yml")

    def ingest(self, name, **tables: LazyFrame | Callable[[], TextFileReader]):
        if name in self._eicu_tables_single:
            return cast("LazyFrame", tables[name])
        if name in self._eicu_tables_partitioned:
            chunksize = self._eicu_tables_partitioned[name]
            return _partition_table(
                name,
                cast("Callable[[], TextFileReader]", tables[name]),
                cast("LazyFrame", tables["patient"]),
                self._n_partitions,
                chunksize,
            )

    @to_chunked
    def keys(self, **tables: LazyChunk):
        return tables["patient"]()[[]]
