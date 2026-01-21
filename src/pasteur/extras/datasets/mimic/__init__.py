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
    pd_keys = pd.DataFrame(index=keys)
    del keys

    for chunk in table(chunksize=chunksize):
        c = chunk.join(pd_keys, on="subject_id", how="inner")

        # # Fix poe id
        # if name == 'pharmacy':
        #     c['poe_seq'] = c['poe_id'].str[1].astype('Int16')
        #     c = c.drop(columns=['poe_id'])

        yield c


def _partition_table(
    name: str, table: Callable, patients: LazyFrame, n_partition: int, chunksize: int
):
    # Deterministic loading = all tables have the same split
    keys = patients(["subject_id"]).index.to_numpy()
    partitions = np.array_split(keys, n_partition)

    return {
        str(i): gen_closure(_split_table, name, chunksize, part, table)
        for i, part in enumerate(partitions)
    }


class MimicDataset(Dataset):
    def __init__(self, n_partitions: int = 5, **_) -> None:
        super().__init__(**_)
        self._n_partitions = n_partitions

    # These tables do not have a subject_id, they are taken as is
    _mimic_tables_unpartitioned = [
        "d_hcpcs",
        "d_icd_diagnoses",
        "d_icd_procedures",
        "d_labitems",
        "icu_d_items",
    ]

    _mimic_tables_single = [
        "patients",
        "transfers",
        "admissions",
        "diagnoses_icd",
        "drgcodes",
        "hcpcsevents",
        "microbiologyevents",
        "poe_detail",
        "procedures_icd",
        "services",
        "icu_datetimeevents",
        "icu_icustays",
        "icu_outputevents",
        "icu_procedureevents",
    ]

    _mimic_tables_partitioned = {
        "emar": 4_000_000,
        "emar_detail": 4_000_000,
        "labevents": 4_000_000,
        "pharmacy": 4_000_000,
        "poe":  4_000_000,
        "prescriptions": 4_000_000,
        "icu_chartevents": 4_000_000,
        "icu_inputevents": 4_000_000,
    }
    _n_partitions = 5

    name = "mimic"
    deps = {
        **{t: [t] for t in _mimic_tables_unpartitioned},
        **{t: [t, "patients"] for t in _mimic_tables_single},
        **{t: [t, "patients"] for t in _mimic_tables_partitioned},
    }
    key_deps = ["patients"]

    folder_name = "mimiciv_2_2"
    catalog = get_relative_fn("catalog.yml")

    def ingest(self, name, **tables: LazyFrame | Callable[[], TextFileReader]):
        if name in self._mimic_tables_unpartitioned:
            return cast("LazyFrame", tables[name])
        if name in self._mimic_tables_single:
            return _partition_table(
                name,
                cast("Callable[[], TextFileReader]", tables[name]),
                cast("LazyFrame", tables["patients"]),
                self._n_partitions,
                1_000_000,
            )
        if name in self._mimic_tables_partitioned:
            chunksize = self._mimic_tables_partitioned[name]
            return _partition_table(
                name,
                cast("Callable[[], TextFileReader]", tables[name]),
                cast("LazyFrame", tables["patients"]),
                self._n_partitions,
                chunksize,
            )

    @to_chunked
    def keys(self, **tables: LazyChunk):
        return tables["patients"]()[[]]
