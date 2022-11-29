from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pandas as pd

from ....dataset import Dataset, TypedDataset
from ....utils import LazyChunk, LazyFrame, gen_closure, get_data, get_relative_fn

if TYPE_CHECKING:
    from pandas.io.parsers.readers import TextFileReader


def _split_table(
    chunksize: int, keys: np.ndarray, table: "Callable[..., TextFileReader]"
):
    pd_keys = pd.DataFrame(index=keys)
    del keys

    for chunk in table(chunksize=chunksize):
        yield chunk.join(pd_keys, on="subject_id", how="inner")


def _partition_table(
    table: Callable, patients: LazyFrame, n_partition: int, chunksize: int
):
    # Deterministic loading = all tables have the same split
    keys = get_data(patients, ["subject_id"]).index.to_numpy()
    partitions = np.array_split(keys, n_partition)

    return {
        str(i): gen_closure(_split_table, chunksize, part, table)
        for i, part in enumerate(partitions)
    }


class MimicDataset(Dataset):
    def __init__(self, **_) -> None:
        super().__init__(**_)

    _mimic_tables_single = [
        "core_patients",
        "core_transfers",
        "core_admissions",
        # "hosp_d_hcpcs",
        # "hosp_diagnoses_icd",
        # "hosp_d_icd_diagnoses",
        # "hosp_d_icd_procedures",
        # "hosp_d_labitems",
        # "hosp_drgcodes",
        # "hosp_hcpcsevents",
        # "hosp_microbiologyevents",
        # "hosp_poe_detail",
        # "hosp_procedures_icd",
        # "hosp_services",
        # "icu_datetimeevents",
        # "icu_d_items",
        # "icu_icustays",
        # "icu_outputevents",
        # "icu_procedureevents",
    ]

    _mimic_tables_partitioned = {
        # "hosp_emar": (5, 5_000_000),
        # "hosp_emar_detail": (5, 5_000_000),
        # "hosp_labevents": (20, 5_000_000),
        # "hosp_pharmacy": (5, 10_000_000),
        # "hosp_poe": (5, 10_000_000),
        "hosp_prescriptions": (5, 5_000_000),
        # "icu_chartevents": (30, 5_000_000),
        "icu_inputevents": (5, 5_000_000),
    }

    name = "mimic"
    deps = {
        **{t: [t] for t in _mimic_tables_single},
        **{t: [t, "core_patients"] for t in _mimic_tables_partitioned},
    }
    key_deps = ["core_patients"]

    folder_name = "mimiciv_1_0"
    catalog = get_relative_fn("catalog.yml")

    cache_typed = True

    def ingest(self, name, **tables: LazyFrame | Callable[..., TextFileReader]):
        if name in self._mimic_tables_single:
            return cast("LazyFrame", tables[name])
        if name in self._mimic_tables_partitioned:
            n_partitions, chunksize = self._mimic_tables_partitioned[name]
            return _partition_table(
                cast("Callable[..., TextFileReader]", tables[name]),
                cast("LazyFrame", tables["core_patients"]),
                n_partitions,
                chunksize,
            )

    def keys(self, **tables: LazyChunk):
        return get_data(tables["core_patients"], [])[[]]
