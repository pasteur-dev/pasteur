from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pandas as pd

from ....dataset import Dataset
from ....utils import LazyChunk, LazyFrame, gen_closure, get_data, get_relative_fn

if TYPE_CHECKING:
    from pandas.io.parsers.readers import TextFileReader


def _split_table(keys: np.ndarray, table: "Callable[..., TextFileReader]"):
    pd_keys = pd.DataFrame(index=keys)
    del keys

    out = []
    for chunk in table():
        out.append(chunk.join(pd_keys, on="subject_id", how="inner"))
        del chunk
    
    return pd.concat(out)


def _partition_table(table: Callable, patients: LazyFrame, n_partition: int | None):
    # Deterministic loading = all tables have the same split
    keys = get_data(patients, ["subject_id"]).index.to_numpy()
    partitions = np.array_split(keys, n_partition or 5)

    return {
        str(i): gen_closure(_split_table, part, table)
        for i, part in enumerate(partitions)
    }


class MimicDataset(Dataset):
    def __init__(self, n_partition: int | None = None,  **_) -> None:
        super().__init__(**_)
        self.n_partition = n_partition

    _mimic_tables_single = [
        "core_patients",
        "core_transfers",
        "core_admissions",
        "hosp_d_hcpcs",
        "hosp_diagnoses_icd",
        "hosp_d_icd_diagnoses",
        "hosp_d_icd_procedures",
        "hosp_d_labitems",
        "hosp_drgcodes",
        "hosp_hcpcsevents",
        "hosp_labevents",
        "hosp_microbiologyevents",
        "hosp_pharmacy",
        "hosp_poe",
        "hosp_poe_detail",
        "hosp_prescriptions",
        "hosp_procedures_icd",
        "hosp_services",
        "icu_chartevents",
        "icu_datetimeevents",
        "icu_d_items",
        "icu_icustays",
        "icu_inputevents",
        "icu_outputevents",
        "icu_procedureevents",
    ]

    _mimic_tables_partitioned = [
        "hosp_emar",
        "hosp_emar_detail",
    ]

    name = "mimic"
    deps = {
        **{t: [t] for t in _mimic_tables_single},
        **{t: [t, "core_patients"] for t in _mimic_tables_partitioned},
    }
    key_deps = ["core_patients"]

    folder_name = "mimiciv_1_0"
    catalog = get_relative_fn("catalog.yml")

    def ingest(self, name, **tables: LazyFrame | Callable[..., TextFileReader]):
        if name in self._mimic_tables_single:
            return cast("LazyFrame", tables[name])
        if name in self._mimic_tables_partitioned:
            return _partition_table(
                cast("Callable[..., TextFileReader]", tables[name]),
                cast("LazyFrame", tables["core_patients"]),
                self.n_partition
            )

    def keys(self, **tables: LazyChunk):
        return get_data(tables["core_patients"], [])[[]]
