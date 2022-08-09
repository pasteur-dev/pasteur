import pandas as pd

from .base import Dataset, split_keys


class MimicDataset(Dataset):
    _mimic_tables_all = [
        "core_patients",
        "core_transfers",
        "core_admissions",
        "hosp_d_hcpcs",
        "hosp_diagnoses_icd",
        "hosp_d_icd_diagnoses",
        "hosp_d_icd_procedures",
        "hosp_d_labitems",
        "hosp_drgcodes",
        "hosp_emar",
        "hosp_emar_detail",
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

    name = "mimic"
    deps = {t: [t] for t in _mimic_tables_all}
    key_deps = ["core_patients"]

    def ingest(self, name, **tables: pd.DataFrame):
        return tables[name]

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables["core_patients"], ratios, random_state)
