"""
This file contains pipelines specific to processing the dataset MIMIC-IV.
"""

from typing import Optional, Set
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..general.pipeline import create_node_split_keys
from ..general.nodes import identity


def create_intermediate_data(inputs: Optional[Set] = None) -> Pipeline:
    mimic_tables_all = [
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

    if inputs:
        inputs_clear = {t.replace("mimic.", "").split("@")[0] for t in inputs}
        mimic_tables = inputs_clear.intersection(mimic_tables_all)
    else:
        mimic_tables = mimic_tables_all

    parquet_pipeline = pipeline(
        [
            node(
                func=identity,
                inputs=["mimic_iv@%s" % t],
                outputs="%s@all" % t,
                name="ingest_%s" % t,
            )
            for t in mimic_tables
        ]
    )
    return parquet_pipeline


def create_pipeline_split_mimic_keys() -> Pipeline:
    return modular_pipeline(
        pipeline([create_node_split_keys()]), inputs={"keys_all": "core_patients@keys"}
    )


def create_pipeline(inputs: Optional[Set] = None) -> Pipeline:
    mimic_keys = create_pipeline_split_mimic_keys()
    inputs = inputs.union(mimic_keys.inputs()) if inputs is not None else None
    mimic_ingest = create_intermediate_data(inputs)

    return modular_pipeline(
        mimic_keys + mimic_ingest,
        namespace="mimic",
        inputs=["mimic_iv"],
        parameters=["random_state"],
    )
