"""
This file contains pipelines specific to processing the dataset MIMIC-IV.
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..general.pipeline import create_pipeline_split_keys
from ..general.nodes import identity


def create_pipeline_split_mimic_keys(**kwargs) -> Pipeline:
    return modular_pipeline(
        create_pipeline_split_keys(),
        namespace="keys",
        parameters={"split_ratios": "ratios", "random_state": "random_state"},
    )


def create_intermediate_data(**kwargs) -> Pipeline:
    mimic_tables = [
        "core.patients",
        "core.transfers",
        "core.admissions",
        "hosp.d_hcpcs",
        "hosp.diagnoses_icd",
        "hosp.d_icd_diagnoses",
        "hosp.d_icd_procedures",
        "hosp.d_labitems",
        "hosp.drgcodes",
        "hosp.emar",
        "hosp.emar_detail",
        "hosp.hcpcsevents",
        "hosp.labevents",
        "hosp.microbiologyevents",
        "hosp.pharmacy",
        "hosp.poe",
        "hosp.poe_detail",
        "hosp.prescriptions",
        "hosp.procedures_icd",
        "hosp.services",
        "icu.chartevents",
        "icu.datetimeevents",
        "icu.d_items",
        "icu.icustays",
        "icu.inputevents",
        "icu.outputevents",
        "icu.procedureevents",
    ]

    return modular_pipeline(
        [
            node(
                func=identity,
                inputs=["raw.%s" % t],
                outputs="data.%s" % t,
                name="convert_to_pq_%s" % t,
            )
            for t in mimic_tables
        ],
        namespace="preprocessing",
        inputs={"raw.%s" % t: "raw.%s" % t for t in mimic_tables},
        outputs={"data.%s" % t: "data.%s" % t for t in mimic_tables},
    )


def create_pipeline(**kwargs) -> Pipeline:
    return modular_pipeline(
        create_pipeline_split_mimic_keys() + create_intermediate_data(),
        namespace="mimic",
        parameters={"random_state": "random_state"},
    )
