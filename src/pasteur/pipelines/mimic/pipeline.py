"""
This file contains pipelines specific to processing the dataset MIMIC-IV.
"""

from typing import Optional, Set
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.mimic.nodes import select_id

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

    mimic_tables = inputs.intersection(mimic_tables_all) if inputs else mimic_tables_all

    parquet_pipeline = pipeline(
        [
            node(
                func=identity,
                inputs=["mimic_iv@%s" % t],
                outputs=t,
                name="ingest_%s" % t,
            )
            for t in mimic_tables
        ]
    )
    return parquet_pipeline


def create_pipeline_split_mimic_keys(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(select_id, inputs=["core_patients"], outputs="keys_all"),
            create_node_split_keys(),
        ],
    )


def create_pipeline_mimic_tasks(**kwargs) -> Pipeline:
    return create_pipeline_split_mimic_keys()


def create_pipeline_ingest():
    return modular_pipeline(
        create_intermediate_data(), namespace="mimic", inputs=["mimic_iv"]
    )


def create_pipeline_experiment():
    return modular_pipeline(
        create_pipeline_mimic_tasks(),
        namespace="mimic",
        parameters=["random_state"],
    )


def create_pipeline(**kwargs) -> Pipeline:
    mimic_pipeline = create_pipeline_split_mimic_keys()
    mimic_inputs = create_intermediate_data(mimic_pipeline.inputs())

    return modular_pipeline(
        mimic_pipeline + mimic_inputs,
        namespace="mimic",
        inputs=["mimic_iv"],
        parameters=["random_state"],
    )
