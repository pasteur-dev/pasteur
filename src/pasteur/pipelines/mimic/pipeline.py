"""
This file contains pipelines specific to processing the dataset MIMIC-IV.
"""

from typing import Collection, Dict, Optional, Set
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..general import create_node_split_keys, identity

from .nodes import mm_core_transform_tables, tab_join_tables

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


def create_intermediate_data(inputs: Optional[Set] = None) -> Pipeline:
    if inputs:
        inputs_clear = {t.replace("mimic.", "").split("@")[0] for t in inputs}
        mimic_tables = inputs_clear.intersection(mimic_tables_all)
    else:
        mimic_tables = mimic_tables_all

    parquet_pipeline = pipeline(
        [
            node(
                func=identity,
                inputs=[f"mimic_iv@{t}"],
                outputs=t,
                name=f"ingest_{t}",
            )
            for t in mimic_tables
        ]
    )
    return parquet_pipeline


def create_pipeline_split_mimic_keys() -> Pipeline:
    return modular_pipeline(
        pipeline([create_node_split_keys()]), inputs={"keys_all": "core_patients"}
    )


def map_mimic_inputs(inputs: Set[str]) -> Dict[str, str]:
    def fix_mimic_import(i):
        name = i.split(".")[-1]
        if name in mimic_tables_all:
            return f"mimic.{name}"
        return i

    return {i: fix_mimic_import(i) for i in inputs}


def create_ingest_pipeline(inputs: Optional[Set] = None) -> Pipeline:
    mimic_keys = create_pipeline_split_mimic_keys()
    inputs = inputs.union(mimic_keys.inputs()) if inputs is not None else None
    mimic_ingest = create_intermediate_data(inputs)

    return modular_pipeline(
        mimic_keys + mimic_ingest,
        namespace="mimic",
        inputs=["mimic_iv"],
        parameters=["random_state"],
    )


def create_views_pipelines() -> Dict[str, Pipeline]:
    pipelines = {}

    pipelines["tab_admissions"] = pipeline(
        [
            node(
                func=tab_join_tables,
                inputs=["core_patients", "core_admissions"],
                outputs="table",
                name="create_tabular_dataset",
            )
        ]
    )

    pipelines["mm_core"] = pipeline(
        [
            node(
                func=mm_core_transform_tables,
                inputs=["core_patients", "core_admissions", "core_transfers"],
                outputs=["patients", "admissions", "transfers"],
                name="create_mm_based_on_core",
            )
        ]
    )

    return {
        "mimic_%s"
        % name: modular_pipeline(
            pipe=pipe,
            inputs=map_mimic_inputs(pipe.inputs()),
            namespace=f"mimic_{name}.view",
        )
        for name, pipe in pipelines.items()
    }


def get_datasets() -> Dict[str, Collection[str]]:
    pipelines = create_views_pipelines()
    return {n: {o.split(".")[-1] for o in p.outputs()} for n, p in pipelines.items()}
