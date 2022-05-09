from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.mimic.pipeline import map_mimic_inputs

from .nodes import mm_core_transform_tables, tab_join_tables


def create_pipeline(**kwargs) -> Pipeline:
    tab_pipeline = pipeline(
        [
            node(
                func=tab_join_tables,
                inputs=["core_patients", "core_admissions"],
                outputs="main",
                name="create_tabular_dataset",
            )
        ]
    )

    tab_pipeline_m = modular_pipeline(
        pipe=tab_pipeline,
        inputs=map_mimic_inputs(tab_pipeline.inputs(), "tab"),
        namespace="mimic_views.tab.admissions",
    )

    mm_core_pipeline = pipeline(
        [
            node(
                func=mm_core_transform_tables,
                inputs=["core_patients", "core_admissions", "core_transfers"],
                outputs=["patients", "admissions", "transfers"],
                name="create_tabular_dataset",
            )
        ]
    )

    mm_core_pipeline_m = modular_pipeline(
        pipe=mm_core_pipeline,
        inputs=map_mimic_inputs(mm_core_pipeline.inputs()),
        namespace="mimic_views.mm.core",
    )

    return tab_pipeline_m + mm_core_pipeline_m
