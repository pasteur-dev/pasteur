from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.pipelines.mimic.pipeline import map_mimic_inputs

from .nodes import mm_core_transform_tables, tab_join_tables


def create_pipeline() -> Pipeline:
    tab_pipeline_m = modular_pipeline(
        pipe=pipeline(
            [
                node(
                    func=tab_join_tables,
                    inputs=["core_patients", "core_admissions"],
                    outputs="table",
                    name="create_tabular_dataset",
                )
            ]
        ),
        namespace="tab_admissions",
    )

    mm_core_pipeline_m = modular_pipeline(
        pipe=pipeline(
            [
                node(
                    func=mm_core_transform_tables,
                    inputs=["core_patients", "core_admissions", "core_transfers"],
                    outputs=["patients", "admissions", "transfers"],
                    name="create_mm_based_on_core",
                )
            ]
        ),
        namespace="mm_core",
    )

    combined_pipeline = tab_pipeline_m + mm_core_pipeline_m
    return modular_pipeline(
        combined_pipeline,
        inputs=map_mimic_inputs(combined_pipeline.inputs()),
        namespace="mimic_views",
    )
