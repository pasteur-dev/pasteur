from typing import Collection, Dict
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..mimic import map_mimic_inputs

from .nodes import mm_core_transform_tables, tab_join_tables


def create_pipeline() -> Dict[str, Pipeline]:
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
    pipelines = create_pipeline()
    return {n: {o.split(".")[-1] for o in p.outputs()} for n, p in pipelines.items()}
