"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic.pipeline import (
    create_pipeline as create_pipeline_mimic,
    create_pipeline_experiment,
    create_pipeline_ingest,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    mimic_pipeline = create_pipeline_mimic()
    mimic_ingest_pipeline = create_pipeline_ingest()
    mimic_experiment_pipeline = create_pipeline_experiment()

    return {
        "__default__": mimic_pipeline,
        "mimic": mimic_pipeline,
        "mimic_ingest": mimic_ingest_pipeline,
        "mimic_experiment": mimic_experiment_pipeline,
    }
