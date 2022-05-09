"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic import create_pipeline as create_pipeline_mimic
from .pipelines.mimic_views import create_pipeline as create_pipeline_mimic_views


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    mimic_ingest_pipeline = create_pipeline_mimic()
    mimic_views = create_pipeline_mimic_views()
    mimic_pipeline = create_pipeline_mimic(mimic_views.inputs()) + mimic_views

    return {
        "__default__": mimic_pipeline,
        "mimic": mimic_pipeline,
        "mimic_ingest": mimic_ingest_pipeline,
        "mimic_experiment": mimic_views,
    }
