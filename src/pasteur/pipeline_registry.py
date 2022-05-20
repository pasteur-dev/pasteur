"""Project pipelines."""
from typing import Dict
from functools import reduce

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic import create_pipeline as create_pipeline_mimic
from .pipelines.mimic_views import create_pipeline as create_pipeline_mimic_views


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    ingest_pipeline = create_pipeline_mimic()

    mimic_views_pipelines = create_pipeline_mimic_views()
    mimic_views_pipelines_combined = reduce(
        lambda a, b: a + b, mimic_views_pipelines.values(), pipeline([])
    )
    mimic_views_pipelines_combined += create_pipeline_mimic(
        mimic_views_pipelines_combined.inputs()
    )

    pipelines = {
        "__default__": mimic_views_pipelines_combined,
        "ingest": ingest_pipeline,
        "mimic_views": mimic_views_pipelines_combined,
    }

    for name, pipe in mimic_views_pipelines.items():
        pipelines[name] = pipe
        pipelines["%s_full" % name] = pipe + create_pipeline_mimic(pipe.inputs())

    return pipelines
