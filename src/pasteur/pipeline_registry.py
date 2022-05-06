"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic.pipeline import create_pipeline as create_pipeline_mimic

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    mimic_pipeline = create_pipeline_mimic()

    return {"__default__": mimic_pipeline, 'mimic': mimic_pipeline}
