"""Project pipelines."""
from typing import Dict
from functools import reduce

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic import create_pipeline as create_pipeline_mimic
from .pipelines.mimic_views import create_pipeline as create_pipeline_mimic_views
from .pipelines.synth import create_pipeline as create_pipeline_synth
from .pipelines.synth import get_algs
from .pipelines.general import create_split_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pipelines = {}
    pipe_ingest_all = create_pipeline_mimic()

    mimic_views_pipelines = create_pipeline_mimic_views()
    for name, pipe in mimic_views_pipelines.items():
        pipe_input = create_pipeline_mimic(pipe.inputs())
        pipe_split = create_split_pipeline("wrk", "mimic", name, pipe.outputs())

        pipe_ingest = pipe_input + pipe + pipe_split
        pipe_ingest_all += pipe_ingest
        pipelines[f"{name}.ingest"] = pipe_ingest

        for alg in get_algs():
            pipe_synth = create_pipeline_synth(name, "wrk", "hma1", pipe.outputs())
            pipelines[f"{name}.{alg}"] = pipe_synth
            pipelines[f"{name}.{alg}_full"] = pipe_ingest + pipe_synth

    pipelines["__default__"] = pipelines["mimic_mm_core.hma1_full"]
    pipelines["ingest"] = pipe_ingest_all
    return pipelines
