"""Project pipelines."""
from typing import Dict
from functools import reduce

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic import create_pipeline as create_pipeline_mimic
from .pipelines.mimic_views import create_pipeline as create_pipeline_mimic_views
from .pipelines.synth import create_pipeline as create_pipeline_synth
from .pipelines.synth import get_algs
from .pipelines.general import create_split_pipeline
from .pipelines.measure import create_pipeline as create_measure_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pipelines = {}
    pipe_ingest_datasets = create_pipeline_mimic()
    pipe_ingest_views = pipeline([])

    mimic_views_pipelines = create_pipeline_mimic_views()
    for name, pipe in mimic_views_pipelines.items():
        tables = [t.split(".")[-1] for t in pipe.outputs()]

        pipe_input = create_pipeline_mimic(pipe.inputs())
        pipe_split = create_split_pipeline("wrk", "mimic", name, tables)
        pipe_split_ref = create_split_pipeline("ref", "mimic", name, tables)

        pipe_ingest = pipe_input + pipe + pipe_split
        pipe_ingest_views += pipe + pipe_split + pipe_split_ref
        pipelines[f"{name}.ingest"] = pipe_ingest + pipe_split_ref

        # Algorithm pipeline
        for alg in get_algs():
            pipe_synth = create_pipeline_synth(
                name, "wrk", alg, tables
            ) + create_measure_pipeline(name, "wrk", alg, tables)
            pipelines[f"{name}.{alg}"] = pipe_synth
            pipelines[f"{name}.{alg}_full"] = pipe_ingest + pipe_synth

        # Validation (sister dataset)
        pipe_ingest = pipe_input + pipe + pipe_split + pipe_split_ref
        pipe_measure = create_measure_pipeline(name, "wrk", "ref", tables)
        pipelines[f"{name}.ref"] = pipe_measure
        pipelines[f"{name}.ref_full"] = pipe_ingest + pipe_measure

    pipelines["__default__"] = pipelines["mimic_mm_core.hma1"]
    pipelines["ingest"] = pipe_ingest_views + pipe_ingest_datasets
    pipelines["ingest.datasets"] = pipe_ingest_datasets
    pipelines["ingest.views"] = pipe_ingest_views
    return pipelines
