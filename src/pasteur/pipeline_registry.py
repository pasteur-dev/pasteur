"""Project pipelines."""
from itertools import chain
from typing import Dict
from functools import reduce

from kedro.pipeline import Pipeline, pipeline

from .pipelines.mimic import create_pipeline as create_pipeline_mimic
from .pipelines.mimic_views import create_pipeline as create_pipeline_mimic_views
from .pipelines.synth import create_pipeline as create_pipeline_synth
from .pipelines.synth import get_algs
from .pipelines.general import create_split_pipeline
from .pipelines.measure import create_pipeline as create_measure_pipeline
from .pipelines.tab import create_ingest_pipelines, create_views_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pipelines = {}

    mimic_ingest_pipeline = create_pipeline_mimic()
    tab_ingest_pipelines = create_ingest_pipelines()
    tab_views_pipelines = create_views_pipeline()

    pipe_ingest_datasets = mimic_ingest_pipeline + reduce(
        lambda a, b: a + b, tab_ingest_pipelines.values()
    )
    pipe_ingest_views = create_pipeline_mimic(set())  # add key generation

    mimic_views_pipelines = create_pipeline_mimic_views()
    for name, pipe in chain(mimic_views_pipelines.items(), tab_views_pipelines.items()):
        if isinstance(pipe, tuple):
            ingest, pipe = pipe
            pipe_input = tab_ingest_pipelines[ingest]
        else:
            ingest = "mimic"
            pipe_input = create_pipeline_mimic(pipe.inputs())

        tables = [t.split(".")[-1] for t in pipe.outputs()]

        pipe_split = create_split_pipeline("wrk", ingest, name, tables)
        pipe_split_ref = create_split_pipeline("ref", ingest, name, tables)

        pipe_ingest = pipe_input + pipe + pipe_split
        pipe_ingest_views += pipe + pipe_split + pipe_split_ref
        pipelines[f"{name}.ingest"] = pipe_ingest + pipe_split_ref

        # Algorithm pipeline
        for alg in get_algs():
            pipe_synth = create_pipeline_synth(name, "wrk", alg, tables)
            pipe_measure = create_measure_pipeline(name, "wrk", alg, tables)
            pipelines[f"{name}.{alg}"] = pipe_ingest + pipe_synth + pipe_measure
            pipelines[f"{name}.{alg}.synth"] = pipe_synth + pipe_measure
            pipelines[f"{name}.{alg}.measure"] = pipe_measure

        # Validation (sister dataset)
        pipe_ingest = pipe_input + pipe + pipe_split + pipe_split_ref
        pipe_measure = create_measure_pipeline(name, "wrk", "ref", tables)
        pipelines[f"{name}.ref"] = pipe_ingest + pipe_measure
        pipelines[f"{name}.ref.measure"] = pipe_measure

    pipelines["__default__"] = pipelines["mimic_mm_core.hma1"]
    pipelines["ingest"] = pipe_ingest_views + pipe_ingest_datasets
    pipelines["ingest.datasets"] = pipe_ingest_datasets
    pipelines["ingest.views"] = pipe_ingest_views
    return pipelines
