from functools import reduce
from itertools import chain
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .dataset import create_dataset_pipeline, create_keys_pipeline
from .measure import create_pipeline as create_measure_pipeline
from .synth import create_synth_pipeline, create_transform_pipeline
from .views import create_filter_pipeline, create_view_pipeline

from ...dataset import get_datasets, Dataset
from ...views import get_views, View
from ...synth import get_synth, Synth


def generate_pipelines(
    views: list[str] | None,
    algs: list[str] | None,
    default: str | None = "tab_adult.privbayes",
) -> Dict[str, Pipeline]:
    """Generates synthetic pipelines for combinations of the provided views and algs.

    If None is passed, all registered classes are included."""

    views: dict[str, View] = {
        n: v() for n, v in get_views().items() if not views or n in views
    }
    datasets = [v.dataset for v in views.values()]
    datasets: dict[str, Dataset] = {
        n: d() for n, d in get_datasets().items() if n in datasets
    }
    algs: dict[str, Synth] = {
        n: a for n, a in get_synth().items() if not algs or n in algs
    }

    splits = ["wrk", "ref", "val", "dev"]

    # Store complete pipelines first for kedro viz (main vs extra pipelines)
    main_pipes = {}
    extr_pipes = {}
    # merge_pipes = lambda pipes: reduce(lambda a, b: a + b, pipes, pipeline([]))

    ## Ingest Pipelines
    # pipe_ingest_datasets = merge_pipes(
    #     create_dataset_pipeline(d) for d in datasets.values()
    # )
    # pipe_ingest_views = merge_pipes(
    #     create_keys_pipeline(datasets[v.dataset], n, splits)
    #     + create_view_pipeline(v)
    #     + create_filter_pipeline(v, splits)
    #     for n, v in views.items()
    # )
    # pipe_ingest_all = pipe_ingest_datasets + pipe_ingest_views

    for name, view in views.items():
        types = [s.type for s in algs.values() if s.tabular == view.tabular]
        types = list(dict.fromkeys(types))  # remove duplicates

        pipe_transform = create_transform_pipeline(name, "wrk", view.tables, types)

        pipe_ingest = (
            create_dataset_pipeline(datasets[view.dataset], view.dataset_tables)
            + create_keys_pipeline(datasets[view.dataset], name, splits)
            + create_view_pipeline(view)
            + create_filter_pipeline(view, splits)
            + pipe_transform
        )
        extr_pipes[f"{name}.ingest"] = pipe_ingest

        # Algorithm pipeline
        for alg, cls in algs.items():
            pipe_synth = create_synth_pipeline(
                name, "wrk", cls, view.tables, view.trn_deps
            )
            pipe_measure = create_measure_pipeline(name, "wrk", alg, view.tables)

            complete_pipe = pipe_ingest + pipe_transform + pipe_synth + pipe_measure

            if "ident" in alg:
                # Hide ident pipelines
                extr_pipes[f"{name}.{alg}"] = complete_pipe
            else:
                main_pipes[f"{name}.{alg}"] = complete_pipe
            extr_pipes[f"{name}.{alg}.synth"] = pipe_synth + pipe_measure
            extr_pipes[f"{name}.{alg}.measure"] = pipe_measure

        # Validation (sister dataset)
        pipe_measure = create_measure_pipeline(name, "wrk", "ref", view.tables)
        main_pipes[f"{name}.ref"] = pipe_ingest + pipe_measure
        extr_pipes[f"{name}.ref.measure"] = pipe_measure

    # extr_pipes["ingest"] = pipe_ingest_all
    # extr_pipes["ingest.datasets"] = pipe_ingest_datasets
    # extr_pipes["ingest.views"] = pipe_ingest_views

    algs_str = list(algs.keys())
    tables = {n: v.tables for n, v in views.items()}

    # Hide extra pipes at the bottom of kedro viz
    # dictionaries are ordered
    pipes = {}
    pipes["__default__"] = main_pipes.get(default, extr_pipes.get(default, []))
    pipes.update(main_pipes)
    pipes["__misc_pipelines__"] = pipeline([])
    pipes.update(extr_pipes)

    return pipes, algs_str, tables
