from functools import reduce
from itertools import chain
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from ...dataset import Dataset, get_datasets
from ...synth import Synth, get_synth
from ...views import View, get_views
from .dataset import create_dataset_pipeline, create_keys_pipeline
from .metrics import (
    get_required_types as metrics_get_required_types,
    create_fit_pipelines as metrics_create_fit_pipelines,
    create_log_pipelines as metrics_create_log_pipelines,
)
from .synth import create_synth_pipeline
from .views import create_filter_pipeline, create_meta_pipeline, create_view_pipeline
from .transform import (
    create_transformers_pipeline,
    create_reverse_pipeline,
    create_transform_pipeline,
)

from .utils import list_unique

WRK_SPLIT = "wrk"
REF_SPLIT = "ref"


def _get_algs(names: list[str]) -> dict[str, type[Synth]]:
    algs = get_synth()
    if names:
        return {n: algs[n] for n in names}
    return algs


def _get_views(names: list[str]) -> dict[str, View]:
    views = get_views()
    if names:
        views = {n: views[n] for n in names}
    return {n: v() for n, v in views.items()}


def _get_datasets(views: dict[str, View]) -> dict[str, Dataset]:
    names = list_unique([v.dataset for v in views.values()])
    datasets = get_datasets()
    return {n: datasets[n]() for n in names}


def get_view_tables(names: list[str]):
    return {n: v.tables for n, v in _get_views(names).items()}


def get_msr_types():
    return metrics_get_required_types()


def _get_all_types(algs: dict[str, Synth]):
    alg_types = [a.type for a in algs.values()]
    return list_unique(alg_types, get_msr_types())


def get_all_types(algs: list[str]):
    return _get_all_types(_get_algs(algs))


def generate_pipelines(
    views: list[str] | None,
    algs: list[str] | None,
    default: str | None = "tab_adult.privbayes",
) -> Dict[str, Pipeline]:
    """Generates synthetic pipelines for combinations of the provided views and algs.

    If None is passed, all registered classes are included."""

    views = _get_views(views)
    algs = _get_algs(algs)
    datasets = _get_datasets(views)

    wrk_split = WRK_SPLIT
    ref_split = REF_SPLIT
    splits = list_unique([wrk_split, ref_split])

    # Wrk, ref splits are transformed to all types
    # Synthetic data is transformed only to syn_types (as required by metrics currently)
    msr_types = get_msr_types()
    all_types = _get_all_types(algs)

    # Store complete pipelines first for kedro viz (main vs extra pipelines)
    main_pipes = {}
    extr_pipes = {}

    for name, view in views.items():
        # Metrics fit pipeline is part of ingest
        # To make debugging metrics easier, it's bundled with `.measure` pipelines
        # as well. That way, only `.measure` needs to run when changes are made
        # to fit functions
        pipe_metrics_fit = metrics_create_fit_pipelines(view, wrk_split, ref_split)

        # Create view transform pipeline that can run as part of ingest
        pipe_transform = (
            create_transformers_pipeline(view, all_types)
            + create_transform_pipeline(view, wrk_split, all_types)
            + create_transform_pipeline(view, ref_split, msr_types)
            + pipe_metrics_fit
        )

        # Metadata needs to be created every time to allow for overrides
        # Fixme: can cause issues with some parameters
        pipe_meta = create_meta_pipeline(view)

        pipe_ingest = (
            create_dataset_pipeline(datasets[view.dataset], view.dataset_tables)
            + create_keys_pipeline(datasets[view.dataset], name, splits)
            + create_view_pipeline(view)
            + pipe_meta
            + create_filter_pipeline(view, splits)
            + pipe_transform
        )

        # `<view>.<alg>` pipelines run all steps required for synthetic data
        # Steps that are view specific (common for all algs) can be run with
        # <view>.ingest, `<view>.<alg>.synth pipelines can be run after that
        extr_pipes[f"{name}.ingest"] = pipe_ingest

        # Algorithm pipeline
        for alg, cls in algs.items():
            pipe_synth = create_synth_pipeline(
                view, wrk_split, cls
            ) + create_reverse_pipeline(view, alg, cls.type)

            pipe_measure = create_transform_pipeline(
                view, alg, msr_types, only_encode=True
            ) + metrics_create_log_pipelines(view, alg, wrk_split, ref_split)

            complete_pipe = pipe_ingest + pipe_synth + pipe_measure

            if "ident" in alg:
                # Hide ident pipelines
                extr_pipes[f"{name}.{alg}"] = complete_pipe
            else:
                main_pipes[f"{name}.{alg}"] = complete_pipe
            extr_pipes[f"{name}.{alg}.synth"] = pipe_synth + pipe_measure + pipe_meta
            extr_pipes[f"{name}.{alg}.measure"] = (
                pipe_metrics_fit + pipe_measure + pipe_meta
            )

    # Hide extra pipes at the bottom of kedro viz
    # dictionaries are ordered
    pipes = {}
    pipes["__default__"] = main_pipes.get(default, extr_pipes.get(default, []))
    pipes.update(main_pipes)
    pipes["__misc_pipelines__"] = pipeline([])
    pipes.update(extr_pipes)

    return pipes
