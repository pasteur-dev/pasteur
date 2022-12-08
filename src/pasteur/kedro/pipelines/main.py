import logging
from os import path
from typing import Dict

from kedro.pipeline import Pipeline

from ...dataset import Dataset
from ...encode import EncoderFactory
from ...module import Module, get_module_dict
from ...synth import SynthFactory
from ...transform import TransformerFactory
from ...view import View
from .dataset import create_dataset_pipeline
from .meta import DatasetMeta, PipelineMeta
from .metrics import (
    create_metrics_ingest_pipeline,
    create_metrics_model_pipeline,
    get_metrics_types,
)
from .synth import create_synth_pipeline
from .transform import (
    create_reverse_pipeline,
    create_transform_pipeline,
    create_transformer_pipeline,
)
from .utils import list_unique
from .views import (
    create_filter_pipeline,
    create_keys_pipeline,
    create_meta_pipeline,
    create_view_pipeline,
)

logger = logging.getLogger(__name__)

WRK_SPLIT = "wrk"
REF_SPLIT = "ref"
BASE_LOCATION = "base_location"
RAW_LOCATION = "raw_location"
NAME_LOCATION = "{}_location"


def _get_alg_types(algs: dict[str, SynthFactory]):
    alg_types = [a.type for a in algs.values()]
    return list_unique(alg_types)


def _is_downloaded(ds: Dataset, params: dict):
    if not ds.folder_name:
        return True

    p = params.get(
        NAME_LOCATION.format(ds.folder_name),
        path.join(params[RAW_LOCATION], ds.folder_name),
    )
    if path.exists(p):
        return True

    logger.warning(f'Disabling dataset {ds}, path "{p}" doesn\'t exist.')
    return False


def _has_dataset(view: View, datasets: dict[str, Dataset]):
    has = view.dataset in datasets

    if has:
        return True

    logger.warning(f"Disabling {view}, missing dataset {view.dataset}.")
    return False


def get_view_names(modules: list[Module]):
    return list(get_module_dict(View, modules).keys())


def generate_pipelines(
    modules: list[Module], params: dict
) -> tuple[
    dict[str, Pipeline],
    list[DatasetMeta],
    list[tuple[str, str | dict]],
    dict[str, dict | str],
]:
    """Generates synthetic pipelines for combinations of the provided views and algs.

    If None is passed, all registered classes are included."""

    datasets = get_module_dict(Dataset, modules)
    views = get_module_dict(View, modules)
    algs = get_module_dict(SynthFactory, modules)

    # Filter views and datasets
    datasets = {k: d for k, d in datasets.items() if _is_downloaded(d, params)}
    views = {k: v for k, v in views.items() if _has_dataset(v, datasets)}

    # Wrk, ref splits are transformed to all types
    # Synthetic data is transformed only to syn_types (as required by metrics currently)
    alg_types = _get_alg_types(algs)
    msr_types = get_metrics_types(modules)

    all_types = list_unique(alg_types, msr_types)
    encoders = {
        k: v
        for k, v in get_module_dict(EncoderFactory, modules).items()
        if k in all_types
    }
    transformers = get_module_dict(TransformerFactory, modules)

    wrk_split = WRK_SPLIT
    ref_split = REF_SPLIT
    splits = list_unique([wrk_split, ref_split])

    # Store complete pipelines first for kedro viz (main vs extra pipelines)
    main_pipes = {}
    extr_pipes = {}

    # Add dataset pipelines
    for name, dataset in datasets.items():
        extr_pipes[f"{name}.ingest"] = create_dataset_pipeline(dataset)

    for name, view in views.items():
        # Metrics fit pipeline is part of ingest
        # To make debugging metrics easier, it's bundled with `.measure` pipelines
        # as well. That way, only `.measure` needs to run when changes are made
        # to fit functions
        pipe_metrics_fit = create_metrics_ingest_pipeline(
            view, modules, wrk_split, ref_split
        )

        # Create view transform pipeline that can run as part of ingest
        pipe_transform = (
            create_transformer_pipeline(view, transformers, encoders, wrk_split)
            + create_transform_pipeline(
                view,
                wrk_split,
                all_types,
            )
            + create_transform_pipeline(view, ref_split, msr_types)
            + pipe_metrics_fit
        )

        # Metadata needs to be created every time to allow for overrides
        # Fixme: can cause issues with some parameters
        pipe_meta = create_meta_pipeline(view)

        pipe_ds_ingest = create_dataset_pipeline(
            datasets[view.dataset], view.dataset_tables
        )

        pipe_ingest = (
            create_keys_pipeline(view, splits)
            + create_view_pipeline(view)
            + pipe_meta
            + create_filter_pipeline(view, splits)
            + pipe_transform
        )

        # `<view>.<alg>` pipelines run all steps required for synthetic data
        # Steps that are view specific (common for all algs) can be run with `<vuew>`
        extr_pipes[f"{name}.ingest"] = pipe_ingest

        # Algorithm pipeline
        for alg, cls in algs.items():
            pipe_synth = create_synth_pipeline(
                view, wrk_split, cls
            ) + create_reverse_pipeline(view, alg, cls.type)

            pipe_measure = create_transform_pipeline(
                view, alg, msr_types, retransform=True
            ) + create_metrics_model_pipeline(view, alg, wrk_split, ref_split, modules)

            complete_pipe = pipe_ds_ingest + pipe_ingest + pipe_synth + pipe_measure

            if "ident" in alg:
                # Hide ident pipelines
                extr_pipes[f"{name}.{alg}"] = complete_pipe
            else:
                main_pipes[f"{name}.{alg}"] = complete_pipe

    # Hide extra pipes at the bottom of kedro viz
    # dictionaries are ordered
    pipes: dict[str, Pipeline | PipelineMeta] = {}
    try:
        default = next(iter(main_pipes))
    except StopIteration:
        # No pipelines
        default = None
    pipes["__default__"] = main_pipes.get(
        default, extr_pipes.get(default, Pipeline([]))
    )
    pipes.update(main_pipes)
    pipes["__misc_pipelines__"] = Pipeline([])
    pipes.update(extr_pipes)

    # Split pipelines
    pipelines = {k: v if isinstance(v, Pipeline) else v[0] for k, v in pipes.items()}

    # Split outputs and run checks
    outputs = {}
    for name, meta in pipes.items():
        if isinstance(meta, Pipeline):
            continue

        # Check for incongruencies
        pipe_out_names = meta.pipeline.all_outputs()
        out_names = {out.name for out in meta.outputs}

        diff = pipe_out_names.symmetric_difference(out_names)
        assert (
            not diff
        ), f"Pipeline meta {name} has different outputs than what is stated in the pipeline:\n{diff}"

        # Check all nodes have tags
        for node in meta.pipeline.nodes:
            assert node.tags, f"Node {node.name} doesn't have tags."

        for out in meta.outputs:
            outputs[out.name] = out

    return (
        pipelines,
        list(outputs.values()),
        [
            (d.folder_name, d.catalog)
            for d in datasets.values()
            if d.folder_name and d.catalog
        ],
        {str(v): v.parameters for v in views.values() if v.parameters},
    )
