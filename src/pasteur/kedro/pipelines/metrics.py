import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from pasteur.metrics.mlflow import mlflow_log_as_str

from ...metrics.models import (
    get_required_types as model_get_required_types,
    mlflow_log_model_results,
    node_calculate_model_scores,
)
from ...metrics.visual import (
    project_hists_for_view,
    create_fitted_hist_holder,
    mlflow_log_hists,
)
from ...views.base import View
from .utils import gen_closure

from ...metrics.distr import log_kl_mlflow, log_cs_mlflow, calc_kl, calc_chisquare


def _create_model_log_pipelines(view: View, alg: str, wrk_split: str, ref_split: str):
    calc_nodes = []
    for table in view.tables:
        in_tables = {}
        for type in model_get_required_types():
            for label, split in [("syn", alg), ("wrk", wrk_split), ("ref", ref_split)]:
                in_tables[f"{type}.{label}.{table}"] = f"{view}.{split}.{type}_{table}"

        calc_nodes += [
            node(
                func=node_calculate_model_scores,
                inputs={
                    "transformer": f"{view}.trn.{table}",
                    **in_tables,
                },
                outputs=f"{view}.{alg}.msr_mdl_{table}",
                namespace=f"{view}.{alg}",
            ),
            node(
                func=gen_closure(mlflow_log_model_results, table),
                inputs=f"{view}.{alg}.msr_mdl_{table}",
                outputs=None,
                namespace=f"{view}.{alg}",
            ),
        ]

    return pipeline(calc_nodes)


def _create_visual_fit_pipelines(view: View, wrk_split: str, ref_split: str):
    hist_nodes = []
    for table in view.tables:
        in_tables = {table: f"{view}.view.{table}" for table in view.tables}
        hist_nodes += [
            node(
                func=gen_closure(create_fitted_hist_holder, table),
                inputs={
                    "meta": f"{view}.metadata",
                    "ids": f"{view}.trn.ids_{table}",
                    **in_tables,
                },
                outputs=f"{view}.msr.hst_{table}",
                namespace=f"{view}.msr",
            )
        ]

        for split in (wrk_split, ref_split):
            in_tables = {table: f"{view}.{split}.{table}" for table in view.tables}
            hist_nodes += [
                node(
                    func=project_hists_for_view,
                    inputs={
                        "holder": f"{view}.msr.hst_{table}",
                        "ids": f"{view}.{split}.ids_{table}",
                        **in_tables,
                    },
                    outputs=f"{view}.{split}.msr_viz_{table}",
                    namespace=f"{view}.{split}",
                ),
            ]

    return pipeline(hist_nodes)


def _create_visual_log_pipelines(view: View, alg: str, wrk_split: str, ref_split: str):
    in_tables = {table: f"{view}.{alg}.{table}" for table in view.tables}

    hist_nodes = []
    for table in view.tables:
        hist_nodes += [
            node(
                func=project_hists_for_view,
                inputs={
                    "holder": f"{view}.msr.hst_{table}",
                    "ids": f"{view}.{alg}.ids_{table}",
                    **in_tables,
                },
                outputs=f"{view}.{alg}.msr_viz_{table}",
                namespace=f"{view}.{alg}",
            ),
            node(
                func=mlflow_log_hists,
                inputs={
                    "holder": f"{view}.msr.hst_{table}",
                    "wrk": f"{view}.{wrk_split}.msr_viz_{table}",
                    "syn": f"{view}.{alg}.msr_viz_{table}",
                    "ref": f"{view}.{ref_split}.msr_viz_{table}",
                },
                outputs=None,
                namespace=f"{view}.{alg}",
            ),
        ]

    return pipeline(hist_nodes)


def _create_distr_fit_pipelines(view: View, wrk_split: str, ref_split: str):
    # TODO: add support for expanded tables

    nodes = []
    for table in view.tables:
        for method, calc in [("cs", calc_chisquare), ("kl", calc_kl)]:
            nodes += [
                node(
                    func=gen_closure(calc, _fn=f"%s_{table}"),
                    inputs={
                        "ref": f"{view}.{wrk_split}.idx_{table}",
                        "syn": f"{view}.{ref_split}.idx_{table}",
                    },
                    outputs=f"{view}.{ref_split}.msr_{method}_{table}",
                    namespace=f"{view}.{ref_split}",
                ),
            ]

    return pipeline(nodes)


def _create_distr_log_pipelines(view: View, alg: str, wrk_split: str, ref_split: str):
    nodes = []
    for table in view.tables:
        for method, calc in [("cs", calc_chisquare), ("kl", calc_kl)]:
            nodes += [
                node(
                    func=calc,
                    inputs={
                        "ref": f"{view}.{wrk_split}.idx_{table}",
                        "syn": f"{view}.{alg}.idx_{table}",
                    },
                    outputs=f"{view}.{alg}.msr_{method}_{table}",
                    namespace=f"{view}.{alg}",
                ),
            ]

        for method, log in [("cs", log_cs_mlflow), ("kl", log_kl_mlflow)]:
            nodes += [
                node(
                    func=gen_closure(log, table, "ref"),
                    inputs={
                        "ref": f"{view}.{ref_split}.msr_{method}_{table}",
                        "syn": f"{view}.{alg}.msr_{method}_{table}",
                    },
                    outputs=None,
                    namespace=f"{view}.{alg}",
                ),
            ]

    return pipeline(nodes)


def _create_synth_log_pipeline(view: View, alg: str):
    return pipeline(
        [
            node(
                func=gen_closure(
                    mlflow_log_as_str, "synth", _fn="mlflow_log_model_view"
                ),
                inputs=[f"{view}.{alg}.model"],
                outputs=None,
                namespace=f"{view}.{alg}",
            ),
        ]
    )


def get_required_types():
    types = list({"idx", "num"}.union(model_get_required_types()))
    types.sort()
    return types


def create_fit_pipelines(view: View, wrk_split: str, ref_split: str):
    return _create_distr_fit_pipelines(
        view, wrk_split, ref_split
    ) + _create_visual_fit_pipelines(view, wrk_split, ref_split)


def create_log_pipelines(view: View, alg: str, wrk_split: str, ref_split: str):
    return (
        _create_distr_log_pipelines(view, alg, wrk_split, ref_split)
        + _create_model_log_pipelines(view, alg, wrk_split, ref_split)
        + _create_visual_log_pipelines(view, alg, wrk_split, ref_split)
        + _create_synth_log_pipeline(view, alg)
    )
