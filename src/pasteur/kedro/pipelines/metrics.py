import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

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


def _create_model_log_pipelines(
    view: View, alg: str, trn_split: str, wrk_split: str, ref_split: str
):
    calc_nodes = []
    for table in view.tables:
        in_tables = {}
        for type in model_get_required_types():
            for split in (alg, wrk_split, ref_split):
                in_tables[
                    f"{type}.{split}.{table}"
                ] = f"{view.name}.{split}.{type}_{table}"

        calc_nodes += [
            node(
                func=node_calculate_model_scores,
                inputs={
                    "transformer": f"{view.name}.{trn_split}.trn_{table}",
                    **in_tables,
                },
                outputs=f"{view.name}.{alg}.meas_models_{table}",
                namespace=f"{view.name}.{alg}",
            ),
            node(
                func=gen_closure(mlflow_log_model_results, table),
                inputs=f"{view.name}.{alg}.meas_models_{table}",
                outputs=None,
                namespace=f"{view.name}.{alg}",
            ),
        ]

    return pipeline(calc_nodes)


def _create_visual_fit_pipelines(view: View, wrk_split: str, ref_split: str):
    in_tables_wrk = {table: f"{view.name}.{wrk_split}.{table}" for table in view.tables}
    in_tables_ref = {table: f"{view.name}.{ref_split}.{table}" for table in view.tables}

    hist_nodes = []
    for table in view.tables:
        hist_nodes += [
            node(
                func=gen_closure(create_fitted_hist_holder, table),
                inputs={
                    "meta": f"{view.name}.metadata",
                    "ids": f"{view.name}.{wrk_split}.ids_{table}",
                    **in_tables_wrk,
                },
                outputs=f"{view.name}.{wrk_split}.meas_hst_{table}",
                namespace=f"{view.name}.{wrk_split}",
            ),
            node(
                func=project_hists_for_view,
                inputs={
                    "holder": f"{view.name}.{wrk_split}.meas_hst_{table}",
                    "ids": f"{view.name}.{wrk_split}.ids_{table}",
                    **in_tables_wrk,
                },
                outputs=f"{view.name}.{wrk_split}.meas_viz_{table}",
                namespace=f"{view.name}.{wrk_split}",
            ),
            node(
                func=project_hists_for_view,
                inputs={
                    "holder": f"{view.name}.{wrk_split}.meas_hst_{table}",
                    "ids": f"{view.name}.{ref_split}.ids_{table}",
                    **in_tables_ref,
                },
                outputs=f"{view.name}.{ref_split}.meas_viz_{table}",
                namespace=f"{view.name}.{ref_split}",
            ),
        ]

    return pipeline(hist_nodes)


def _create_visual_log_pipelines(view: View, alg: str, wrk_split: str, ref_split: str):
    in_tables = {table: f"{view.name}.{alg}.{table}" for table in view.tables}

    hist_nodes = []
    for table in view.tables:
        hist_nodes += [
            node(
                func=project_hists_for_view,
                inputs={
                    "holder": f"{view.name}.{wrk_split}.meas_hst_{table}",
                    "ids": f"{view.name}.{alg}.ids_{table}",
                    **in_tables,
                },
                outputs=f"{view.name}.{alg}.meas_viz_{table}",
                namespace=f"{view.name}.{alg}",
            ),
            node(
                func=mlflow_log_hists,
                inputs={
                    "holder": f"{view.name}.{wrk_split}.meas_hst_{table}",
                    "wrk": f"{view.name}.{wrk_split}.meas_viz_{table}",
                    "syn": f"{view.name}.{alg}.meas_viz_{table}",
                    "ref": f"{view.name}.{ref_split}.meas_viz_{table}",
                },
                outputs=None,
                namespace=f"{view.name}.{alg}",
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
                        "ref": f"{view.name}.{wrk_split}.idx_{table}",
                        "syn": f"{view.name}.{ref_split}.idx_{table}",
                    },
                    outputs=f"{view.name}.{ref_split}.meas_{method}_{table}",
                    namespace=f"{view.name}.{ref_split}",
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
                        "ref": f"{view.name}.{wrk_split}.idx_{table}",
                        "syn": f"{view.name}.{alg}.idx_{table}",
                    },
                    outputs=f"{view.name}.{alg}.meas_{method}_{table}",
                    namespace=f"{view.name}.{alg}",
                ),
            ]

        for method, log in [("cs", log_cs_mlflow), ("kl", log_kl_mlflow)]:
            nodes += [
                node(
                    func=gen_closure(log, table, "ref"),
                    inputs={
                        "ref": f"{view.name}.{ref_split}.meas_{method}_{table}",
                        "syn": f"{view.name}.{alg}.meas_{method}_{table}",
                    },
                    outputs=None,
                    namespace=f"{view.name}.{alg}",
                ),
            ]

    return pipeline(nodes)


def get_required_types():
    types = list({"idx", "num"}.union(model_get_required_types()))
    types.sort()
    return types


def create_fit_pipelines(view: View, wrk_split: str, ref_split: str):
    return _create_distr_fit_pipelines(
        view, wrk_split, ref_split
    ) + _create_visual_fit_pipelines(view, wrk_split, ref_split)


def create_log_pipelines(
    view: View, alg: str, trn_split: str, wrk_split: str, ref_split: str
):
    return (
        _create_distr_log_pipelines(view, alg, wrk_split, ref_split)
        + _create_model_log_pipelines(view, alg, trn_split)
        + _create_visual_log_pipelines(view, alg, wrk_split, ref_split)
    )
