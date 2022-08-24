import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metrics.models import (
    get_required_types,
    mlflow_log_model_closure,
    node_calculate_model_scores,
)
from ...metrics.visual import (
    project_hists_for_view,
    create_fitted_hist_holder_closure,
    mlflow_log_hists,
)
from ...views.base import View
from .synth import create_transform_pipeline


def create_model_transform_pipelines(view: View):
    return create_transform_pipeline(
        view.name, "tst", view.tables, get_required_types(), "wrk"
    )


def create_model_calc_pipelines(view: View, alg: str):
    pipe_ingest = create_transform_pipeline(
        view.name, alg, view.tables, get_required_types(), "wrk", False
    )

    calc_nodes = []
    for table in view.tables:
        in_tables = {}
        for type in get_required_types():
            for split in (alg, "wrk", "tst"):
                in_tables[
                    f"{type}.{split}.{table}"
                ] = f"{view.name}.{split}.{type}_{table}"

        calc_nodes += [
            node(
                func=node_calculate_model_scores,
                inputs={"transformer": f"{view.name}.wrk.trn_{table}", **in_tables},
                outputs=f"{view.name}.{alg}.meas_models_{table}",
                namespace=f"{view.name}.{alg}",
            ),
            node(
                func=mlflow_log_model_closure(table),
                inputs=f"{view.name}.{alg}.meas_models_{table}",
                outputs=None,
                namespace=f"{view.name}.{alg}",
            ),
        ]

    return pipe_ingest + pipeline(calc_nodes)


def create_visual_fit_pipelines(view: View):
    in_tables_wrk = {table: f"{view.name}.wrk.{table}" for table in view.tables}
    in_tables_tst = {table: f"{view.name}.tst.{table}" for table in view.tables}

    hist_nodes = []
    for table in view.tables:
        hist_nodes += [
            node(
                func=create_fitted_hist_holder_closure(table),
                inputs={"transformer": f"{view.name}.wrk.trn_{table}", **in_tables_wrk},
                outputs=f"{view.name}.wrk.meas_hst_{table}",
                namespace=f"{view.name}.wrk",
            ),
            node(
                func=project_hists_for_view,
                inputs={"holder": f"{view.name}.wrk.meas_hst_{table}", **in_tables_wrk},
                outputs=f"{view.name}.wrk.meas_viz_{table}",
                namespace=f"{view.name}.wrk",
            ),
            node(
                func=project_hists_for_view,
                inputs={"holder": f"{view.name}.wrk.meas_hst_{table}", **in_tables_tst},
                outputs=f"{view.name}.tst.meas_viz_{table}",
                namespace=f"{view.name}.tst",
            ),
        ]

    return pipeline(hist_nodes)


def create_visual_log_pipelines(view: View, alg: str):
    in_tables = {table: f"{view.name}.{alg}.{table}" for table in view.tables}

    hist_nodes = []
    for table in view.tables:
        hist_nodes += [
            node(
                func=project_hists_for_view,
                inputs={"holder": f"{view.name}.wrk.meas_hst_{table}", **in_tables},
                outputs=f"{view.name}.{alg}.meas_viz_{table}",
                namespace=f"{view.name}.{alg}",
            ),
            node(
                func=mlflow_log_hists,
                inputs={
                    "holder": f"{view.name}.wrk.meas_hst_{table}",
                    "wrk": f"{view.name}.wrk.meas_viz_{table}",
                    "alg": f"{view.name}.{alg}.meas_viz_{table}",
                    "ref": f"{view.name}.tst.meas_viz_{table}",
                },
                outputs=None,
                namespace=f"{view.name}.{alg}",
            ),
        ]

    return pipeline(hist_nodes)
