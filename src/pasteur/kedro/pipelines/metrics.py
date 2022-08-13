import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...views.base import View

from ...metrics.models import node_calculate_model_scores, get_required_types
from .synth import create_transform_pipeline


def create_model_transform_pipelines(view: View):
    return create_transform_pipeline(
        view.name, "dev", view.tables, get_required_types(), "wrk"
    )


def create_model_calc_pipelines(view: View, alg: str):
    pipe_ingest = create_transform_pipeline(
        view.name, alg, view.tables, get_required_types(), "wrk", False
    )

    calc_nodes = []
    for table in view.tables:
        in_tables = {}
        for type in get_required_types():
            for split in (alg, "wrk", "dev"):
                in_tables[
                    f"{type}.{split}.{table}"
                ] = f"{view.name}.{split}.{type}_{table}"

        calc_nodes += [
            node(
                func=node_calculate_model_scores,
                inputs={"transformer": f"{view.name}.wrk.trn_{table}", **in_tables},
                outputs=f"tst_{table}",
            )
        ]

    return pipe_ingest + pipeline(calc_nodes)
