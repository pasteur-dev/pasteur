from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...views import View
from .utils import gen_closure


from ...metadata import Metadata
from ...utils import get_params_for_pipe
from ...transform import DEFAULT_TRANSFORMERS

import pandas as pd


def _create_metadata(view: str, params: dict, **tables: dict[str, pd.DataFrame]):
    meta_dict = get_params_for_pipe(view, params)
    return Metadata(meta_dict, tables, DEFAULT_TRANSFORMERS)


def create_view_pipeline(view: View):
    tables = view.tables

    trn_pipe = pipeline(
        [
            node(
                func=gen_closure(view.ingest, t, _fn=f"ingest_{t}"),
                inputs={dep: f"{view.dataset}.{dep}" for dep in view.deps[t]},
                namespace=f"{view}.view",
                outputs=f"{view}.view.{t}",
            )
            for t in tables
        ]
    )

    meta_pipe = pipeline(
        [
            node(
                func=gen_closure(_create_metadata, view.name, _fn="create_metadata"),
                inputs={
                    "params": "parameters",
                    **{t: f"{view}.view.{t}" for t in tables},
                },
                outputs=f"{view}.view.metadata",
                namespace=f"{view}.view",
            )
        ]
    )

    return trn_pipe + meta_pipe


def create_filter_pipeline(view: View, splits: list[str]):
    tables = view.tables

    pipe = pipeline([])
    for split in splits:
        pipe += pipeline(
            [
                node(
                    func=gen_closure(view.filter, _fn=f"filter_{split}"),
                    inputs={
                        "keys": f"keys.{split}",
                        **{t: f"view.{t}" for t in tables},
                    },
                    outputs={t: f"{split}.{t}" for t in tables},
                    namespace=split,
                )
            ]
        )

    return modular_pipeline(
        pipe=pipe,
        namespace=view.name,
    )
