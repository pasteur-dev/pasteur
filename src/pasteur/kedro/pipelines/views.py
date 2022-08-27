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

    pipe = pipeline(
        [
            node(
                func=gen_closure(view.ingest, t, _fn=f"ingest_{t}"),
                inputs={dep: f"in_{dep}" for dep in view.deps[t]},
                outputs=f"view.{t}",
            )
            for t in tables
        ]
    )

    trn_pipe = modular_pipeline(
        pipe=pipe,
        namespace=view.name,
        inputs={f"in_{dep}": f"{view.dataset}.{dep}" for dep in view.dataset_tables},
    )

    meta_pipe = pipeline(
        [
            node(
                func=gen_closure(_create_metadata, view.name, _fn="create_metadata"),
                inputs={
                    "params": "parameters",
                    **{t: f"{view.name}.view.{t}" for t in tables},
                },
                outputs=f"{view.name}.metadata",
                namespace=view.name,
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
                    func=view.filter,
                    inputs={
                        "keys": f"keys.{split}",
                        **{t: f"view.{t}" for t in tables},
                    },
                    outputs={t: f"{split}.{t}" for t in tables},
                )
            ]
        )

    return modular_pipeline(
        pipe=pipe,
        namespace=view.name,
    )
