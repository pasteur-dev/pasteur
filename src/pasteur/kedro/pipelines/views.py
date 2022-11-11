from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...utils.parser import get_params_for_pipe
from .meta import DatasetMeta as D
from .meta import PipelineMeta
from .utils import gen_closure

if TYPE_CHECKING:
    import pandas as pd

    from ...view import View


def _create_metadata(view: str, params: dict):
    meta_dict = get_params_for_pipe(view, params)
    return Metadata(meta_dict)


def _check_tables(metadata: Metadata, **tables: dict[str, pd.DataFrame]):
    metadata.check(tables)


def create_view_pipeline(view: View):
    return PipelineMeta(
        pipeline(
            [
                node(
                    func=gen_closure(view.ingest, t, _fn=f"ingest_{t}"),
                    inputs={dep: f"{view.dataset}.{dep}" for dep in view.deps[t]},
                    namespace=f"{view}.view",
                    outputs=f"{view}.view.{t}",
                )
                for t in view.tables
            ]
            + [
                node(
                    func=gen_closure(_check_tables, _fn="check_tables"),
                    inputs={
                        "metadata": f"{view}.metadata",
                        **{t: f"{view}.view.{t}" for t in view.tables},
                    },
                    outputs=None,
                    namespace=f"{view}.view",
                )
            ]
        ),
        [
            D("primary", f"{view}.view.{t}", ["views", "primary", view, t])
            for t in view.tables
        ],
    )


def create_meta_pipeline(view: View):
    return PipelineMeta(
        pipeline(
            [
                node(
                    func=gen_closure(
                        _create_metadata, view.name, _fn="create_metadata"
                    ),
                    inputs="parameters",
                    outputs=f"{view}.metadata",
                    namespace=f"{view}",
                )
            ]
        ),
        [D("metadata", f"{view}.metadata", ["views", "metadata", view], type="pkl")],
    )


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

    return PipelineMeta(
        modular_pipeline(
            pipe=pipe,
            namespace=view.name,
        ),
        [
            D("primary", f"{view}.{s}.{t}", ["views", "primary", f"{view}.{s}", t])
            for t in tables
            for s in splits
        ],
    )
