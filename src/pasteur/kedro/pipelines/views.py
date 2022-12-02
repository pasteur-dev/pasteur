from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...utils.parser import get_params_for_pipe
from .meta import TAGS_VIEW, TAGS_VIEW_SPLIT
from .meta import DatasetMeta as D
from .meta import PipelineMeta
from .utils import gen_closure, get_params_closure

if TYPE_CHECKING:
    import pandas as pd

    from ...view import View


def _create_metadata(view: str, params: dict):
    meta_dict = get_params_for_pipe(view, params)
    return Metadata(meta_dict)


def _check_tables(metadata: Metadata, **tables: pd.DataFrame):
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
            ],
            tags=TAGS_VIEW,
        ),
        [
            D("primary", f"{view}.view.{t}", ["views", "primary", view, t], type="pq")
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
            ],
            tags=TAGS_VIEW_SPLIT,
        ),
        [D("metadata", f"{view}.metadata", ["views", "metadata", view], type="pkl")],
    )


def _filter_keys(
    view: View,
    req_splits: list[str] | None,
    ratios: dict[str, float],
    random_state: int,
    keys: pd.DataFrame,
):
    return {
        split: view.split_keys(keys, split, ratios, random_state)
        for split in req_splits or ratios.keys()
    }


def create_keys_pipeline(view: View, splits: list[str]):
    fun = get_params_closure(
        gen_closure(_filter_keys, view, splits, _fn="split_keys"),
        str(view),
        "ratios",
        "random_state",
    )

    pipe = pipeline(
        [
            node(
                func=fun,
                inputs={
                    "params": "parameters",
                    "keys": f"{view.dataset}.keys",
                },
                namespace=f"{view}.keys",
                outputs={s: f"{view}.keys.{s}" for s in splits},
                tags=TAGS_VIEW_SPLIT,
            )
        ]
    )

    return PipelineMeta(
        pipe,
        [D("keys", f"{view}.keys.{s}", ["views", "keys", view, s]) for s in splits],
    )


def create_filter_pipeline(view: View, splits: list[str]):
    tables = view.tables

    nodes = []
    for split in splits:
        for table in tables:
            nodes.append(
                node(
                    func=gen_closure(view.filter_table, _fn=f"filter_{table}_{split}"),
                    inputs={
                        "keys": f"keys.{split}",
                        **{t: f"view.{t}" for t in tables},
                    },
                    outputs=f"{split}.{table}",
                    namespace=split,
                    tags=TAGS_VIEW_SPLIT,
                )
            )

    return PipelineMeta(
        modular_pipeline(
            pipe=pipeline(nodes, tags=["view"]),
            namespace=view.name,
        ),
        [
            D("splits", f"{view}.{s}.{t}", ["views", "primary", f"{view}.{s}", t])
            for t in tables
            for s in splits
        ],
    )
