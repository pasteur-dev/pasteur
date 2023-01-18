from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...utils import LazyFrame
from ...utils.parser import get_params_for_pipe
from .meta import TAGS_VIEW, TAGS_VIEW_META, TAGS_VIEW_SPLIT
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node
from .utils import get_params_closure

if TYPE_CHECKING:
    from ...view import View


def _create_metadata(view: str, params: dict):
    meta_dict = get_params_for_pipe(view, params)
    return Metadata(meta_dict)


def _check_tables(metadata: Metadata, **tables: LazyFrame):
    partitions = {}
    for name, table in tables.items():
        partitions[name] = table.sample()
    metadata.check(partitions)


def create_view_pipeline(view: View):
    return PipelineMeta(
        pipeline(
            [
                node(
                    func=view.ingest,
                    name=f"ingest_{t}",
                    args=[t],
                    inputs={dep: f"{view.dataset}.{dep}" for dep in view.deps[t]},
                    namespace=f"{view}.view",
                    outputs=f"{view}.view.{t}",
                    tags=TAGS_VIEW,
                )
                for t in view.tables
            ]
            + [
                node(
                    func=_check_tables,
                    name="check_tables",
                    inputs={
                        "metadata": f"{view}.metadata",
                        **{t: f"{view}.view.{t}" for t in view.tables},
                    },
                    outputs=None,
                    namespace=f"{view}.view",
                    tags=TAGS_VIEW_META,
                )
            ]
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
                    func=_create_metadata,
                    name="create_metadata",
                    args=[view.name],
                    inputs="parameters",
                    outputs=f"{view}.metadata",
                    namespace=f"{view}",
                )
            ],
            tags=TAGS_VIEW_META,
        ),
        [D("metadata", f"{view}.metadata", ["views", "metadata", view], type="pkl")],
    )


def _filter_keys(
    view: View,
    req_splits: list[str] | None,
    ratios: dict[str, float],
    random_state: int,
    keys: LazyFrame,
):
    return view.split_keys(keys, req_splits, ratios, random_state)


def create_keys_pipeline(view: View, splits: list[str]):
    fun = get_params_closure(
        partial(_filter_keys, view, splits),
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
                name="split_keys",
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
                    func=view.filter_table,
                    args=[table],
                    name=f"filter_{table}_{split}",
                    inputs={
                        "keys": f"keys.{split}",
                        table: f"view.{table}",
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
