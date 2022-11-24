from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .meta import DatasetMeta as D
from .meta import PipelineMeta, TAGS_DATASET
from .utils import gen_closure, get_params_closure

if TYPE_CHECKING:
    from ...dataset import Dataset


def create_dataset_pipeline(
    dataset: Dataset, tables: list[str] | None = None
) -> PipelineMeta:

    if tables is None:
        tables = dataset.tables

    pipe = pipeline(
        [
            node(
                func=gen_closure(dataset.ingest, t, _fn=f"ingest_{t}"),
                inputs={dep: f"raw@{dep}" for dep in dataset.deps[t]},
                outputs=t,
                tags=TAGS_DATASET,
            )
            for t in tables
        ]
    )

    meta_tables = PipelineMeta(
        modular_pipeline(pipe=pipe, namespace=dataset.name),
        [
            D("interim", f"{dataset}.{t}", ["orig", "interim", dataset, t], type="ppq")
            for t in tables
        ],
    )

    pipe = pipeline(
        [
            node(
                func=gen_closure(dataset.keys, _fn="gen_keys"),
                inputs={dep: f"{dataset}.{dep}" for dep in dataset.key_deps},
                namespace="keys",
                outputs=f"{dataset}.keys",
                tags=TAGS_DATASET,
            )
        ]
    )

    meta_keys = PipelineMeta(
        pipe,
        [D("keys", f"{dataset}.keys", ["orig", "keys", dataset])],
    )

    return meta_tables + meta_keys
