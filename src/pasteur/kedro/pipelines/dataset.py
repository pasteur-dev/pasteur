from __future__ import annotations

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...dataset import Dataset, TypedDataset
from .meta import TAGS_DATASET
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node


def create_dataset_pipeline(
    dataset: Dataset, tables: list[str] | None = None
) -> PipelineMeta:
    if not tables:
        tables = dataset.tables
    nodes = []
    outputs = []

    if isinstance(dataset, TypedDataset):
        nodes += [
            node(
                func=dataset.type,
                name=f"type_{t}",
                inputs=[f"raw@{t}"],
                outputs=f"typed.{t}",
                tags=TAGS_DATASET,
            )
            for t in dataset.raw_tables
        ]
        outputs += [
            D("typed", f"{dataset}.typed.{t}", ["orig", "typed", dataset, t], type="pq")
            for t in dataset.raw_tables
        ]
        prefix = "typed."
    else:
        prefix = "raw@"

    nodes += [
        node(
            name=f"ingest_{t}",
            func=dataset.ingest,
            args=[t],
            inputs={dep: f"{prefix}{dep}" for dep in dataset.deps[t]},
            outputs=t,
            tags=TAGS_DATASET,
        )
        for t in tables
    ]

    outputs += [
        D("interim", f"{dataset}.{t}", ["orig", "interim", dataset, t], type="pq")
        for t in tables
    ]
    meta_tables = PipelineMeta(
        modular_pipeline(pipe=pipeline(nodes), namespace=dataset.name),
        outputs,
    )

    pipe = pipeline(
        [
            node(
                func=dataset.keys,
                name="gen_keys",
                inputs={dep: f"{dataset}.{dep}" for dep in dataset.key_deps},
                namespace=str(dataset),
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
