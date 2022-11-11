from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .meta import DatasetMeta as D
from .meta import PipelineMeta
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
            )
            for t in tables
        ]
    )

    return PipelineMeta(
        modular_pipeline(pipe=pipe, namespace=dataset.name),
        [
            D("interim", f"{dataset}.{t}", ["orig", "interim", dataset, t])
            for t in tables
        ],
    )


def create_keys_pipeline(dataset: Dataset, view: str, splits: list[str] | None):
    fun = (
        gen_closure(dataset.keys_filtered, splits, _fn="gen_keys")
        if splits
        else dataset.keys
    )
    fun = get_params_closure(fun, view, "ratios", "random_state")

    req_tables = {t: f"in_{t}" for t in dataset.key_deps}
    namespaced_tables = {f"in_{t}": f"{dataset}.raw@{t}" for t in dataset.key_deps}

    pipe = pipeline(
        [
            node(
                func=fun,
                inputs={
                    "params": "parameters",
                    **req_tables,
                },
                namespace="keys",
                outputs={s: f"keys.{s}" for s in splits},
            )
        ]
    )

    return PipelineMeta(
        modular_pipeline(
            pipe=pipe,
            namespace=view,
            inputs=namespaced_tables,
            parameters={"parameters": "parameters"},
        ),
        [
            D("keys", f"{view}.keys.{s}", ["views", "keys", view, s])
            for s in splits
        ],
    )
