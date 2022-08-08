from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..dataset import Dataset, get_datasets


def create_dataset_pipeline(dataset: Dataset, tables: list[str] | None = None):

    if tables is None:
        tables = dataset.tables

    pipe = pipeline(
        [
            node(
                func=dataset.ingest_closure(t),
                inputs={dep: f"raw@{dep}" for dep in dataset.deps[t]},
                outputs=t,
            )
            for t in tables
        ]
    )

    return modular_pipeline(pipe=pipe, namespace=dataset.name)


def create_keys_pipeline(dataset: Dataset, view: str, splits: list[str]):
    fun = dataset.keys_closure(splits) if splits else dataset.keys
    req_tables = {t: f"in_{t}" for t in dataset.key_deps}
    namespaced_tables = {f"in_{t}": f"{dataset.name}.raw@{t}" for t in dataset.key_deps}

    pipe = pipeline(
        [
            node(
                func=fun,
                inputs={
                    "split": "params:ratios",
                    "random_state": "params:random_state",
                    **req_tables,
                },
                outputs={s: f"keys.{s}" for s in splits},
            )
        ]
    )

    return modular_pipeline(
        pipe=pipe,
        namespace=view,
        inputs=namespaced_tables,
        parameters={"random_state": "random_state"},
    )
