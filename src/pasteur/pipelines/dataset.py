from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..dataset import Dataset
from ..utils import get_params_closure


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
    fun = get_params_closure(fun, view, "ratios", "random_state")

    req_tables = {t: f"in_{t}" for t in dataset.key_deps}
    namespaced_tables = {f"in_{t}": f"{dataset.name}.raw@{t}" for t in dataset.key_deps}

    pipe = pipeline(
        [
            node(
                func=fun,
                inputs={
                    "params": "parameters",
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
        parameters={"parameters": "parameters"},
    )
