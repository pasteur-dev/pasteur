from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..views import View


def create_view_pipeline(view: View):
    tables = view.tables

    pipe = pipeline(
        [
            node(
                func=view.ingest_closure(t),
                inputs={dep: f"in_{dep}" for dep in view.deps[t]},
                outputs=f"view.{t}",
            )
            for t in tables
        ]
    )

    return modular_pipeline(
        pipe=pipe,
        namespace=view.name,
        inputs={f"in_{dep}": f"{view.dataset}.{dep}" for dep in view.dataset_tables},
    )


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
