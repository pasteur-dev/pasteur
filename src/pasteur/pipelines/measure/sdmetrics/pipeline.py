"""
This is a boilerplate pipeline 'measure'
generated using Kedro 0.18.1
"""

from typing import Collection
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    measure_sdmetrics_multi_table,
    measure_sdmetrics_single_table,
)


def create_pipeline(
    view: str, split: str, alg: str, tables: Collection[str]
) -> Pipeline:
    tables = [t.split(".")[-1] for t in tables]

    return pipeline(
        [
            node(
                func=measure_sdmetrics_single_table,
                inputs=[
                    f"params:{view}.metadata.tables.{t}",
                    f"{view}.{split}.{t}",
                    f"{view}.{alg}.{t}",
                ],
                outputs=f"{view}.{alg}.metrics_sdst_{t}",
            )
            for t in tables
        ]
    ) + pipeline(
        [
            node(
                func=measure_sdmetrics_multi_table,
                inputs={
                    "metadata": f"params:{view}.metadata",
                    **{f"real.{t}": f"{view}.{split}.{t}" for t in tables},
                    **{f"synth.{t}": f"{view}.{alg}.{t}" for t in tables},
                },
                outputs=f"{view}.{alg}.metrics_sdmt",
            )
        ]
    )


def create_pipeline(
    view: str, split: str, alg: str, tables: Collection[str]
) -> Pipeline:
    return pipeline([])