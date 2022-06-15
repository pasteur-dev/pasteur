from typing import Collection
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(
    view: str, split: str, alg: str, tables: Collection[str]
) -> Pipeline:
    return pipeline([])
