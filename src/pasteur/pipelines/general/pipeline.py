"""
This file contains pipelines for general pre-processing of data.
"""

from .nodes import identity, split_keys

from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline_split_keys(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=identity,
                inputs=["in"],
                outputs="all",
                name="save_keys",
                tags="sql"
            ),
            node(
                func=split_keys,
                inputs=["all", "params:split_ratios"],
                outputs=["wrk", "ref", "dev", "val"],
                name="split_keys",
            )
        ]
    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
