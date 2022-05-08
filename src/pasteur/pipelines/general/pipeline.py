"""
This file contains pipelines for general pre-processing of data.
"""

from .nodes import identity, split_keys

from kedro.pipeline import Pipeline, node, pipeline


def create_node_split_keys(**kwargs) -> Pipeline:
    return node(
        func=split_keys,
        inputs=["keys_all", "params:ratios", "params:random_state"],
        outputs=["keys_wrk", "keys_ref", "keys_dev", "keys_val"],
        name="split_keys",
    )


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
