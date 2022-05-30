"""
This file contains pipelines for general pre-processing of data.
"""

from .nodes import split_keys, filter_by_keys

from kedro.pipeline import Pipeline, node, pipeline


def create_split_pipeline(split, dataset, view, tables):
    return pipeline(
        [
            node(
                func=filter_by_keys,
                inputs=["%s.view.%s" % (view, t), "%s.keys_%s" % (dataset, split)],
                outputs="%s.%s.%s" % (view, split, t),
                namespace="%s.%s" % (view, split),
            )
            for t in tables
        ]
    )


def create_node_split_keys() -> Pipeline:
    return node(
        func=split_keys,
        inputs=["keys_all", "params:ratios", "params:random_state"],
        outputs=["keys_wrk", "keys_ref", "keys_dev", "keys_val"],
        name="split_keys",
    )


def create_pipeline() -> Pipeline:
    return pipeline([])
