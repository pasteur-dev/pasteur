"""
This file contains pipelines for general pre-processing of data.
"""

from .nodes import split_keys, filter_by_keys

from kedro.pipeline import Pipeline, node, pipeline


def create_split_pipeline(split, dataset, view, tables):
    assert split in ["wrk", "ref", "dev", "val"]
    tables = [t.split(".")[-1] for t in tables]

    return pipeline(
        [
            node(
                func=filter_by_keys,
                inputs=[
                    f"{view}.view.{t}",
                    f"{dataset}.keys_{split}",
                ],
                outputs=f"{view}.{split}.{t}",
                namespace=f"{view}.{split}",
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
