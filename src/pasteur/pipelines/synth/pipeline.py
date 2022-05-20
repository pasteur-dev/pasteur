from importlib.metadata import metadata
from kedro.pipeline import Pipeline, node, pipeline

from pasteur.pipelines.synth.nodes import transform_tables


def create_transform_pipeline(tables):
    return pipeline(
        [
            node(
                func=transform_tables,
                inputs=["metadata", *tables],
                outputs=["transformer", *["encoded.%s" % t for t in tables]],
            ),
            node(
                func=transform_tables,
                inputs=["metadata", *["encoded.%s" % t for t in tables]],
                outputs=["decoded.%s" % t for t in tables],
            ),
        ]
    )


def create_pipeline(tables) -> Pipeline:
    return create_transform_pipeline(tables)
