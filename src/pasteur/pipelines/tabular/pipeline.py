from typing import Collection, Dict, Tuple
from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .nodes import set_index_name_closure
from ..general.nodes import identity
from ..general.pipeline import create_node_split_keys


def create_ingest_pipelines(**kwargs) -> Pipeline:

    adult_pipeline = pipeline(
        [
            node(
                func=set_index_name_closure("id"),
                inputs=["adult.raw"],
                outputs="adult.interim",
                name="ingest_adult",
            )
        ]
    )

    adult_keys = modular_pipeline(
        pipeline([create_node_split_keys()]),
        namespace="adult",
        inputs={"keys_all": "adult.interim"},
        parameters={"random_state": "random_state"},
    )

    return {"adult": adult_pipeline + adult_keys}


def create_views_pipelines(**kwargs) -> Dict[str, Tuple[str, Pipeline]]:

    tab_adult_pipeline = pipeline(
        [
            node(
                func=identity,
                inputs=["adult.interim"],
                outputs="tab_adult.view.table",
                name="create_tab_adult",
            )
        ]
    )

    return {"tab_adult": ("adult", tab_adult_pipeline)}


def get_datasets() -> Dict[str, Collection[str]]:
    pipelines = create_views_pipelines()
    return {n: {o.split(".")[-1] for o in p[1].outputs()} for n, p in pipelines.items()}
