"""
This file contains pipelines specific to processing the dataset MIMIC-IV.
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ..general.pipeline import create_pipeline_split_keys


def create_pipeline_split_mimic_keys(**kwargs) -> Pipeline:
    return modular_pipeline(
        create_pipeline_split_keys(),
        namespace="keys",
        parameters={"split_ratios": "ratios", "random_state": "random_state"},
    )


def create_pipeline(**kwargs) -> Pipeline:
    return modular_pipeline(
        create_pipeline_split_mimic_keys(),
        namespace="mimic",
        parameters={"random_state": "random_state"},
    )
