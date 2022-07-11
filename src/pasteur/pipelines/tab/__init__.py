"""
This is a boilerplate pipeline 'tab'
generated using Kedro 0.18.1
"""

from .pipeline import create_views_pipeline, create_ingest_pipelines, get_datasets

__all__ = ["create_views_pipeline", "create_ingest_pipelines", "get_datasets"]

__version__ = "0.1"
