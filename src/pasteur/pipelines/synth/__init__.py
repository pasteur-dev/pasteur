"""
This is a boilerplate pipeline 'sdv'
generated using Kedro 0.18.0
"""

from .pipeline import create_pipeline
from .hooks import AddDatasetsForViewsHook

__all__ = ["create_pipeline", "AddDatasetsForViewsHook"]

__version__ = "0.1"
