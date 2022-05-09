"""
This is a boilerplate pipeline 'general'
generated using Kedro 0.18.0
"""

from .pipeline import create_pipeline, create_node_split_keys
from .nodes import identity

__all__ = ["create_pipeline", "identity", "create_node_split_keys"]

__version__ = "0.1"
