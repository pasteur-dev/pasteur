from __future__ import annotations
from typing import TYPE_CHECKING

from .mlflow import mlflow_log_model_results
from .models import get_required_types

if TYPE_CHECKING:
    import pandas as pd

    from ...transform import TableTransformer


def node_calculate_model_scores(transformer: TableTransformer, **tables: pd.DataFrame):
    from .runner import node_calculate_model_scores

    return node_calculate_model_scores(transformer, **tables)


__all__ = [
    "node_calculate_model_scores",
    "get_required_types",
    "mlflow_log_model_results",
]
