"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Create custom pipelines for your project. By default, Pasteur injects
    pipelines for synthesis executions, so this step is not required.
    """
    return {}
