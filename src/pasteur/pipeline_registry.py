from kedro.pipeline import Pipeline

from .const import ALGS, VIEWS
from .kedro.pipelines.main import generate_pipelines


def register_pipelines() -> dict[str, Pipeline]:
    return generate_pipelines(VIEWS, ALGS)
