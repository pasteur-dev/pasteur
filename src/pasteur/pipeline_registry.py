from kedro.pipeline import Pipeline, pipeline

from .pipelines.main import generate_pipelines
from .synth import get_synth
from .views import get_views

sel_views = []
views = [n for n in get_synth().keys() if not sel_views or n in sel_views]
tables = {
    n: v().tables for n, v in get_views().items() if not sel_views or n in sel_views
}

sel_algs = []
algs = [a for a in get_synth().keys() if not sel_algs or a in sel_algs]


def register_pipelines() -> dict[str, Pipeline]:
    return generate_pipelines(sel_views, sel_algs)
