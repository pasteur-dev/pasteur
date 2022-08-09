from kedro.pipeline import Pipeline, pipeline

from .pipelines.main import generate_pipelines

sel_views = []
sel_algs = ["privbayes", "ident_bhr"]

pipes, algs, tables = generate_pipelines(sel_views, sel_algs)


def register_pipelines() -> dict[str, Pipeline]:
    return pipes
