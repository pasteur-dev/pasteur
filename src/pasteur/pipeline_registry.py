from kedro.pipeline import Pipeline, pipeline

from .kedro.pipelines.main import generate_pipelines

sel_views = []
sel_algs = ["privbayes", "ident_bhr", "ident_num"]

pipes, algs, tables, splits = generate_pipelines(sel_views, sel_algs)


def register_pipelines() -> dict[str, Pipeline]:
    return pipes
