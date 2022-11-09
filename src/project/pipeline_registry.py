from kedro.pipeline import Pipeline
from pasteur.extras import get_recommended_modules
from pasteur.kedro.pipelines import generate_pipelines

def register_pipelines() -> dict[str, Pipeline]:
    return generate_pipelines(get_recommended_modules())[0]
