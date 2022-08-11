from IPython import get_ipython
from kedro.extras.extensions.ipython import reload_kedro
from kedro.framework.context import KedroContext
from kedro.framework.session.session import KedroSession
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline import Pipeline

from ..utils import str_params_to_dict
from .runner import SimpleRunner

# Removes lint errors from VS Code
context: KedroContext = None
catalog: DataCatalog = None
session: KedroSession = None
pipelines: dict[str, Pipeline] = None


def pipe(pipe: str, params: dict):
    reload_kedro(extra_params=params)
    session = get_ipython().ev("session")
    session.run(
        pipe,
        runner=SimpleRunner(pipe, " ".join([f"{n}={v}" for n, v in params.items()])),
    )


def _pipe_magic(line):
    """Runs a pipeline with the provided params.

    Format: <pipeline> <param1>=<value1> <param2>=<value2>"""

    args = line.split(" ")
    pipe_str = args[0]
    params = str_params_to_dict(args[1:]) if len(args) > 1 else {}
    pipe(pipe_str, params)


def register_kedro():
    reload_kedro()
    get_ipython().register_magic_function(_pipe_magic, "line", "pipe")
    get_ipython().register_magic_function(_pipe_magic, "line", "p")


def load_ipython_extension(ipython):
    register_kedro()


__all__ = [
    "load_ipython_extension",
    "register_kedro",
    "pipe",
    "context",
    "catalog",
    "session",
    "pipelines",
]
