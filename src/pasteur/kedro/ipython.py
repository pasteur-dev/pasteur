from IPython import get_ipython
from kedro.extras.extensions.ipython import reload_kedro
from kedro.framework.context import KedroContext
from kedro.framework.session.session import KedroSession
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline import Pipeline

from ..utils import str_params_to_dict, flat_params_to_dict
from .runner import SimpleRunner
from rich import reconfigure
from ..progress import PBAR_JUP_NCOLS

# Removes lint errors from VS Code
context: KedroContext = None
catalog: DataCatalog = None
session: KedroSession = None
pipelines: dict[str, Pipeline] = None

_rich_console_args = {
    "color_system": "truecolor",
    "force_terminal": True,
    "force_interactive": True,
    "force_jupyter": False,
    "width": PBAR_JUP_NCOLS,
    "height": 100,
}


def _pipe(pipe: str, params_str: str, params: dict):
    reload_kedro(extra_params=params)
    reconfigure(**_rich_console_args)
    session = get_ipython().ev("session")
    session.run(
        pipe,
        runner=SimpleRunner(pipe, params_str),
    )


def pipe(name: str, params: dict = None):
    params = params or {}
    params_dict = flat_params_to_dict(params)
    params_str = ""
    for n, p in params.items():
        params_str += f"{n}={p}"
    _pipe(name, params_str, params_dict)


def _pipe_magic(line):
    """Runs a pipeline with the provided params.

    Format: <pipeline> <param1>=<value1> <param2>=<value2>"""

    args = line.split(" ")
    pipe_str = args[0]
    if len(args) > 1:
        params_str = " ".join(args[1:])
        params = str_params_to_dict(args[1:])
    else:
        params_str = ""
        params = {}
    _pipe(pipe_str, params_str, params)


def register_kedro():
    ipy = get_ipython()
    ipy.register_magic_function(_pipe_magic, "line", "pipe")
    ipy.register_magic_function(_pipe_magic, "line", "p")
    reload_kedro()
    reconfigure(**_rich_console_args)


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
