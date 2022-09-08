from pathlib import Path
from IPython import get_ipython
from kedro.framework.context import KedroContext
from kedro.framework.session.session import KedroSession
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline import Pipeline
from kedro.framework.startup import bootstrap_project

from ..utils import str_params_to_dict, flat_params_to_dict
from .runner import SimpleRunner
from ..progress import PBAR_JUP_NCOLS

# Removes lint errors from VS Code
context: KedroContext = None
catalog: DataCatalog = None
session: KedroSession = None
pipelines: dict[str, Pipeline] = None


def _reconfigure_rich():
    from rich import reconfigure, _console

    _rich_console_args = {
        "width": PBAR_JUP_NCOLS,
        "height": 100,
    }

    reconfigure(**_rich_console_args)

    # Disable html rendering when using jupyter
    # force_jupyter=False messes with pretty print
    _console_check_buffer = _console._check_buffer

    def non_html_check_buffer(self):
        tmp = self.is_jupyter
        self.is_jupyter = False
        _console_check_buffer.__get__(self)()
        self.is_jupyter = tmp

    _console._check_buffer = non_html_check_buffer.__get__(_console)


def _pipe(pipe: str, params_str: str, params: dict):
    project_path = get_ipython().ev("session")._project_path

    metadata = bootstrap_project(project_path)
    session = KedroSession.create(
        metadata.package_name, project_path, env=None, extra_params=params
    )
    _reconfigure_rich()
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

    from kedro.extras.extensions.ipython import reload_kedro, _find_kedro_project
    import logging

    # Disable path message
    logging.getLogger().handlers = []
    reload_kedro(_find_kedro_project(Path.cwd()))
    _reconfigure_rich()


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
