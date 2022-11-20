from pathlib import Path

from IPython import get_ipython
from kedro.framework.context import KedroContext
from kedro.framework.session.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline import Pipeline

from ..utils.parser import flat_params_to_dict, str_params_to_dict
from ..utils.progress import PBAR_JUP_NCOLS
from .runner import SimpleRunner

# Removes lint errors from VS Code
context: KedroContext = None  # type: ignore
catalog: DataCatalog = None  # type: ignore
session: KedroSession = None  # type: ignore
pipelines: dict[str, Pipeline] = None  # type: ignore


def _reconfigure_rich():
    from rich import _console, reconfigure

    _rich_console_args = {
        "width": PBAR_JUP_NCOLS,
        "height": 100,
    }

    reconfigure(**_rich_console_args)

    # Disable html rendering when using jupyter
    # force_jupyter=False messes with pretty print
    _console_check_buffer = _console._check_buffer  # type: ignore

    def non_html_check_buffer(self):
        tmp = self.is_jupyter
        self.is_jupyter = False
        _console_check_buffer.__get__(self)()
        self.is_jupyter = tmp

    _console._check_buffer = non_html_check_buffer.__get__(_console)  # type: ignore


def _pipe(pipe: str, params_str: str, params: dict):
    project_path = get_ipython().ev("session")._project_path  # type: ignore

    metadata = bootstrap_project(project_path)
    session = KedroSession.create(
        metadata.package_name, project_path, extra_params=params
    )
    _reconfigure_rich()
    session.run(
        pipe,
        runner=SimpleRunner(pipe, params_str),
    )


def pipe(name: str, params: dict | None = None):
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
    ipy.register_magic_function(_pipe_magic, "line", "pipe")  # type: ignore
    ipy.register_magic_function(_pipe_magic, "line", "p")  # type: ignore

    import logging

    from kedro.ipython import _find_kedro_project, reload_kedro

    # Disable path message
    logging.getLogger().handlers = []
    _reconfigure_rich()
    reload_kedro(_find_kedro_project(Path.cwd()))  # type: ignore
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
