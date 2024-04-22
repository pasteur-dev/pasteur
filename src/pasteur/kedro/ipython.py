""" This module extends Kedro's ipython functionality. """

from pathlib import Path

from IPython.core.getipython import get_ipython
from kedro.framework.context import KedroContext
from kedro.framework.session.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io.data_catalog import DataCatalog
from kedro.pipeline import Pipeline

from ..utils.parser import flat_params_to_dict, str_params_to_dict
from ..utils.progress import PBAR_JUP_NCOLS, RICH_TRACEBACK_ARGS
from .runner import SimpleRunner

# Removes lint errors from VS Code
context: KedroContext = None  # type: ignore
catalog: DataCatalog = None  # type: ignore
session: KedroSession = None  # type: ignore
pipelines: dict[str, Pipeline] = None  # type: ignore


def _reconfigure_rich(tracebacks: bool = True):
    import rich
    from rich import get_console, reconfigure
    from rich.pretty import _ipy_display_hook

    _rich_console_args = {
        "width": PBAR_JUP_NCOLS,
        "height": 100,
    }

    reconfigure(**_rich_console_args)

    # Disable html rendering when using jupyter
    # force_jupyter=False messes with pretty print
    _console_check_buffer = rich._console._check_buffer  # type: ignore

    def non_html_check_buffer(self):
        tmp = self.is_jupyter
        self.is_jupyter = False
        _console_check_buffer.__get__(self)()
        self.is_jupyter = tmp

    rich._console._check_buffer = non_html_check_buffer.__get__(rich._console)  # type: ignore

    # Install optional rich formatter
    # controlled by the PPRINT variable
    from IPython.core.formatters import BaseFormatter

    class RichFormatter(BaseFormatter):  # type: ignore[misc]
        def __call__(self, value):
            if get_ipython().ev('globals().get("PPRINT", True)'):  # type: ignore
                return _ipy_display_hook(value, console=get_console())
            else:
                return repr(value)

    # Replace plain text formatter with rich formatter
    rich_formatter = RichFormatter()
    get_ipython().display_formatter.formatters["text/plain"] = rich_formatter  # type: ignore

    # Install tracebacks
    if tracebacks:
        import rich.traceback

        rich.traceback.install(**RICH_TRACEBACK_ARGS, console=rich._console)  # type: ignore


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


def register_kedro(path: str | None = None, tracebacks: bool = True):
    from kedro.ipython import reload_kedro
    from kedro.utils import _find_kedro_project

    top_level = Path(path) if path else Path.cwd()
    proj_path = _find_kedro_project(top_level)

    ipy = get_ipython()
    ipy.register_magic_function(_pipe_magic, "line", "pipe")  # type: ignore
    ipy.register_magic_function(_pipe_magic, "line", "p")  # type: ignore

    # Disable path message
    logging.getLogger().handlers = []
    _reconfigure_rich(tracebacks)
    if not proj_path:
        raise Exception(f"Kedro project not found along path: '{top_level}'")
    else:
        reload_kedro(proj_path)  # type: ignore

    _reconfigure_rich(tracebacks)
    ipy.register_magic_function(lambda _: reload_kedro(proj_path), "line", "reload_kedro")  # type: ignore


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
