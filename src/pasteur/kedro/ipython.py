import logging
from kedro.extras.extensions.ipython import (
    reload_kedro,
    load_ipython_extension as kedro_load,
)
from IPython import get_ipython
from ..utils import str_params_to_dict

logger = logging.getLogger(__name__)


def pipe(line):
    """Runs a pipeline with the provided params.

    Format: <pipeline> <param1>=<value1> <param2>=<value2>"""

    args = line.split(" ")
    pipe = args[0]
    params = str_params_to_dict(args[1:]) if len(args) > 1 else {}
    reload_kedro(extra_params=params)
    session = get_ipython().ev("session")
    session.run(pipe)


def load_ipython_extension(ipython):
    kedro_load(ipython)
    ipython.register_magic_function(pipe, "line", "pipe")
