import click
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.session import KedroSession
from kedro.runner.sequential_runner import SequentialRunner

from .utils import str_params_to_dict


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@project_group.command()
@click.argument("pipeline", type=str, default=None)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
def p(pipeline, params):
    """Modified version of run with minified logging and shorter syntax"""

    param_dict = str_params_to_dict(params)

    with KedroSession.create(env=None, extra_params=param_dict) as session:
        session.run(
            tags=[],
            runner=SequentialRunner(True),
            node_names="",
            from_nodes="",
            to_nodes="",
            from_inputs="",
            to_outputs="",
            load_versions={},
            pipeline_name=pipeline,
        )
