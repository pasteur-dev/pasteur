""" Adds great expectations command """
import importlib
from typing import List, Optional

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS
import logging

from kedro.framework.startup import ProjectMetadata

from pasteur.pipelines.measure.expectations.data_context import KedroDataContext

log = logging.getLogger(__name__)


class CLI(click.MultiCommand):
    def list_commands(self, ctx: click.Context) -> List[str]:
        # note that if --help is called this method is invoked before any flags
        # are parsed or context set.
        commands = [
            "checkpoint",
            "datasource",
            "docs",
            # "init",
            # "project",
            "store",
            "suite",
        ]

        return commands

    def get_command(self, ctx: click.Context, name: str) -> Optional[str]:
        module_name = name.replace("-", "_")
        legacy_module = ""
        if not self.is_v3_api(ctx):
            legacy_module += ".v012"

        try:
            requested_module = f"great_expectations.cli{legacy_module}.{module_name}"
            module = importlib.import_module(requested_module)
            return getattr(module, module_name)

        except ModuleNotFoundError:
            print(
                f"<red>The command `{name}` does not exist.\nPlease use one of: {self.list_commands(None)}</red>"
            )
            return None

    @staticmethod
    def print_ctx_debugging(ctx: click.Context) -> None:
        print(f"ctx.args: {ctx.args}")
        print(f"ctx.params: {ctx.params}")
        print(f"ctx.obj: {ctx.obj}")
        print(f"ctx.protected_args: {ctx.protected_args}")
        print(f"ctx.find_root().args: {ctx.find_root().args}")
        print(f"ctx.find_root().params: {ctx.find_root().params}")
        print(f"ctx.find_root().obj: {ctx.find_root().obj}")
        print(f"ctx.find_root().protected_args: {ctx.find_root().protected_args}")

    @staticmethod
    def is_v3_api(ctx: click.Context) -> bool:
        """Determine if v3 api is requested by searching context params."""
        if ctx.params:
            return ctx.params and "v3_api" in ctx.params.keys() and ctx.params["v3_api"]

        root_ctx_params = ctx.find_root().params
        return (
            root_ctx_params
            and "v3_api" in root_ctx_params.keys()
            and root_ctx_params["v3_api"]
        )


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.group(cls=CLI, name="great_expectations")
@click.option(
    "--v3-api/--v2-api",
    "v3_api",
    is_flag=True,
    default=True,
    help="Default to v3 (Batch Request) API. Use --v2-api for v2 (Batch Kwargs) API",
)
@click.option(
    "--assume-yes",
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help='Assume "yes" for all prompts.',
)
@click.pass_context
@click.pass_obj
def ge_cli(
    metadata: ProjectMetadata,
    ctx: click.Context,
    v3_api: bool,
    assume_yes: bool,
) -> None:
    """
    Great Expectations shunt to use kedro config loader. Use instead of `great_expectations` command.
    """

    # Avoid loading ge unless the command is run
    from kedro.framework.session import KedroSession
    from kedro.framework.cli.utils import KedroCliError

    from great_expectations.data_context import DataContext
    from great_expectations.cli.cli import CLIState
    from great_expectations.cli.pretty_printing import cli_message

    def _create_session(package_name: str, **kwargs):
        kwargs.setdefault("save_on_close", False)
        try:
            return KedroSession.create(package_name, **kwargs)
        except Exception as exc:
            raise KedroCliError(
                f"Unable to instantiate Kedro session.\nError: {exc}"
            ) from exc

    # State shunt that overrides config file cmd
    class CLIStateShunt(CLIState):
        def __init__(
            self,
            v3_api: bool = True,
            data_context: Optional[DataContext] = None,
            assume_yes: bool = False,
        ) -> None:
            self.v3_api = v3_api
            self._data_context = data_context
            self.assume_yes = assume_yes

        def get_data_context_from_config_file(self) -> DataContext:
            return self._data_context

    context = _create_session(metadata.package_name).load_context()
    data_context = KedroDataContext(context)

    ctx.obj = CLIStateShunt(
        v3_api=v3_api, data_context=data_context, assume_yes=assume_yes
    )

    if v3_api:
        cli_message("Using v3 (Batch Request) API")
    else:
        cli_message("Using v2 (Batch Kwargs) API")
