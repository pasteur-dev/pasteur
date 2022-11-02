import logging
from copy import deepcopy
from typing import Any, Collection

import mlflow
from kedro.config import MissingConfigException
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from mlflow.entities import RunStatus
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

from ....logging import MlflowHandler
from ....perf import PerformanceTracker
from ....utils import dict_to_flat_params, merge_dicts
from .config import KedroMlflowConfig

logger = logging.getLogger(__name__)


def _flatten_dict(d: dict, recursive: bool = True, sep: str = ".") -> dict:
    def expand(key, value):
        if isinstance(value, dict):
            new_value = (
                _flatten_dict(value, recursive=recursive, sep=sep)
                if recursive
                else value
            )
            return [(f"{key}{sep}{k}", v) for k, v in new_value.items()]
        else:
            return [(f"{key}", value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)

def _get_git_suffix():
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return f" (git:{sha[:8]})"
    except:
        return ""

def get_run_name(pipeline: str, params: dict[str]):
    run_name = pipeline
    for param, val in dict_to_flat_params(params).items():
        if param.startswith("_"):
            continue
        run_name += f" {param}={val}"

    return run_name + _get_git_suffix()


def get_parent_name(
    pipeline: str, hyperparams: list[str], iterators: list[str], params: list[str]
):
    hyper_str = "".join(map(lambda x: f" -h {x}", hyperparams))
    iter_str = "".join(map(lambda x: f" -i {x}", iterators))
    param_str = "".join(map(lambda x: f" {x}", params))
    return f"{pipeline}{hyper_str}{iter_str}{param_str}{_get_git_suffix()}"


def _sanitize_name(name: str):
    # todo: properly escape
    return name.replace('"', '\\"').replace("'", "\\'")


def check_run_done(name: str, parent: str):
    return (
        len(
            mlflow.search_runs(
                search_all_experiments=True,
                filter_string=f"tags.pasteur_id = '{_sanitize_name(name)}' and "
                + f"tags.pasteur_pid = '{_sanitize_name(parent)}' and "
                + f"attribute.status = '{RunStatus.to_string(RunStatus.FINISHED)}'",
            )
        )
        > 0
    )


def remove_runs(parent: str):
    """Removes runs with provided parent"""

    # Delete children
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_pid = '{_sanitize_name(parent)}'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)

    # Delete parent
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_id = '{_sanitize_name(parent)}' and tags.pasteur_parent = '1'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)


class MlflowTrackingHook:
    def __init__(
        self,
        datasets: dict[str, Collection[str]],
        algs: Collection[str],
    ):
        self.datasets = datasets
        self.algs = algs

        self.recursive = True
        self.sep = "."
        self.long_parameters_strategy = "fail"
        self._is_mlflow_enabled = True

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def _log_param(self, name: str, value: dict | int | bool | str) -> None:
        str_value = str(value)
        str_value_length = len(str_value)
        if str_value_length <= MAX_PARAM_VAL_LENGTH:
            return mlflow.log_param(name, value)
        else:
            if self.long_params_strategy == "fail":
                raise ValueError(
                    f"Parameter '{name}' length is {str_value_length}, "
                    f"while mlflow forces it to be lower than '{MAX_PARAM_VAL_LENGTH}'. "
                    "If you want to bypass it, try to change 'long_params_strategy' to"
                    " 'tag' or 'truncate' in the 'mlflow.yml'configuration file."
                )
            elif self.long_params_strategy == "tag":
                self._logger.warning(
                    f"Parameter '{name}' (value length {str_value_length}) is set as a tag."
                )
                mlflow.set_tag(name, value)
            elif self.long_params_strategy == "truncate":
                self._logger.warning(
                    f"Parameter '{name}' (value length {str_value_length}) is truncated to its {MAX_PARAM_VAL_LENGTH} first characters."
                )
                mlflow.log_param(name, str_value[0:MAX_PARAM_VAL_LENGTH])

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        try:
            conf_mlflow_yml = context.config_loader.get("mlflow*", "mlflow*/**")
        except MissingConfigException:
            logger.warning(
                "No 'mlflow.yml' config file found in environment. Default configuration will be used. Use ``kedro mlflow init`` command in CLI to customize the configuration."
            )
            # we create an empty dict to have the same behaviour when the mlflow.yml
            # is commented out. In this situation there is no MissingConfigException
            # but we got an empty dict
            conf_mlflow_yml = {}
        mlflow_config = KedroMlflowConfig.parse_obj(conf_mlflow_yml)

        self.mlflow_config = mlflow_config  # store for further reuse
        self.mlflow_config.setup(context)

        self.params = context.params.copy()
        self.base_location = self.params.pop("base_location")
        self.parent_name = self.params.pop("_mlflow_parent_name", None)
        self.context = context

    @hook_impl
    def before_pipeline_run(
        self, run_params: dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:

        # Disable tracking for pipelines that don't meet criteria
        pipeline_name = run_params["pipeline_name"]
        disabled_pipelines = self.mlflow_config.tracking.disable_tracking.pipelines
        self._is_mlflow_enabled = True

        if pipeline_name in disabled_pipelines:
            self._is_mlflow_enabled = False
            logger.info(
                f"Disabled mlflow logging for blacklisted pipeline {pipeline_name}."
            )
            return

        if "ingest" in pipeline_name:
            self._is_mlflow_enabled = False
            logger.info(f"Disabled mlflow logging for ingest pipeline {pipeline_name}.")

        pipe_seg = run_params["pipeline_name"].split(".")
        if len(pipe_seg) < 2:
            self._is_mlflow_enabled = False
            self._logger.warn(
                f"Pipeline name {pipeline_name} is not compatible with mlflow hook (<view>.<alg>.<misc>), disabling logging."
            )
        else:
            current_view = pipe_seg[0]
            alg = pipe_seg[1]

        # Exit if mlflow bit was set to false
        if not self._is_mlflow_enabled:
            return

        # Setup global mlflow configuration with view as experiment name
        if self.mlflow_config.tracking.experiment.name == "Default":
            self.mlflow_config.tracking.experiment.name = current_view

        self.mlflow_config.set_experiment()

        # params for further for node logging
        self.flatten = self.mlflow_config.tracking.params.dict_params.flatten
        self.recursive = self.mlflow_config.tracking.params.dict_params.recursive
        self.sep = self.mlflow_config.tracking.params.dict_params.sep
        self.long_params_strategy = (
            self.mlflow_config.tracking.params.long_params_strategy
        )

        run_name = get_run_name(run_params["pipeline_name"], run_params["extra_params"])

        if self.parent_name:
            query = f"tags.pasteur_id = '{_sanitize_name(self.parent_name)}' and tags.pasteur_parent = '1'"
            parent_runs = mlflow.search_runs(
                experiment_ids=[
                    self.mlflow_config.tracking.experiment._experiment.experiment_id
                ],
                filter_string=query,
            )

            if len(parent_runs):
                parent_run_id = parent_runs["run_id"][0]
                logger.info(f"Nesting mlflow run under:\n{self.parent_name}")
                mlflow.start_run(
                    parent_run_id,
                )
            else:
                logger.info(f"Creating mlflow parent run:\n{self.parent_name}")
                mlflow.start_run(
                    run_name=self.parent_name,
                    experiment_id=self.mlflow_config.tracking.experiment._experiment.experiment_id,
                )
                mlflow.set_tag("pasteur_id", self.parent_name)
                mlflow.set_tag("pasteur_parent", "1")

        mlflow.start_run(
            experiment_id=self.mlflow_config.tracking.experiment._experiment.experiment_id,
            run_name=run_name,
            nested=bool(self.parent_name),
        )
        mlflow.set_tag("pasteur_id", run_name)

        if self.parent_name:
            mlflow.set_tag("pasteur_pid", self.parent_name)

        # Set tags only for run parameters that have values.
        mlflow.set_tags({k: v for k, v in run_params.items() if v})
        # add manually git sha for consistency with the journal
        # TODO : this does not take into account not committed files, so it
        # does not ensure reproducibility. Define what to do.

        self.flatten = self.mlflow_config.tracking.params.dict_params.flatten
        self.recursive = self.mlflow_config.tracking.params.dict_params.recursive
        self.sep = self.mlflow_config.tracking.params.dict_params.sep
        self.long_params_strategy = (
            self.mlflow_config.tracking.params.long_params_strategy
        )

        # We use 3 namespaces:
        # the unbounded namespace with highest priority, which is used for overrides
        # the `<view>` namespace that sets the parameters for the specific view
        # the `default` namespace that sets a baseline of parameters

        override_params = deepcopy(self.params)
        # Remove all views
        for view in self.datasets:
            override_params.pop(view, {})

        # Get default and view params
        default_params = deepcopy(self.params.get("default", {}))
        view_params = deepcopy(self.params.get(current_view, {}))

        # Create params that contain the alg data to log as parameters.
        run_params = merge_dicts(view_params, default_params, override_params)
        run_params.pop("default", {})
        params = deepcopy(run_params)
        params.pop("tables", {})
        ratios = params.pop("ratios", {})
        algs = params.pop("algs", {})
        params["alg._name"] = alg
        params["alg"] = merge_dicts(algs.get(alg, {}), params.pop("alg", {}))
        # filter dir, venv
        params["alg"] = {
            k: v for k, v in params["alg"].items() if k not in ("venv", "dir")
        }
        params["view"] = current_view

        # The rest of the parameters get flattened
        flattened_params = _flatten_dict(
            d=params, recursive=self.recursive, sep=self.sep
        )

        # logging parameters based on defined strategy
        for k, v in flattened_params.items():
            self._log_param(k, v)

        if ratios:
            self._log_param("ratios", ratios)

        # And for good measure, store all parameters as a yml file
        mlflow.log_dict(self.params, f"_params/all.yml")
        mlflow.log_dict(run_params, f"_params/run.yml")

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        t = PerformanceTracker.get("nodes")
        t.log_to_file()
        t.start(node.name.split("(")[0])

    @hook_impl
    def after_node_run(self, node: Node):
        PerformanceTracker.get("nodes").stop(node.name.split("(")[0])

    @hook_impl
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: DataCatalog,
    ):
        if not self._is_mlflow_enabled:
            return

        MlflowHandler.reset_all()
        PerformanceTracker.log()
        while mlflow.active_run():
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

    @hook_impl
    def after_pipeline_run(
        self, run_params: dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        if not self._is_mlflow_enabled:
            return

        MlflowHandler.reset_all()
        PerformanceTracker.log()
        while mlflow.active_run():
            mlflow.end_run()
