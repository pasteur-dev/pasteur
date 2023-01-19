import logging
from copy import deepcopy
from typing import Any

import mlflow
from kedro.config import MissingConfigException
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
from mlflow.entities import RunStatus
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

from ...utils.parser import merge_dicts
from ...utils.logging import MlflowHandler
from ...utils.perf import PerformanceTracker
from .base import flatten_dict, get_run_id, get_run_name, sanitize_name
from .config import KedroMlflowConfig

logger = logging.getLogger(__name__)


class MlflowTrackingHook:
    def __init__(self):
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
        self.context = context

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any]) -> None:
        self.params = self.context.params.copy()
        self.base_location = self.params.pop("base_location")
        self.parent_name = self.params.pop("_mlflow_parent_name", "")

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
            self._logger.warning(
                "Running ingest dataset/view pipeline, disabling mlflow"
                # f"Pipeline name {pipeline_name} is not compatible with mlflow hook (<view>.<alg>.<misc>), disabling logging."
            )
            return
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
            query = f"tags.pasteur_id = '{sanitize_name(self.parent_name)}' and tags.pasteur_parent = '1'"
            parent_runs = mlflow.search_runs(
                experiment_ids=[
                    self.mlflow_config.tracking.experiment._experiment.experiment_id
                ],
                filter_string=query,
            )

            if len(parent_runs):
                parent_run_id = parent_runs["run_id"][0]  # type: ignore
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

        run_id = get_run_id(run_name, self.parent_name, finished=False)
        if run_id:
            logger.info("Resuming unfinished mlflow run.")
            mlflow.start_run(
                run_id=run_id,
                nested=bool(self.parent_name),
            )
        else:
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

        override_params = self.params.copy()
        if "_views" in override_params:
            for view in override_params.pop("_views"):
                override_params.pop(view, None)
        else:
            logger.warning(
                '"_views" key not found in params, view parameters won\'t be stripped from mlflow params.'
            )

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
        alg_overrides = params.pop("alg", {})
        params["alg._name"] = alg
        params["alg"] = merge_dicts(algs.get(alg, {}), alg_overrides)
        # filter dir, venv
        params["alg"] = {
            k: v for k, v in params["alg"].items() if k not in ("venv", "dir")
        }
        params["view"] = current_view

        # The rest of the parameters get flattened
        flattened_params = flatten_dict(
            d=params, recursive=self.recursive, sep=self.sep
        )

        # logging parameters based on defined strategy
        for k, v in flattened_params.items():
            self._log_param(k, v)

        if ratios:
            self._log_param("ratios", dict(sorted(ratios.items())))

        # And for good measure, store all parameters as a yml file
        mlflow.log_dict(self.params, f"_raw/params_all.yml")
        mlflow.log_dict(run_params, f"_raw/params_run.yml")

    @hook_impl
    def on_pipeline_error(self):
        if not self._is_mlflow_enabled:
            return

        MlflowHandler.reset_all()
        PerformanceTracker.log()
        while mlflow.active_run():
            mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

    @hook_impl
    def after_pipeline_run(self) -> None:
        if not self._is_mlflow_enabled:
            return

        MlflowHandler.reset_all()
        PerformanceTracker.log()
        while mlflow.active_run():
            mlflow.end_run()
