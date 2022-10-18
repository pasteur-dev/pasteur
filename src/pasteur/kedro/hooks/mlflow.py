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
from kedro_mlflow.config.kedro_mlflow_config import KedroMlflowConfig
from kedro_mlflow.framework.hooks import MlflowHook
from kedro_mlflow.framework.hooks.utils import _flatten_dict
from kedro_mlflow.io.catalog.switch_catalog_logging import switch_catalog_logging

from ...logging import MlflowHandler
from ...perf import PerformanceTracker
from ...utils import dict_to_flat_params, merge_dicts

logger = logging.getLogger(__name__)


class CustomMlflowTrackingHook(MlflowHook):
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

        # store in context for interactive use
        # we use __setattr__ instead of context.mlflow because
        # the class will become frozen in kedro>=0.19
        context.__setattr__("mlflow", mlflow_config)

        self.mlflow_config = mlflow_config  # store for further reuse
        self.params = context.params
        self.base_location = self.params["base_location"]
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
            switch_catalog_logging(catalog, False)
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
            switch_catalog_logging(catalog, False)
            return

        # Setup global mlflow configuration with view as experiment name
        if self.mlflow_config.tracking.experiment.name == "Default":
            self.mlflow_config.tracking.experiment.name = current_view
        self.mlflow_config.setup(self.context)

        # params for further for node logging
        self.flatten = self.mlflow_config.tracking.params.dict_params.flatten
        self.recursive = self.mlflow_config.tracking.params.dict_params.recursive
        self.sep = self.mlflow_config.tracking.params.dict_params.sep
        self.long_params_strategy = (
            self.mlflow_config.tracking.params.long_params_strategy
        )

        run_name = self.mlflow_config.tracking.run.name
        if not run_name:
            run_name = run_params["pipeline_name"]
            for param, val in dict_to_flat_params(run_params["extra_params"]).items():
                run_name += f" {param}={val}"

        mlflow.start_run(
            run_id=self.mlflow_config.tracking.run.id,
            experiment_id=self.mlflow_config.tracking.experiment._experiment.experiment_id,
            run_name=run_name,
            nested=self.mlflow_config.tracking.run.nested,
        )
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
        # Remove unwanted parameters
        override_params.pop("base_location", {})
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
        MlflowHandler.reset_all()
        PerformanceTracker.log()
        return super().on_pipeline_error(error, run_params, pipeline, catalog)

    @hook_impl
    def after_pipeline_run(
        self, run_params: dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        MlflowHandler.reset_all()
        PerformanceTracker.log()
        return super().after_pipeline_run(run_params, pipeline, catalog)
