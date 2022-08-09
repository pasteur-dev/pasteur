from cProfile import run
from copy import deepcopy
import logging
from typing import Any, Collection, Dict, Union

import mlflow
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

from kedro_mlflow.framework.hooks.utils import _assert_mlflow_enabled, _flatten_dict
from kedro_mlflow.framework.hooks import MlflowHook

from ..utils import merge_dicts


class CustomMlflowTrackingHook(MlflowHook):
    def __init__(
        self,
        datasets: Dict[str, Collection[str]],
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
        super().after_context_created(context)
        self.params = context.params
        self.base_location = self.params["base_location"]

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        super().before_pipeline_run(run_params, pipeline, catalog)
        self._is_mlflow_enabled = _assert_mlflow_enabled(
            run_params["pipeline_name"], self.mlflow_config
        )

        if not self._is_mlflow_enabled:
            return

        self.flatten = self.mlflow_config.tracking.params.dict_params.flatten
        self.recursive = self.mlflow_config.tracking.params.dict_params.recursive
        self.sep = self.mlflow_config.tracking.params.dict_params.sep
        self.long_params_strategy = (
            self.mlflow_config.tracking.params.long_params_strategy
        )

        # Get current view and alg
        pipe_seg = run_params["pipeline_name"].split(".")
        if len(pipe_seg) < 2:
            self._logger.warn(
                f"Pipeline name {run_params['pipeline_name']} is not compatible with MlFlow hook, skipping logging parameters."
            )
            return
        current_view = pipe_seg[0]
        alg = pipe_seg[1]

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
        params["alg"] = algs.get(alg, {})

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
        mlflow.log_dict(self.params, f"params/all.yml")
        mlflow.log_dict(run_params, f"params/run.yml")

    @hook_impl
    def before_node_run(
        self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool
    ) -> None:
        pass
