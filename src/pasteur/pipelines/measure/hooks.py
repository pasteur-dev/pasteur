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

        # We track all of kedro params
        mod_params = deepcopy(self.params)

        # Remove unwanted parameters
        mod_params.pop("base_location", {})

        # First we start with params that should be considered together and as tags
        # such as metadata
        meta = {}

        def get_meta(params: Dict, prefix=""):
            for tag, sub_params in deepcopy(params).items():
                # Pop Metadata and add it to meta dict
                new_tag = f"{prefix}.{tag}" if prefix else tag
                if not isinstance(sub_params, dict):
                    continue
                elif "metadata" in tag:
                    meta[new_tag] = params.pop(tag)
                else:
                    get_meta(sub_params, new_tag)

        get_meta(mod_params)

        for name, metadata in meta.items():
            mlflow.log_dict(
                metadata, f"params/metadata/{name.replace('.metadata', '')}.yml"
            )

        # The rest of the parameters get flattened
        flattened_params = _flatten_dict(
            d=mod_params, recursive=self.recursive, sep=self.sep
        )

        # logging parameters based on defined strategy
        for k, v in flattened_params.items():
            self._log_param(k, v)

        # And for good measure, store all parameters as a yml file
        mlflow.log_dict(mod_params, f"params/parameters.yml")

    @hook_impl
    def before_node_run(
        self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool
    ) -> None:
        pass
