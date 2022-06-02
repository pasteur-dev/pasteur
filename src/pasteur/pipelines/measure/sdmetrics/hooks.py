from copy import deepcopy
import logging
from typing import Any, Collection, Dict

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

from kedro_mlflow.config import get_mlflow_config

from .datasets import MlflowSDMetricsDataset


class MlflowSDMetricsTrackingHook:
    def __init__(
        self,
        datasets: Dict[str, Collection[str]],
        algs: Collection[str],
    ):
        self.datasets = datasets
        self.algs = algs

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        """Hooks to be invoked after a `KedroContext` is created. This is the earliest
        hook triggered within a Kedro run. The `KedroContext` stores useful information
        such as `credentials`, `config_loader` and `env`.
        Args:
            context: The context that was created.
        """
        self.mlflow_config = get_mlflow_config(context)
        self.params = context.params
        self.base_location = self.params["base_location"]

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
    ) -> None:
        # Add Mlflow Metrics Datasets
        for dataset, tables in self.datasets.items():
            for alg in [*self.algs, "ref"]:
                #
                name = f"{dataset}.{alg}.metrics_sdmt"
                catalog.add(
                    name,
                    MlflowSDMetricsDataset(
                        prefix="multi_table",
                        local_path=f"{self.base_location}/reporting/cache/sdmetrics/multi_table.csv",
                        artifact_path=f"sdmetrics",
                    ),
                )
                catalog.layers["metrics"].add(name)

                for table in tables:
                    for metric in ["sdst"]:
                        name = f"{dataset}.{alg}.metrics_{metric}_{table}"
                        catalog.add(
                            name,
                            MlflowSDMetricsDataset(
                                prefix=table,
                                local_path=f"{self.base_location}/reporting/cache/sdmetrics/single_table/{table}.csv",
                                artifact_path=f"sdmetrics/single_table",
                            ),
                        )
                        catalog.layers["metrics"].add(name)
