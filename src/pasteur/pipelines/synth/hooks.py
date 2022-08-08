import logging
from typing import Any, Collection, Dict, List, Union
from os import path

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet


class AddDatasetsForViewsHook:
    def __init__(
        self,
        datasets: Dict[str, Collection[str]],
        algs: Collection[str],
        types: Collection[str],
    ) -> None:
        self.datasets = datasets
        self.algs = algs
        self.types = types

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.base_location = context.params["base_location"]

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    def add_set(self, layer, name, path_seg):
        self.catalog.add(
            name,
            ParquetDataSet(
                path.join(
                    self.base_location,
                    *path_seg[:-1],
                    path_seg[-1] + ".pq",
                ),
                save_args=self.pq_save_args,
            ),
        )
        self.catalog.layers[layer].add(name)

    def add_pkl(self, layer, name, path_seg):
        self.catalog.add(
            name,
            PickleDataSet(
                path.join(
                    self.base_location,
                    *path_seg[:-1],
                    path_seg[-1] + ".pkl",
                )
            ),
        )
        self.catalog.layers[layer].add(name)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
    ) -> None:
        # Parquet converts timestamps, but synthetic data can contain ns variations
        # which result in a loss of quality. This causes an exception.
        # By defining save args explicitly that exception is ignored.
        self.pq_save_args = {
            "coerce_timestamps": "us",
            "allow_truncated_timestamps": True,
        }
        self.catalog = catalog

        for dataset, tables in self.datasets.items():
            for split in ["wrk", "ref", "val", "dev"]:
                self.add_set(
                    "keys",
                    f"{dataset}.keys.{split}",
                    ["views", "keys", dataset, split],
                )

            for table in tables:
                self.add_set(
                    "primary",
                    f"{dataset}.view.{table}",
                    ["views", "primary", dataset, table],
                )

                # Add datasets for splits
                for split in ["wrk", "ref", "val", "dev"]:
                    self.add_set(
                        "split",
                        f"{dataset}.{split}.{table}",
                        ["views", "primary", f"{dataset}.{split}", table],
                    )

                    # For each materialized view table, add datasets for encoded, decoded forms
                    for type in ["ids", *self.types]:
                        self.add_set(
                            "split_encoded",
                            f"{dataset}.{split}.{type}_{table}",
                            ["views", type, f"{dataset}.{split}", table],
                        )

                    # Add pickle dataset for transformers
                    self.add_pkl(
                        "transformers",
                        f"{dataset}.{split}.trn_{table}",
                        ["views", "transformer", f"{dataset}.{split}", table],
                    )

        for dataset, tables in self.datasets.items():
            for alg in self.algs:
                self.add_pkl(
                    "synth_models",
                    f"{dataset}.{alg}.model",
                    ["synth", "models", f"{dataset}.{alg}"],
                )

                for table in tables:
                    for type in ("enc", "ids"):
                        self.add_set(
                            "synth_encoded",
                            f"{dataset}.{alg}.{type}_{table}",
                            ["synth", type, f"{dataset}.{alg}", table],
                        )

                    self.add_set(
                        "synth_decoded",
                        f"{dataset}.{alg}.{table}",
                        ["synth", "dec", f"{dataset}.{alg}", table],
                    )
