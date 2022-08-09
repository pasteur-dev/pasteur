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
        tables: Dict[str, Collection[str]],
        algs: Collection[str],
        types: Collection[str],
    ) -> None:
        self.tables = tables
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

        for view, tables in self.tables.items():
            for split in ["wrk", "ref", "val", "dev"]:
                self.add_set(
                    "keys",
                    f"{view}.keys.{split}",
                    ["views", "keys", view, split],
                )

            for table in tables:
                self.add_set(
                    "primary",
                    f"{view}.view.{table}",
                    ["views", "primary", view, table],
                )

                # Add datasets for splits
                for split in ["wrk", "ref", "val", "dev"]:
                    self.add_set(
                        "split",
                        f"{view}.{split}.{table}",
                        ["views", "primary", f"{view}.{split}", table],
                    )

                    # For each materialized view table, add datasets for encoded, decoded forms
                    for type in ["ids", *self.types]:
                        self.add_set(
                            "split_encoded",
                            f"{view}.{split}.{type}_{table}",
                            ["views", type, f"{view}.{split}", table],
                        )

                    # Add pickle dataset for transformers
                    self.add_pkl(
                        "transformers",
                        f"{view}.{split}.trn_{table}",
                        ["views", "transformer", f"{view}.{split}", table],
                    )

        for view, tables in self.tables.items():
            for alg in self.algs:
                self.add_pkl(
                    "synth_models",
                    f"{view}.{alg}.model",
                    ["synth", "models", f"{view}.{alg}"],
                )

                for table in tables:
                    for type in ("enc", "ids"):
                        self.add_set(
                            "synth_encoded",
                            f"{view}.{alg}.{type}_{table}",
                            ["synth", type, f"{view}.{alg}", table],
                        )

                    self.add_set(
                        "synth_decoded",
                        f"{view}.{alg}.{table}",
                        ["synth", "dec", f"{view}.{alg}", table],
                    )
