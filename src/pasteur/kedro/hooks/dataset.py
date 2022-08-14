import logging
from typing import Any, Collection, Dict, List, Union
from os import path

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet


class AddDatasetsForViewsHook:
    def __init__(
        self,
        tables: Dict[str, Collection[str]],
        algs: Collection[str],
        types: Collection[str],
        splits: Collection[str],
    ) -> None:
        self.tables = tables
        self.algs = algs
        self.types = types
        self.splits = splits

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.base_location = context.params["base_location"]

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    def get_version(self, name: str, versioned: bool):
        load_version = (
            self.load_versions.get(name, None) if self.load_versions else None
        )
        if versioned:
            return Version(load_version, self.save_version)
        return None

    def add_set(self, layer, name, path_seg, versioned=False):
        self.catalog.add(
            name,
            ParquetDataSet(
                path.join(
                    self.base_location,
                    *path_seg[:-1],
                    path_seg[-1] + ".pq",
                ),
                save_args=self.pq_save_args,
                version=self.get_version(name, versioned),
            ),
        )
        self.catalog.layers[layer].add(name)

    def add_pkl(self, layer, name, path_seg, versioned=False):
        self.catalog.add(
            name,
            PickleDataSet(
                path.join(
                    self.base_location,
                    *path_seg[:-1],
                    path_seg[-1] + ".pkl",
                ),
                version=self.get_version(name, versioned),
            ),
        )
        self.catalog.layers[layer].add(name)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: Dict[str, Any],
        conf_creds: Dict[str, Any],
        save_version: str,
        load_versions: Dict[str, str],
    ) -> None:
        # Parquet converts timestamps, but synthetic data can contain ns variations
        # which result in a loss of quality. This causes an exception.
        # By defining save args explicitly that exception is ignored.
        self.pq_save_args = {
            "coerce_timestamps": "us",
            "allow_truncated_timestamps": True,
        }
        self.catalog = catalog
        self.save_version = save_version
        self.load_versions = load_versions

        for view, tables in self.tables.items():
            for split in self.splits:
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
                for split in self.splits:
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
                    versioned=True,
                )

                for table in tables:
                    for type in ("enc", "ids"):
                        self.add_set(
                            "synth_encoded",
                            f"{view}.{alg}.{type}_{table}",
                            ["synth", type, f"{view}.{alg}", table],
                            versioned=True,
                        )

                    self.add_set(
                        "synth_decoded",
                        f"{view}.{alg}.{table}",
                        ["synth", "dec", f"{view}.{alg}", table],
                        versioned=True,
                    )
