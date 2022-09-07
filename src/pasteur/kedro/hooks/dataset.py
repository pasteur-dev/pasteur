import logging
from typing import Any
from os import path

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, Version
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet


class AddDatasetsForViewsHook:
    def __init__(
        self,
        tables: dict[str, list[str]],
        algs: list[str],
        wrk_split: str,
        ref_split: str,
        all_types: list[str],
        msr_types: list[str],
    ) -> None:
        self.tables = tables
        self.algs = algs

        self.all_types = all_types
        self.msr_types = msr_types

        self.wrk_split = wrk_split
        self.ref_split = ref_split
        self.splits = list(dict.fromkeys([wrk_split, ref_split]))

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
        if layer:
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
        if layer:
            self.catalog.layers[layer].add(name)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
        conf_catalog: dict[str, Any],
        conf_creds: dict[str, Any],
        save_version: str,
        load_versions: dict[str, str],
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
            #
            # Add view datasets
            #

            # Add metadata
            self.add_pkl(
                "metadata",
                f"{view}.metadata",
                ["views", "metadata", view],
            )

            # Add keys
            for split in self.splits:
                self.add_set(
                    "keys",
                    f"{view}.keys.{split}",
                    ["views", "keys", view, split],
                )

            # Add primary tables
            for table in tables:
                self.add_set(
                    "primary",
                    f"{view}.view.{table}",
                    ["views", "primary", view, table],
                )

            # Add transformers
            for table in tables:
                self.add_pkl(
                    "transformers",
                    f"{view}.trn.{table}",
                    ["views", "transformer", view, table],
                )

            #
            # Add tables for splits
            #
            for table in tables:
                for split in self.splits:
                    self.add_set(
                        "splits",
                        f"{view}.{split}.{table}",
                        ["views", "primary", f"{view}.{split}", table],
                    )
                    for type in ["ids", "bst", *self.all_types]:
                        match type:
                            case "ids":
                                layer = "split_ids"
                            case "bst":
                                layer = "split_transformed"
                            case others:
                                layer = "split_encoded"

                        self.add_set(
                            layer,
                            f"{view}.{split}.{type}_{table}",
                            ["views", type, f"{view}.{split}", table],
                        )

            #
            # Add algorithm datasets
            #
            for alg in self.algs:
                self.add_pkl(
                    "synth_models",
                    f"{view}.{alg}.model",
                    ["synth", "models", f"{view}.{alg}"],
                    versioned=True,
                )
                for table in tables:
                    for type in ("enc", "bst", "ids"):
                        layer = "synth_decoded" if type == "bst" else "synth_output"

                        self.add_set(
                            layer,
                            f"{view}.{alg}.{type}_{table}",
                            ["synth", type, f"{view}.{alg}", table],
                            versioned=True,
                        )

                    self.add_set(
                        "synth_reversed",
                        f"{view}.{alg}.{table}",
                        ["synth", "dec", f"{view}.{alg}", table],
                        versioned=True,
                    )

                    for type in self.msr_types:
                        self.add_set(
                            "synth_reencoded",
                            f"{view}.{alg}.{type}_{table}",
                            ["synth", type, f"{view}.{alg}", table],
                            versioned=True,
                        )

            #
            # Add measurement datasets
            #
            for table in tables:
                # Histograms
                self.add_pkl(
                    None,  # TODO: fix circular dependency of this node layer
                    f"{view}.{self.wrk_split}.msr_hst_{table}",
                    ["views", "measure", "hist", f"{view}.holder", table],
                )

                for split in [*self.algs, *self.splits]:
                    self.add_pkl(
                        "measure",
                        f"{view}.{split}.msr_viz_{table}",
                        [
                            "synth" if split in self.algs else "views",
                            "measure",
                            "visual",
                            f"{view}.{split}",
                            table,
                        ],
                        versioned=split in self.algs,
                    )

                # Models
                for alg in self.algs:
                    self.add_pkl(
                        "measure",
                        f"{view}.{alg}.msr_mdl_{table}",
                        ["synth", "measure", "models", f"{view}.{alg}", table],
                        versioned=True,
                    )

                # Distributions
                for method in ["kl", "cs"]:
                    for split in [self.ref_split, self.wrk_split, *self.algs]:
                        self.add_pkl(
                            "measure",
                            f"{view}.{split}.msr_{method}_{table}",
                            [
                                "synth" if split in self.algs else "views",
                                "measure",
                                "distr",
                                method,
                                f"{view}.{split}",
                                table,
                            ],
                            versioned=split in self.algs,
                        )
