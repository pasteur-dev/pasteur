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
    ) -> None:
        self.datasets = datasets
        self.algs = algs

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.base_location = context.params["base_location"]

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

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
        save_args = {"coerce_timestamps": "us", "allow_truncated_timestamps": True}

        for dataset, tables in self.datasets.items():
            for table in tables:
                # Add primary dataset tables
                name = "%s.view.%s" % (dataset, table)
                catalog.add(
                    name,
                    ParquetDataSet(
                        path.join(
                            self.base_location,
                            "views",
                            "primary",
                            dataset,
                            table + ".pq",
                        ),
                        save_args=save_args,
                    ),
                )
                catalog.layers["primary"].add(name)

                # Add datasets for splits
                for split in ["wrk"]:
                    name = "%s.%s.%s" % (dataset, split, table)
                    catalog.add(
                        name,
                        ParquetDataSet(
                            path.join(
                                self.base_location,
                                "views",
                                "primary",
                                "%s.%s" % (dataset, split),
                                table + ".pq",
                            ),
                            save_args=save_args,
                        ),
                    )
                    catalog.layers["split"].add(name)

                    # For each materialized view table, add datasets for encoded, decoded forms
                    for type in ["encoded", "decoded"]:
                        name = "%s.%s.%s_%s" % (dataset, split, type, table)
                        catalog.add(
                            name,
                            ParquetDataSet(
                                path.join(
                                    self.base_location,
                                    "views",
                                    type,
                                    "%s.%s" % (dataset, split),
                                    table + ".pq",
                                ),
                                save_args=save_args,
                            ),
                        )
                        catalog.layers["split_%s" % type].add(name)

                    # Add pickle dataset for transformers
                    name = "%s.%s.%s_%s" % (dataset, split, "transformer", table)
                    catalog.add(
                        name,
                        PickleDataSet(
                            path.join(
                                self.base_location,
                                "views",
                                "transformer",
                                "%s.%s" % (dataset, split),
                                table + ".pkl",
                            )
                        ),
                    )
                    catalog.layers["transformers"].add(name)

        for dataset, tables in self.datasets.items():
            for alg in self.algs:
                name = "%s.%s.%s" % (dataset, alg, "model")
                catalog.add(
                    name,
                    PickleDataSet(
                        path.join(
                            self.base_location,
                            "synth",
                            "models",
                            "%s.%s.pkl" % (dataset, alg),
                        )
                    ),
                )
                catalog.layers["synth_models"].add(name)

                for table in tables:
                    name = "%s.%s.%s_%s" % (dataset, alg, "encoded", table)
                    catalog.add(
                        name,
                        ParquetDataSet(
                            path.join(
                                self.base_location,
                                "synth",
                                "encoded",
                                "%s.%s" % (dataset, alg),
                                table + ".pq",
                            ),
                            save_args=save_args,
                        ),
                    )
                    catalog.layers["synth_encoded"].add(name)

                    name = "%s.%s.%s" % (dataset, alg, table)
                    catalog.add(
                        name,
                        ParquetDataSet(
                            path.join(
                                self.base_location,
                                "synth",
                                "decoded",
                                "%s.%s" % (dataset, alg),
                                table + ".pq",
                            ),
                            save_args=save_args,
                        ),
                    )
                    catalog.layers["synth_decoded"].add(name)
