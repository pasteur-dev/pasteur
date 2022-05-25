import logging
from typing import Any, Collection, Dict, List, Union
from os import path

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet


class AddDatasetsForViewsHook:
    def __init__(
        self, base_location, *info: Union[Dict[str, Collection[str]], List[str]]
    ) -> None:
        self.base_location = base_location
        self.datasets = {}
        self.algs = []

        for item in info:
            if isinstance(item, dict):
                self.datasets.update(item)
            elif isinstance(item, list):
                self.algs.extend(item)

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
                catalog.add(
                    "%s.view.%s" % (dataset, table),
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

                # Add datasets for splits
                for split in ["wrk"]:
                    catalog.add(
                        "%s.%s.%s" % (dataset, split, table),
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

                    # For each materialized view table, add datasets for encoded, decoded forms
                    for type in ["encoded", "decoded"]:
                        catalog.add(
                            "%s.%s.%s_%s" % (dataset, split, type, table),
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

                    # Add pickle dataset for transformers
                    catalog.add(
                        "%s.%s.%s_%s" % (dataset, split, "transformer", table),
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

        for dataset, tables in self.datasets.items():
            for alg in self.algs:
                catalog.add(
                    "%s.%s.%s" % (dataset, alg, "model"),
                    PickleDataSet(
                        path.join(
                            self.base_location,
                            "synth",
                            "models",
                            "%s.%s.pkl" % (dataset, alg),
                        )
                    ),
                )
                for table in tables:
                    catalog.add(
                        "%s.%s.%s_%s" % (dataset, alg, "encoded", table),
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

                    catalog.add(
                        "%s.%s.%s" % (dataset, alg, table),
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
