import logging
from typing import Any, Collection, Dict
from os import path

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet


class AddDatasetsForViewsHook:
    def __init__(self, base_location, *datasets: Dict[str, Collection[str]]) -> None:
        self.base_location = base_location
        self.datasets = {}
        for dataset in datasets:
            self.datasets.update(dataset)

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
                # Primarh dataset doesn't have prefix
                catalog.add(
                    "%s.%s" % (dataset, table),
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

                # For each materialized view table, add datasets for encoded, decoded forms
                for type in ["encoded", "decoded"]:
                    catalog.add(
                        "%s.%s_%s" % (dataset, type, table),
                        ParquetDataSet(
                            path.join(
                                self.base_location,
                                "views",
                                type,
                                dataset,
                                table + ".pq",
                            ),
                            save_args=save_args,
                        ),
                    )

                # Add pickle dataset for transformers
                catalog.add(
                    "%s.%s_%s" % (dataset, "transformer", table),
                    PickleDataSet(
                        path.join(
                            self.base_location,
                            "views",
                            "transformer",
                            dataset,
                            table + ".pkl",
                        )
                    ),
                )
