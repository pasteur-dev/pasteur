import logging
from os import path
from typing import Any

from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, Version

from ..pipelines import generate_pipelines
from ..utils import get_pasteur_modules

logger = logging.getLogger("Pasteur")


class PasteurHook:
    def __init__(self, modules: list[type] | callable | None) -> None:
        self.modules = modules

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.base_location = context.params["base_location"]

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
