import logging
from os import path
from typing import Any, Callable

import yaml
from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.framework.project import pipelines
from kedro.io import DataCatalog, Version

from ...module import Module
from ..pipelines import generate_pipelines
from ..pipelines.main import NAME_LOCATION, RAW_LOCATION, get_view_names

logger = logging.getLogger(__name__)


class PasteurHook:
    def __init__(
        self,
        modules: list[Module]
        | Callable[[Any], list[Module]]
        | Callable[[], list[Module]],
    ) -> None:
        self.lazy_modules = modules
        self.modules = None

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.raw_location = context.params["raw_location"]
        self.base_location = context.params["base_location"]
        self.context = context

        if callable(self.lazy_modules):
            try:
                self.modules = self.lazy_modules(context)  # type: ignore
            except Exception:
                self.modules = self.lazy_modules()  # type: ignore
        else:
            self.modules = self.lazy_modules

        (
            self.pipelines,
            self.outputs,
            self.catalog_fns,
            self.parameter_fns,
        ) = generate_pipelines(self.modules, context.params)

        # FIXME: clean this up
        # Add pipelines
        pipelines._load_data()
        pipelines._content.update(self.pipelines)

        # Add metadata
        if self.parameter_fns:
            orig_params = context._extra_params
            context._extra_params = context.config_loader.get(
                *self.parameter_fns
            ).copy()
            # Add hidden dict for view names to strip params
            if orig_params:
                context._extra_params.update(orig_params)

        context._extra_params = context._extra_params.copy()
        context._extra_params["_views"] = get_view_names(self.modules)

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
                version=self.get_version(name, versioned),  # type: ignore
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
                version=self.get_version(name, versioned),  # type: ignore
            ),
        )
        if layer:
            self.catalog.layers[layer].add(name)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
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

        if catalog.layers is None:
            from collections import defaultdict

            catalog.layers = defaultdict(set)

        # Add raw datasets from packaged datasets
        # Just replace `${<folder_name>_location}` with raw/<folder_name> or that parameter
        if self.catalog_fns:
            params = self.context.params

            for folder_name, catalog_fn in self.catalog_fns:
                name = NAME_LOCATION.format(folder_name)

                with open(catalog_fn, "r") as f:
                    data = f.read()

                if folder_name:
                    dir = params.get(name, path.join(params[RAW_LOCATION], folder_name))
                    data = data.replace(f"${{{name}}}", dir)
                conf = yaml.safe_load(data)

                tmp_catalog = DataCatalog.from_config(
                    conf,
                    conf_creds,
                    load_versions,
                    save_version,
                )
                catalog.add_all(tmp_catalog._data_sets)
                for layer, children in tmp_catalog.layers.items():
                    catalog.layers[layer] = {
                        *children,
                        *catalog.layers.get(layer, set()),
                    }

        # Add pipeline outputs
        for d in self.outputs:
            match d.type:
                case "pkl":
                    self.add_pkl(d.layer, d.name, d.str_path, d.versioned)
                case "pq":
                    self.add_set(d.layer, d.name, d.str_path, d.versioned)
                case _:
                    assert False, "Not implemented"
