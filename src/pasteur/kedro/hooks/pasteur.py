import logging
from os import path
from typing import Any, Callable

import yaml
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.framework.project import pipelines
from kedro.io import DataCatalog, Version
from kedro.io.memory_dataset import MemoryDataSet

from ...module import Module
from ..dataset import FragmentedParquetDataset
from ..pipelines import generate_pipelines
from ..pipelines.main import NAME_LOCATION, get_view_names

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
        self._param_hash = None
        self._module_id = None

    def update_data(self):
        params = self.context.params
        _param_hash = hash(str(params))
        _module_id = id(self.lazy_modules)

        if _param_hash == self._param_hash and _module_id == self._module_id:
            # SKip computation, params and modules are the same
            logger.debug("Using cached pipelines")
            return

        self._param_hash = _param_hash
        self._module_id = _module_id

        if callable(self.lazy_modules):
            try:
                self.modules = self.lazy_modules(params)  # type: ignore
            except Exception:
                self.modules = self.lazy_modules()  # type: ignore
        else:
            self.modules = self.lazy_modules

        (
            self.pipelines,
            self.outputs,
            self.catalogs,
            self.parameters,
        ) = generate_pipelines(self.modules, params)

    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.raw_location = context.params["raw_location"]
        self.base_location = context.params["base_location"]
        self.context = context

        self.update_data()

        # FIXME: clean this up
        # Add pipelines
        pipelines._load_data()
        pipelines._content.update(self.pipelines)

        # Add view metadata for loaded modules
        extra_params = {}
        for name, view_params in self.parameters.items():
            # dict gets added straight away
            if isinstance(view_params, dict):
                extra_params[name] = view_params
            # string is considered to point to a file
            else:
                extra_params[name] = context.config_loader.get(view_params).copy()

        # Add hidden dict with views to remove their params in mlflow
        assert self.modules
        extra_params["_views"] = get_view_names(self.modules)

        # Restore original overrides
        if context._extra_params:
            extra_params.update(context._extra_params)

        # Apply overrides
        context._extra_params = extra_params

        setattr(context, "pasteur", self)

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
            FragmentedParquetDataset(
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

    def add_mem(self, layer, name):
        self.catalog.add(
            name,
            MemoryDataSet(),
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
        if self.catalogs:
            params = self.context.params

            for folder_name, ds_catalog in self.catalogs:
                name = NAME_LOCATION.format(folder_name)

                if isinstance(ds_catalog, str):
                    with open(ds_catalog, "r") as f:
                        data = f.read()

                    if folder_name:
                        raw_dir = params.get(
                            name, path.join(self.raw_location, folder_name)
                        )
                        data = data.replace(f"${{location}}", raw_dir)

                        data = data.replace(
                            f"${{bootstrap}}",
                            path.join(self.base_location, "bootstrap", folder_name),
                        )
                    conf = yaml.safe_load(data)
                else:
                    conf = ds_catalog

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
                case "mem":
                    self.add_mem(d.layer, d.name)
                case _:
                    assert False, "Not implemented"
