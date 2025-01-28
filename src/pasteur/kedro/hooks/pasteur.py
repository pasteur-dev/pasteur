import logging
from os import path
from typing import Any, Callable

import yaml
from kedro.config.abstract_config import MissingConfigException
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.framework.project import pipelines
from kedro.io import DataCatalog, Version
from kedro.io.memory_dataset import MemoryDataset

from ...module import Module
from ..dataset import AutoDataset, Multiset, PickleDataset
from ..pipelines import generate_pipelines
from ..pipelines.main import NAME_LOCATION, get_view_names

logger = logging.getLogger(__name__)


def _load_config(fn: str):
    import yaml

    with open(fn, encoding="utf8") as yml:
        d = yaml.safe_load(yml)

    assert isinstance(d, dict), f"Could not load config file: '{fn}'"
    return {k: v for k, v in d.items() if not k.startswith("_")}


class PasteurHook:
    def __init__(
        self,
        modules: (
            list[Module] | Callable[[Any], list[Module]] | Callable[[], list[Module]]
        ),
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
        ) = generate_pipelines(self.modules, params, self.locations)

    # Has to be first to add location hook.
    # FIXME: remove try_first
    @hook_impl(tryfirst=True)
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        try:
            # Try to use location configs for locations
            patterns = getattr(context.config_loader, "config_patterns", {})
            if "locations" not in patterns:
                patterns["locations"] = ["location*", "location*/**", "**/location*"]
            locations = context.config_loader.get("locations")
        except MissingConfigException:
            locations = {}
            logger.warn(
                f"Consider using a 'locations.yml' file in the future. Using paths from params."
            )

        def location_resolver(loc: str, default=None):
            if "_location" in loc:
                logger.warn(
                    "Found '_location' in location name. Not required in locations.yml file."
                )
            dir = locations.get(loc, default)
            if not dir:
                logger.warn(
                    f"Location '{loc}' not found in 'locations.yml'. Falling back to `parameters.yml`."
                )
                dir = context.params.get(
                    loc + "_location", context.params.get(loc, None)
                )

            assert dir, f"Dir '{loc}' not found."
            return context.project_path / dir

        # Try to register resolver with OmegaConfigLoader
        if hasattr(context.config_loader, "_register_new_resolvers"):
            getattr(context.config_loader, "_register_new_resolvers")(
                {"location": location_resolver}
            )

        self.raw_location = location_resolver("raw")
        self.base_location = location_resolver("base")
        self.locations = {k: location_resolver(k) for k in [*locations, "raw", "base"]}
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
                extra_params[name] = _load_config(view_params).copy()

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
        load_version = self.save_version
        if self.load_versions:
            load_version = self.load_versions.get(name, load_version)
        if versioned:
            return Version(load_version, self.save_version)
        return None

    def add_set(self, layer, name, path_seg, versioned=False, multi=False):
        fn = path.join(
            self.base_location,
            *path_seg[:-1],
            path_seg[-1],
        )
        if multi:
            ds = Multiset(
                fn,
                {
                    "type": AutoDataset,
                    "save_args": self.pq_save_args,
                    "metadata": {"kedro-viz": {"layer": layer}} if layer else None,
                },
                version=self.get_version(name, versioned),
            )
        else:
            ds = AutoDataset(
                fn + ".pq",
                save_args=self.pq_save_args,
                version=self.get_version(name, versioned),  # type: ignore
                metadata={"kedro-viz": {"layer": layer}} if layer else None,
            )

        self.catalog.add(
            name,
            ds,
        )
        # if layer:
        #     self.catalog.layers[layer].add(name)

    def add_pkl(self, layer, name, path_seg, versioned=False):
        self.catalog.add(
            name,
            PickleDataset(
                path.join(
                    self.base_location,
                    *path_seg[:-1],
                    path_seg[-1] + ".pkl",
                ),
                version=self.get_version(name, versioned),  # type: ignore
                metadata={"kedro-viz": {"layer": layer}} if layer else None,
            ),
        )

    def add_mem(self, layer, name):
        self.catalog.add(
            name,
            MemoryDataset(metadata={"kedro-viz": {"layer": layer}} if layer else None),  # type: ignore
        )
        # if layer:
        #     self.catalog.layers[layer].add(name)

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

        # if catalog.layers is None:
        #     from collections import defaultdict

        #     catalog.layers = defaultdict(set)

        # Add raw datasets from packaged datasets
        # Just replace `${<folder_name>_location}` with raw/<folder_name> or that parameter
        if self.catalogs:
            params = self.context.params

            for ds, folder_name, ds_catalog in self.catalogs:
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

                # Normalize old catalog names to be '{ds}.raw@{name}' unless
                # they are already that
                # TODO: find clear criteria for when to do it
                conf = {
                    f"{ds}.raw@{name}" if "." not in name else name: dataset
                    for name, dataset in conf.items()
                }

                tmp_catalog = DataCatalog.from_config(
                    conf,
                    conf_creds,
                    load_versions,
                    save_version,
                )

                # Add all traditional layers that exist
                catalog.add_all(tmp_catalog._datasets)
                depr_tag = set()
                # if hasattr(tmp_catalog, "layers"):
                #     # Passthrough layers if they are not provided through metadata
                #     cl = getattr(tmp_catalog, "layers")
                #     for layer, children in cl.items():
                #         cl[layer].update(children)
                #         depr_tag.update(children)

                # Skip constructor and set metadata attribute on datasets with
                # a raw layer. Datasets without a metadata key word argument crash otherwise.
                for n, d in tmp_catalog._datasets.items():
                    # Datasets with layer attribute are skipped
                    if n in depr_tag:
                        continue

                    if not hasattr(d, "metadata") or getattr(d, "metadata") is None:
                        setattr(d, "metadata", {"kedro-viz": {"layer": "raw"}})
                    elif "kedro-viz" not in getattr(d, "metadata"):
                        getattr(d, "metadata")["kedro-viz"] = {"layer": "raw"}
                    elif "layer" not in getattr(d, "metadata")["kedro-viz"]:
                        getattr(d, "metadata")["kedro-viz"]["layer"] = "raw"

        # Add pipeline outputs
        for d in self.outputs:
            match d.type:
                case "pkl":
                    self.add_pkl(d.layer, d.name, d.str_path, d.versioned)
                case "pq":
                    self.add_set(d.layer, d.name, d.str_path, d.versioned)
                case "mpq":
                    self.add_set(d.layer, d.name, d.str_path, d.versioned, multi=True)
                case "auto":
                    self.add_set(d.layer, d.name, d.str_path, d.versioned)
                case "multi":
                    self.add_set(d.layer, d.name, d.str_path, d.versioned, multi=True)
                case "mem":
                    self.add_mem(d.layer, d.name)
                case _:
                    assert False, "Not implemented"
