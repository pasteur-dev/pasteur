"""https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# FIXME: disable logging until customized logger loads
# context: currently kedro/config/logging.yml is too agressive and causes info
# messages to get printed to console. Same with rich as well
import logging
from rich.traceback import install

logging.getLogger().setLevel(logging.ERROR)
logging.captureWarnings(True)
install(show_locals=False)

# Instantiated project hooks.
# from iris_example.hooks import ProjectHooks
from .kedro.hooks import CustomMlflowTrackingHook, AddDatasetsForViewsHook

from .metadata import DEFAULT_TRANSFORMERS

types = list(DEFAULT_TRANSFORMERS.keys())

from .pipeline_registry import algs, tables

HOOKS = (
    AddDatasetsForViewsHook(tables, algs, types),
    CustomMlflowTrackingHook(tables, algs),
)

# Installed plugins for which to disable hook auto-registration.
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow",)

# Class that manages storing KedroSession data.
from kedro.framework.session.store import ShelveStore

SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
SESSION_STORE_ARGS = {"path": "./sessions"}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
from kedro.config import TemplatedConfigLoader

CONFIG_LOADER_CLASS = TemplatedConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "globals_pattern": "*globals.yml",
    "globals_dict": {},
}

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
