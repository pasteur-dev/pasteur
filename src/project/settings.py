"""https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# FIXME: disable logging until customized logger loads
# context: currently kedro/config/logging.yml is too agressive and causes info
# messages to get printed to console. Same with rich as well
import logging
import warnings

from rich.traceback import install
from pasteur.utils.progress import RICH_TRACEBACK_ARGS

logging.captureWarnings(True)
# TODO: verify this works
# remove handlers added by the default config
logging.getLogger("kedro").handlers = []
logging.root.handlers = []
install(**RICH_TRACEBACK_ARGS)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Instantiated project hooks.
HOOKS = ()

# Installed plugins for which to disable hook auto-registration.
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# # Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {"path": "./sessions"}

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

from pasteur.extras import get_recommended_modules
from pasteur.extras.synth.pgm import AIM, MST
from pasteur.extras.views.mimic import MimicBillion

PASTEUR_MODULES = get_recommended_modules() + [
    AIM.get_factory(),
    MST.get_factory(),
    MimicBillion(),
]
