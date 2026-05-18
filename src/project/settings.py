"""https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

# FIXME: disable logging until customized logger loads
# context: currently kedro/config/logging.yml is too agressive and causes info
# messages to get printed to console. Same with rich as well
import logging
import warnings

import os

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["JAX_PLATFORMS"] = "cpu"

from rich.traceback import install
from pasteur.utils.progress import RICH_TRACEBACK_ARGS

logging.captureWarnings(True)
# TODO: verify this works
# remove handlers added by the default config
logging.getLogger("kedro").handlers = []
logging.root.handlers = []

if not bool(int(os.environ.get("AGENT", 0))):
    install(**RICH_TRACEBACK_ARGS)

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", module=r"^kedro\.pipeline\.node")
warnings.filterwarnings("ignore", module=r"^kedro\.pipeline\.pipeline")

# Instantiated project hooks.
HOOKS = ()

# Installed plugins for which to disable hook auto-registration.
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow", "kedro-viz", "kedro-telemetry")

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
from kedro.config import OmegaConfigLoader

CONFIG_LOADER_CLASS = OmegaConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
# Without an explicit base_env, OmegaConfigLoader scans `conf/` directly,
# which breaks `**/parameters*` matching for files inside `conf/base/parameters/`.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # Default `destructive` merge replaces top-level dicts wholesale, so a
    # local override like `server: {url: ...}` would drop base's
    # `server.mlflow_tracking_uri`. Use soft merge for mlflow.
    "merge_strategy": {"mlflow": "soft"},
}

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog

from pasteur.extras import get_recommended_modules
from pasteur.extras.synth.pgm import AIM as RefAIM, MST as RefMST
from pasteur.extras.synth.sota import AIM, MST, PrivMRF, PrivPGD
from pasteur.extras.synth.adjuvant import (
    AdjuvantMare,
    AdjuvantSynth,
    AdjuvantMareEdp,
    AdjuvantSynthEdp,
)
from pasteur.extras.views.mimic import MimicBillion, MimicCore, MimicIcu
from pasteur.extras.encoders import JsonEncoder, FlatEncoder

# from pasteur.synth import IdentSynth
from pasteur.mare.synth import MareSynth
from pasteur.extras.synth.privbayes import PrivBayesMare, PrivBayesSynth

# from pasteur.extras.metrics.syntheval import SynthEvalMetric
from pasteur.amalgam import AmalgamSynth
from pasteur.extras.metrics.llm import LlmEvaluatorMetric

# class MareSynth(IdentSynth):
#     name = "ident_mare"
#     type = "mare"

from pasteur import IS_AGENT

PASTEUR_MODULES = get_recommended_modules() + [
    RefAIM.get_factory(),
    RefMST.get_factory(),
    AIM.get_factory(),
    MST.get_factory(),
    PrivPGD.get_factory(),
    PrivMRF.get_factory(),
    AdjuvantSynth.get_factory(),
    AdjuvantSynthEdp.get_factory(),
    MimicBillion(),
    MimicCore(),
    MimicIcu(),
    MareSynth.get_factory(PrivBayesMare),
    MareSynth.get_factory(AdjuvantMare, name="mare_adj"),
    MareSynth.get_factory(AdjuvantMareEdp, name="mare_adj_edp"),
    # SynthEvalMetric.get_factory(),
    JsonEncoder.get_factory(),
    FlatEncoder.get_factory(),
]

SKIP_LLM = os.environ.get("PASTEUR_NO_LLM_EVAL", False)

if not IS_AGENT and not SKIP_LLM:
    # LLM Evaluator takes 10-30 minutes to run, not needed when benching with LLMs
    PASTEUR_MODULES.append(LlmEvaluatorMetric.get_factory())
if not SKIP_LLM:
    PASTEUR_MODULES.append(AmalgamSynth.get_factory(PrivBayesMare))

PB_EVAL = os.environ.get("PASTEUR_PRIVBAYES_EVAL", False)
AJ_EVAL = os.environ.get("PASTEUR_ADJUVANT_EVAL", False)

if PB_EVAL:
    PASTEUR_MODULES.extend(
        [
            PrivBayesSynth.get_factory(name="privbayes_md", mirror_descent=True),
            PrivBayesSynth.get_factory(
                name="privbayes_md_s", mirror_descent={"sample": True}
            ),
            PrivBayesSynth.get_factory(name="privbayes_rb", rebalance=True),
            PrivBayesSynth.get_factory(
                name="privbayes_rb_md", rebalance=True, mirror_descent=True
            ),
            PrivBayesSynth.get_factory(
                name="privbayes_rb_md_s",
                rebalance=True,
                mirror_descent={"sample": True},
            ),
        ]
    )

if AJ_EVAL:
    PASTEUR_MODULES.extend(
        [
            AdjuvantSynth.get_factory(name="adjuvant_ab_1-way", ablation="1-way"),
            AdjuvantSynth.get_factory(
                name="adjuvant_ab_no-compression", ablation="no-compression"
            ),
            AdjuvantSynth.get_factory(
                name="adjuvant_ab_no-confidence", ablation="no-confidence"
            ),
        ]
    )
