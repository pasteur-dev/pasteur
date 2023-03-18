""" This module contains the Kedro starters for Pasteur. """

from pathlib import Path

from kedro.framework.cli.starters import KedroStarterSpec

import pasteur

PASTEUR_PATH = Path(pasteur.__file__).parent
TEMPLATE_PATH = PASTEUR_PATH / "templates" / "project"

# plugin.py
starters = [
    KedroStarterSpec(
        alias="pasteur",
        template_path=str(TEMPLATE_PATH),
    )
]
