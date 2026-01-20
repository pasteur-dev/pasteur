"""Pasteur is a library for performing end-to-end data synthesis.
Gather your raw data and preprocess, synthesize, and evaluate it within a single
project.
Use the tools you're familiar with (numpy, pandas, scikit-learn, scipy) and when
your dataset grows, scale to out-of-core data by using Pasteur's parallelization
primitives without code changes or using different libraries.
"""

import os

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

IS_AGENT = bool(int(os.environ.get("AGENT", 0)))

if IS_AGENT:
    from logging import StreamHandler

    class log_handler(StreamHandler):
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.pop("log_time_format", None)
            super().__init__(*args, **kwargs)
else:
    from rich.logging import RichHandler

    log_handler = RichHandler

version = metadata.version("pasteur")
__version__ = version


def load_ipython_extension(ipython):
    """Allows loading ipython functionality with `load_ext pasteur`"""
    from pasteur.kedro.ipython import load_ipython_extension as ld

    ld(ipython)
