""" Pasteur is a library for performing end-to-end data synthesis.
Gather your raw data and preprocess, synthesize, and evaluate it within a single
project.
Use the tools you're familiar with (numpy, pandas, scikit-learn, scipy) and when
your dataset grows, scale to out-of-core data by using Pasteur's parallelization 
primitives without code changes or using different libraries.
"""

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

version = metadata.version("pasteur")
__version__ = version