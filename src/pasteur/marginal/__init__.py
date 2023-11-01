""" This module provides a system for marginal calculation named MarginalOracle.

Marginal oracle can perform marginal calculations out-of-core, parallelized,
while using SIMD instruction sets. """

from .numpy import AttrSelector, AttrSelectors
from .oracle import (
    CalculationInfo,
    MarginalOracle,
    MarginalRequest,
    PreprocessFun,
    PostprocessFun,
    counts_preprocess,
)
from .postprocess import ZERO_FILL, normalize, two_way_normalize, unpack

__all__ = [
    "ZERO_FILL",
    "normalize",
    "two_way_normalize",
    "unpack",
    "MarginalOracle",
    "MarginalRequest",
    "PostprocessFun",
    "AttrSelectors",
    "AttrSelector",
    "PreprocessFun",
    "CalculationInfo",
    "counts_preprocess",
]
