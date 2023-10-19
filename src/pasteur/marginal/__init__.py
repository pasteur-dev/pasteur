""" This module provides a system for marginal calculation named MarginalOracle.

Marginal oracle can perform marginal calculations out-of-core, parallelized,
while using SIMD instruction sets. """

from .numpy import AttrSelectors
from .oracle import CalculationInfo, MarginalOracle, MarginalRequest
from .postprocess import ZERO_FILL, normalize, two_way_normalize, unpack

__all__ = [
    "ZERO_FILL",
    "normalize",
    "two_way_normalize",
    "unpack",
    "MarginalOracle",
    "MarginalRequest",
    "AttrSelectors",
    "CalculationInfo",
]
