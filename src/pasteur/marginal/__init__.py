""" This module provides a system for marginal calculation named MarginalOracle.

Marginal oracle can perform marginal calculations out-of-core, parallelized,
while using SIMD instruction sets. """

from .numpy import ZERO_FILL, AttrSelector, AttrSelectors
from .oracle import MarginalOracle, CalculationInfo

__all__ = [
    "AttrSelector",
    "AttrSelectors",
    "MarginalOracle",
    "CalculationInfo",
    "ZERO_FILL",
]
