from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..module import Module
    from ..dataset import Dataset
    from ..view import View

def get_recommended_datasets() -> list[Dataset | View]:
    from .datasets.adult import AdultDataset
    from .datasets.mimic import MimicDataset

    from .views.adult import TabAdultView
    from .views.mimic import MimicMmCoreView, MimicTabAdmissions

    return [
        # Views and Datasets
        AdultDataset(),
        MimicDataset(),
        TabAdultView(),
        # MimicMmCoreView,
        MimicTabAdmissions(),
    ]

def get_recommended_system_modules() -> list[Module]:
    from .encoders import IdxEncoder, NumEncoder
    from .synth.extern import AimSynth, PrivMrfSynth
    from .synth.privbayes import PrivBayesSynth
    from .transformers import (
        DatetimeTransformer,
        DateTransformer,
        FixedValueTransformer,
        IdxTransformer,
        NumericalTransformer,
        OrdinalTransformer,
        TimeTransformer,
    )

    return [
        # Transformers
        DatetimeTransformer.get_factory(),
        DateTransformer.get_factory(),
        TimeTransformer.get_factory(),
        FixedValueTransformer.get_factory(),
        IdxTransformer.get_factory(),
        OrdinalTransformer.get_factory(),
        NumericalTransformer.get_factory(),
        # Encoders
        IdxEncoder.get_factory(),
        NumEncoder.get_factory(),
        # Synthesizers
        PrivBayesSynth.get_factory(),
        AimSynth.get_factory(),
        PrivMrfSynth.get_factory(),
    ]

def get_recommended_modules() -> list[Module]:
    return get_recommended_datasets() + get_recommended_system_modules()