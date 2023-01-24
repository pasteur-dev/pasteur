from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset
    from ..module import Module
    from ..view import View


def get_recommended_datasets() -> list[Dataset | View]:
    from .datasets.adult import AdultDataset
    from .datasets.mimic import MimicDataset
    from .datasets.texas import TexasDataset
    from .views.adult import TabAdultView
    from .views.mimic import MimicMmCoreView, MimicTabAdmissions
    from .views.texas import TexasChargesView, TexasBaseView, TexasBillionView

    return [
        # Views and Datasets
        AdultDataset(),
        MimicDataset(),
        TexasDataset(),
        TexasChargesView(),
        TexasBaseView(),
        TabAdultView(),
        # MimicMmCoreView,
        MimicTabAdmissions(),
        TexasBillionView(),
    ]


def get_recommended_system_modules() -> list[Module]:
    from .encoders import IdxEncoder, NumEncoder
    from .metrics.distr import ChiSquareMetric, KullbackLeiblerMetric
    from .metrics.visual import (
        CategoricalHist,
        DateHist,
        DatetimeHist,
        FixedHist,
        NumericalHist,
        OrdinalHist,
        TimeHist,
    )

    # from .synth.extern import AimSynth, PrivMrfSynth
    from .synth.privbayes import PrivBayesSynth
    from ..synth import IdentSynth
    from .transformers import (
        DatetimeTransformer,
        DateTransformer,
        FixedValueTransformer,
        IdxTransformer,
        NumericalTransformer,
        OrdinalTransformer,
        TimeTransformer,
    )

    # from .metrics.models import (
    #     ModelMetric,
    #     XGBoostlassifierModel,
    #     # RandomForestClassifierSklearn,
    # )

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
        IdentSynth.get_factory(),
        PrivBayesSynth.get_factory(),
        # AimSynth.get_factory(),
        # PrivMrfSynth.get_factory(),
        # Metrics
        ChiSquareMetric.get_factory(),
        KullbackLeiblerMetric.get_factory(),
        NumericalHist.get_factory(),
        OrdinalHist.get_factory(),
        CategoricalHist.get_factory(),
        FixedHist.get_factory(),
        DateHist.get_factory(),
        TimeHist.get_factory(),
        DatetimeHist.get_factory(),
        # ModelMetric.get_factory(XGBoostlassifierModel),
    ]


def get_recommended_modules() -> list[Module]:
    return get_recommended_datasets() + get_recommended_system_modules()
