"""This package contains reference implementations for Pasteur modules, which
may be extracted to a separate package in the future."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataset import Dataset
    from ..module import Module
    from ..view import View


def get_recommended_datasets() -> list[Dataset | View]:
    from .datasets.acs import AcsDataset
    from .datasets.adult import AdultDataset
    from .datasets.mimic import MimicDataset
    from .datasets.eicu import EicuDataset
    # from .datasets.texas import TexasDataset

    # from .datasets.boston import BostonDataset
    # from .datasets.pad import PadDataset
    from .datasets.rfel import ConsumerExpendituresDataset, FinancialDataset, StudentLoanDataset
    from .views.acs import (
        AcsEmploymentView,
        AcsIncomeView,
        AcsPersonView,
        AcsPublicCoverageView,
        AcsRelationalView,
        AcsTravelTimeView,
    )
    from .views.adult import TabAdultView
    from .views.mimic import MimicCore, MimicTabAdmissions
    from .views.eicu import EicuRelational
    from .views.rfel import ConsumerExpendituresView, FinancialClientView, StudentLoanView

    # from .views.boston import BostonView

    # from .views.texas import TexasChargesView, TexasBaseView, TexasBillionView

    return [
        # Views and Datasets
        AcsDataset(),
        AdultDataset(),
        # BostonDataset(),
        MimicDataset(),
        EicuDataset(),
        # PadDataset(),
        # TexasDataset(),
        # TexasChargesView(),
        # TexasBaseView(),
        TabAdultView(),
        AcsIncomeView(),
        AcsEmploymentView(),
        AcsPublicCoverageView(),
        AcsTravelTimeView(),
        AcsPersonView(),
        AcsRelationalView(),
        # BostonView(),
        # MimicCore,
        MimicTabAdmissions(),
        EicuRelational(),
        # TexasBillionView(),
        ConsumerExpendituresDataset(),
        StudentLoanDataset(),
        FinancialDataset(),
        ConsumerExpendituresView(),
        StudentLoanView(),
        FinancialClientView(),
    ]


def get_recommended_system_modules() -> list[Module]:
    from .encoders import IdxEncoder, NumEncoder
    from .metrics.distr import DistributionMetric
    from .metrics.visual import (
        CategoricalHist,
        DateHist,
        DatetimeHist,
        FixedHist,
        NumericalHist,
        OrdinalHist,
        TimeHist,
        SeqHist,
    )

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
        PrivBayesSynth.get_factory(rebalance=False),
        PrivBayesSynth.get_factory(name="privbayes_md", mirror_descent=True),
        PrivBayesSynth.get_factory(name="privbayes_md_s", mirror_descent={"sample": True}),
        PrivBayesSynth.get_factory(name="privbayes_rb", rebalance=True),
        PrivBayesSynth.get_factory(
            name="privbayes_rb_md", rebalance=True, mirror_descent=True
        ),
        PrivBayesSynth.get_factory(
            name="privbayes_rb_md_s", rebalance=True, mirror_descent={"sample": True}
        ),
        # Metrics
        DistributionMetric.get_factory(),
        NumericalHist.get_factory(),
        OrdinalHist.get_factory(),
        CategoricalHist.get_factory(),
        FixedHist.get_factory(),
        DateHist.get_factory(),
        TimeHist.get_factory(),
        DatetimeHist.get_factory(),
        SeqHist.get_factory(),
        # ModelMetric.get_factory(XGBoostlassifierModel),
    ]


def get_recommended_modules() -> list[Module]:
    return get_recommended_datasets() + get_recommended_system_modules()
