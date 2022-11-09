def get_recommended_modules() -> list[type]:
    from .datasets.adult import AdultDataset
    from .datasets.mimic import MimicDataset
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
    from .views.adult import TabAdultView
    from .views.mimic import MimicMmCoreView, MimicTabAdmissions

    return [
        # Views and Datasets
        AdultDataset,
        MimicDataset,
        TabAdultView,
        # MimicMmCoreView,
        MimicTabAdmissions,
        # Transformers
        DatetimeTransformer,
        DateTransformer,
        TimeTransformer,
        FixedValueTransformer,
        IdxTransformer,
        OrdinalTransformer,
        NumericalTransformer,
        # Encoders
        IdxEncoder,
        NumEncoder,
        # Synthesizers
        PrivBayesSynth,
        AimSynth,
        PrivMrfSynth,
    ]
