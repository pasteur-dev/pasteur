from ..synth import Synth

class MareSynth(Synth):
    name = "mare"
    type = "idx"

    def __init__(self, *args, _from_factory: bool = False, **kwargs) -> None:
        self.kwargs = kwargs