from ..synth import Synth
from ..marginal import MarginalOracle

class MareSynth(Synth):
    name = "mare"
    type = "idx"

    def __init__(
        self,
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        skip_zero_counts: bool = False,
        **kwargs
    ) -> None:
        self.kwargs = kwargs
