from jsonschema._utils import unbool
from typing import Any, Mapping, Type, TypedDict, Literal

from pasteur.mare.synth import MareModel
from pasteur.marginal import MarginalOracle
from pasteur.synth import Synth
from pasteur.utils import LazyDataset, gen_closure
from pasteur.hierarchy import rebalance_attributes


def _repack(pid, ids, data):
    return {
        "ids": {pid: ids()},
        "data": {pid: data()},
    }


class AmalgamInput(TypedDict):
    flat: dict[str, LazyDataset]
    json: Any


class AmalgamHFParams(TypedDict):
    type: Literal["hf"]
    repo_id: str
    filename: str


MODEL_PARAMS_QWEN3: AmalgamInput = {
    "type": "hf",
    "repo_id": "Qwen/Qwen3-8B-GGUF",
    "filename": "Qwen3-8B-Q4_K_M.gguf",
}


class AmalgamMarginalParams(TypedDict):
    mode: MarginalOracle.MODES
    worker_mult: int
    min_chunk: int


MARGINAL_PARAMS_DEFAULT: AmalgamMarginalParams = {
    "mode": "out_of_core",
    "worker_mult": 1,
    "min_chunk": 100,
}


class PgmParams(TypedDict):
    etotal: float


PGM_PARAMS_DEFAULT: PgmParams = {
    "etotal": 2.0,
}


class RebalanceParams(TypedDict):
    unbounded_dp: bool
    fixed: list[int]
    u: float


REBALANCE_DEFAULT: RebalanceParams = {
    "unbounded_dp": True,
    "fixed": [4, 9, 18, 32],
    "u": 7.0,
}


class AmalgamSynth(Synth):
    name = "amalgam"
    in_types = ["json", "flat"]
    in_sample = True
    type = "json"
    partitions = 1

    def __init__(
        self,
        pgm_cls: Type[MareModel],
        pgm: PgmParams = PGM_PARAMS_DEFAULT,
        marginal: AmalgamMarginalParams = MARGINAL_PARAMS_DEFAULT,
        prompt: str = "",
        model: AmalgamHFParams = MODEL_PARAMS_QWEN3,
        rebalance: RebalanceParams | Literal[False] = REBALANCE_DEFAULT,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.pgm_cls = pgm_cls
        self.pgm = pgm
        self.marginal = marginal
        self.model = model
        self.rebalance = rebalance
        self.prompt = prompt
        # assert False

    def preprocess(self, meta: Any, data: AmalgamInput):
        self.meta = meta

    def bake(self, data: AmalgamInput): ...

    def fit(self, data: AmalgamInput):
        attrs = self.meta["flat"]["meta"]

        if self.rebalance != False:
            with MarginalOracle(
                data["flat"],  # type: ignore
                attrs,  # type: ignore
                mode=self.marginal["mode"],
                min_chunk_size=self.marginal["min_chunk"],
                max_worker_mult=self.marginal["worker_mult"],
            ) as o:
                self.counts = o.get_counts(
                    desc="Calculating counts for column rebalancing"
                )
                self.attrs = rebalance_attributes(
                    self.counts[None],
                    attrs,
                    unbounded_dp=self.rebalance["unbounded_dp"],
                    fixed=self.rebalance["fixed"],
                    u=self.rebalance["u"],
                    **self.kwargs,
                )

        else:
            self.attrs = attrs

        with MarginalOracle(
            data["flat"],  # type: ignore
            self.attrs,  # type: ignore
            mode=self.marginal["mode"],
            min_chunk_size=self.marginal["min_chunk"],
            max_worker_mult=self.marginal["worker_mult"],
        ) as o:
            kwargs = dict(self.kwargs)

            model = self.pgm_cls(**{**self.pgm, **kwargs})
            model.fit(
                data["flat"]["table"].shape[0],
                None,
                {None: self.attrs},
                o,
            )
            self.model = model

    def sample(self, n: int | None = None, data=None):
        return {
            gen_closure(_repack, pid, i, d)
            for pid, (i, d) in LazyDataset.zip(data["json"]["ids"], data["json"]["data"]).items()  # type: ignore
        }
