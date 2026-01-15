from typing import Any, Literal, Mapping, Type, TypedDict

from pasteur.amalgam.llm import hold_gpu_lock, load_llm_model
from pasteur.hierarchy import rebalance_attributes
from pasteur.mare.synth import MareModel
from pasteur.marginal import MarginalOracle
from pasteur.synth import Synth
from pasteur.utils import LazyDataset, gen_closure

from .llm import AmalgamHFParams, AmalgamORParams


def _repack(pid, ids, data):
    return {
        "ids": {pid: ids()},
        "data": {pid: data()},
    }


class AmalgamInput(TypedDict):
    flat: dict[str, LazyDataset]
    json: Any


MODEL_PARAMS_QWEN3: AmalgamHFParams = {
    "type": "hf",
    "repo_id": "Qwen/Qwen3-8B-GGUF",
    "filename": "Qwen3-8B-Q4_K_M.gguf",
    "n_ctx": 40960,
    "n_gpu_layers": -1,
    "workers": 1,
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
        model: AmalgamHFParams | AmalgamORParams = MODEL_PARAMS_QWEN3,
        rebalance: RebalanceParams | Literal[False] = REBALANCE_DEFAULT,
        samples: int | None = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.pgm_cls = pgm_cls
        self.pgm = pgm
        self.marginal = marginal
        self.model = {
            **MODEL_PARAMS_QWEN3,
            **model,
        }
        self.rebalance = rebalance
        self.prompt = prompt
        self.n = samples

    def preprocess(self, meta: Any, data: AmalgamInput):
        self.meta = meta

    def bake(self, data: AmalgamInput): ...

    def fit(self, data: AmalgamInput):
        attrs = self.meta["flat"]["meta"]
        with MarginalOracle(
            data["flat"],  # type: ignore
            attrs,  # type: ignore
            mode=self.marginal["mode"],
            min_chunk_size=self.marginal["min_chunk"],
            max_worker_mult=self.marginal["worker_mult"],
        ) as o:
            self.counts = o.get_counts(desc="Calculating counts for column rebalancing")
            if self.rebalance != False:
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
            self.pgm_model = model

    def _sample(self, n: int | None = None, data=None, _llm=None):
        import pandas as pd

        from pasteur.extras.encoders import create_pydantic_model

        from .llm import load_llm_model, sample

        if not _llm:
            llm = load_llm_model(
                self.model,
                create_pydantic_model(
                    self.meta["json"]["relationships"],
                    self.meta["json"]["attrs"],
                    self.meta["json"]["ctx_attrs"],
                ),
            )
            if _llm is not None:
                _llm.update(llm)
        else:
            llm = _llm

        if n is None:
            n = self.n

        if n is None:
            n = data["flat"]["table"].shape[0]

        return sample(
            llm,
            self.prompt,
            self.counts[None],
            self.meta,
            self.pgm_model.sample(pd.RangeIndex(0, n), {}),
            data["flat"]["table"](),
            data["json"],
        )

    def sample(self, n: int | None = None, data=None, _llm=None) -> AmalgamInput:
        with hold_gpu_lock("sampling"):
            return self._sample(n=n, data=data, _llm=_llm)
