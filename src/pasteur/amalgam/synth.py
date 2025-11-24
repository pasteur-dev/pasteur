from typing import Any, Mapping, Type, TypedDict

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


class AmalgamSynth(Synth):
    name = "amalgam"
    in_types = ["json", "flat"]
    in_sample = True
    type = "json"
    partitions = 1

    def __init__(
        self,
        model_cls: Type[MareModel],
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        rebalance: bool = True,
        unbounded_dp: bool = True,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.model_cls = model_cls
        self.rebalance = rebalance
        self.unbounded_dp = unbounded_dp

    def preprocess(self, meta: Any, data: AmalgamInput):
        self.meta = meta

    def bake(self, data: AmalgamInput): ...

    def fit(self, data: AmalgamInput):
        attrs = self.meta["flat"]["meta"]

        if self.rebalance:
            with MarginalOracle(
                data["flat"],  # type: ignore
                attrs,  # type: ignore
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
            ) as o:
                self.counts = o.get_counts(
                    desc="Calculating counts for column rebalancing"
                )
                self.attrs = rebalance_attributes(
                    self.counts[None],
                    attrs,
                    unbounded_dp=self.unbounded_dp,
                    u = 2,
                    fixed = [2, 5, 10],
                    **self.kwargs,
                )

        else:
            self.attrs = attrs

        with MarginalOracle(
            data["flat"],  # type: ignore
            self.attrs,  # type: ignore
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as o:
            kwargs = dict(self.kwargs)

            model = self.model_cls(**kwargs)
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
