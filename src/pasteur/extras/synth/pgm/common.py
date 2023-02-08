import logging
from typing import cast

from mbi import Dataset, Domain

from ....attribute import IdxValue
from ....marginal import AttrSelector, MarginalOracle

logger = logging.getLogger(__name__)


class OracleDataset(Dataset):
    def __init__(
        self,
        o: MarginalOracle,
        domain: Domain | None = None,
        force_cache: bool = True,
        cache: dict = {},
    ):
        self.o = o
        self.attrs = o.attrs
        self.cache = cache
        self.force_cache = force_cache

        if domain is not None:
            self.domain = domain
        else:
            names = []
            domains = []

            for attr in self.attrs.values():
                for val in attr.vals.values():
                    names.append(val.name)
                    domains.append(cast(IdxValue, val).get_domain(0))

            self.domain = Domain(names, domains)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        domain = self.domain.project(cols)
        return OracleDataset(
            self.o, domain, force_cache=self.force_cache, cache=self.cache
        )

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.o.get_shape()[0]

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        assert flatten

        cols = tuple(self.domain.attrs)

        if cols in self.cache:
            return self.cache[cols]
        else:
            assert not self.force_cache, "You set to force use cache and marginal is not in cache."

        req = [{col: AttrSelector(col, 0, {col: 0}) for col in self.domain.attrs}]
        return self.o.process(req, "", normalize=False)[0]

    def cache_marginals(self, requests: list[tuple[str]]):
        non_cached_req = [req for req in requests if req not in self.cache]
        marginals = self.o.process(
            [
                {col: AttrSelector(col, 0, {col: 0}) for col in cols}
                for cols in non_cached_req
            ],
            normalize=False,
        )

        for req, mar in zip(non_cached_req, marginals):
            self.cache[req] = mar
