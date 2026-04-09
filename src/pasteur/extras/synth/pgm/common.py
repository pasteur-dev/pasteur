import logging
from typing import cast

from mbi import Dataset, Domain

from ....attribute import IdxValue
from ....marginal import MarginalOracle

logger = logging.getLogger(__name__)


class OracleDataset(Dataset):
    def __init__(
        self,
        o: MarginalOracle,
        domain: "Domain | None" = None,
        force_cache: bool = False,
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

            # o.attrs is DatasetAttributes: {table_name: Attributes}
            for table_key, table_attrs in self.attrs.items():
                if not isinstance(table_attrs, dict):
                    continue
                for attr in table_attrs.values():
                    if not hasattr(attr, 'vals'):
                        continue
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

    def _build_request(self, cols):
        """Build a MarginalRequest for the given column names at height 0."""
        req = []
        for col in cols:
            # Find the attr that owns this value name
            for table_key, table_attrs in self.attrs.items():
                if not isinstance(table_attrs, dict):
                    continue
                for attr in table_attrs.values():
                    if not hasattr(attr, 'vals'):
                        continue
                    if col in [v.name for v in attr.vals.values()]:
                        if attr.common:
                            req.append((col, 0))
                        else:
                            req.append((col, {col: 0}))
                        break
                else:
                    continue
                break
        return req

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        assert flatten

        cols = tuple(self.domain.attrs)

        if cols in self.cache:
            return self.cache[cols]
        else:
            assert (
                not self.force_cache
            ), "You set to force use cache and marginal is not in cache."

        req = self._build_request(cols)
        return self.o.process([req], postprocess=None)[0].ravel()

    def cache_marginals(self, requests: list[tuple[str, ...]]):
        non_cached_req = [req for req in requests if req not in self.cache]
        if not non_cached_req:
            return
        reqs = [self._build_request(cols) for cols in non_cached_req]
        marginals = self.o.process(reqs, postprocess=None)

        for req, mar in zip(non_cached_req, marginals):
            self.cache[req] = mar.ravel()
