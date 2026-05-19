"""ACS folktables-task views matching the PrivPGD paper setup.

PrivPGD (ICML 2024, https://github.com/jaabmar/private-pgd) trains five separate
folktables tasks on California ACS 2018 5-Year person-level PUMS data. We
reproduce four of them — Income, Employment, PublicCoverage, TravelTime —
on California 2014–2018 1-Year partitions concatenated, which matches the
2018 5-Year sample window and yields a comparable row count. Each view
exposes the task's feature subset plus a binary `label` derived from the
task's target column, after applying the same row filter folktables uses
(`ACSIncome.preprocess_fn`, etc.)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Callable, ClassVar

from ....utils import LazyFrame, gen_closure, get_relative_fn
from ....utils.data import LazyDataset, LazyPartition
from ....view import TabularView, View

if TYPE_CHECKING:
    import pandas as pd


# PrivPGD's "CA 2018 5-Year" sample = California PUMS data covering 2014–2018.
# We approximate it by concatenating the matching 1-Year state partitions.
_DEFAULT_STATES = ("ca",)
_DEFAULT_YEARS = (2014, 2015, 2016, 2017, 2018)


# ---- per-task filter / label functions (module-level so they pickle cleanly
# across the worker processes pasteur spawns for partition processing) ----

def _filter_income(df):
    return (df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0)


def _label_income(s):
    return s > 50_000


def _filter_employment(df):
    return (df["AGEP"] > 16) & (df["AGEP"] < 90)


def _label_eq_one(s):
    # Employment + PublicCoverage targets are both "code == 1"; ESR / PUBCOV
    # are categoricals so we coerce to string for the comparison.
    return s.astype("string") == "1"


def _filter_public_coverage(df):
    return (df["AGEP"] < 65) & (df["PINCP"] <= 30_000)


def _filter_travel_time(df):
    return (df["AGEP"] > 16) & (df["ESR"].astype("string") == "1")


def _label_travel_time(s):
    return s > 20


def _process_target(
    load: Callable,
    features: list[str],
    target_col: str,
    target_fn: Callable,
    filter_fn: Callable | None,
    needed_cols: list[str],
):
    """Load only the columns we need, filter rows, select features, label.
    Keeps SERIALNO as a column so `filter_by_keys` can match the table
    against `acs.keys` (which is SERIALNO-indexed)."""
    df = load(columns=needed_cols)
    if filter_fn is not None:
        df = df[filter_fn(df)]
    cols = list(features)
    if "SERIALNO" in df.columns and "SERIALNO" not in cols:
        cols.append("SERIALNO")
    out = df[cols]
    out = out.assign(label=target_fn(df[target_col]))
    return out


# Union of all columns the catalog declares per table — every partition the
# acs_person / acs (relational) views emit gets padded to this column set so
# the union-schema parameters.yml validates against any sample year. The
# year-versioned column pairs (TYPE/TYPEHUGQ, YBL/YRBLT, RELP/RELSHIPP,
# JWTR/JWTRNS) are coalesced in `_combine_*_versioned_cols` before padding,
# so only the unified names appear here.
_PERSON_FULL_COLS = (
    "ST", "PUMA", "PWGTP", "SPORDER", "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "POWPUMA", "RELP", "WKHP", "SEX", "RAC1P", "PINCP",
    "PUBCOV", "ESR", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
    "DEAR", "DEYE", "DREM", "FER", "GCL", "JWMNP", "JWTR", "POVPIP",
    "NWLA", "NWAB", "NWAV", "NWLK", "NWRE",
)
_HOUSEHOLD_FULL_COLS = (
    "ST", "PUMA", "NP", "HINCP", "FINCP", "TYPE", "BLD", "TEN",
    "VEH", "YRBLT", "HHL", "HHT", "HUPAC", "FS", "ACR", "BDSP", "RMSP",
    "VALP", "RNTP", "BROADBND", "ACCESS",
)


# YBL (pre-2020) uses mixed-width bin codes; YRBLT (2020+) uses the first
# year of a decade as the bin label. Project YBL onto YRBLT's decade scale
# so a single `YRBLT` column spans 2014–2023. The mapping is lossy for
# YBL 9–22 (per-year 2005–2017 bins collapse into the 2000s/2010s decades),
# which is acceptable since 2020+ rows have no finer resolution anyway.
_YBL_TO_YRBLT = {
    1: 1939,
    2: 1940, 3: 1950, 4: 1960, 5: 1970, 6: 1980, 7: 1990,
    8: 2000, 9: 2000, 10: 2000, 11: 2000, 12: 2000, 13: 2000,
    14: 2010, 15: 2010, 16: 2010, 17: 2010, 18: 2010, 19: 2010,
    20: 2010, 21: 2010, 22: 2010, 23: 2010,
}


def _combine_household_versioned_cols(df):
    """Coalesce year-versioned ACS household column pairs:
      - TYPE (≤2021) + TYPEHUGQ (2022+) share the same {1,2,3} code space →
        merged into `TYPE`.
      - YBL (≤2019) is a 22-level mixed-width bin code; YRBLT (2020+) is a
        decade-stamp year. YBL is remapped to YRBLT's decade scale and the
        two are merged into `YRBLT`.
    The redundant source columns are dropped so downstream padding and the
    schema only see the unified names."""
    if "TYPEHUGQ" in df.columns:
        if "TYPE" in df.columns:
            df = df.assign(TYPE=df["TYPE"].fillna(df["TYPEHUGQ"]))
        else:
            df = df.rename(columns={"TYPEHUGQ": "TYPE"})
        if "TYPEHUGQ" in df.columns:
            df = df.drop(columns=["TYPEHUGQ"])
    if "YBL" in df.columns:
        mapped = df["YBL"].map(_YBL_TO_YRBLT).astype("Int16")
        if "YRBLT" in df.columns:
            df = df.assign(YRBLT=df["YRBLT"].fillna(mapped))
        else:
            df = df.assign(YRBLT=mapped)
        df = df.drop(columns=["YBL"])
    return df


# RELSHIPP (2019+ codes 20-38) → RELP (≤2018 codes 0-17).
# Same-sex/opposite-sex spouse splits (21+23) and partner splits (22+24)
# collapse to RELP's combined spouse (1) and unmarried partner (13). RELSHIPP
# dropped "roomer/boarder" (RELP 11), folding it into "other nonrelative" (36).
_RELSHIPP_TO_RELP = {
    20: 0,
    21: 1, 22: 13, 23: 1, 24: 13,
    25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
    30: 7, 31: 8, 32: 9, 33: 10,
    34: 12, 35: 14, 36: 15,
    37: 16, 38: 17,
}
# RELP code 11 (roomer/boarder) was retired in RELSHIPP — fold it into 15
# (other nonrelative) so the unified column has one codebook across all years.
_RELP_NORMALIZE = {11: 15}

# The 2019 PUMS redesign reshuffled JWTR codes 2-5 into different transit
# categories (bus vs trolley split, subway/streetcar/light rail/long-distance
# rebinned). Collapse all rail variants into a single "rail" bucket (3) so
# JWTR and JWTRNS agree code-for-code; bus stays at 2.
#   1=car · 2=bus · 3=rail · 6=ferry · 7=taxi · 8=motorcycle · 9=bike
#   10=walk · 11=worked-from-home · 12=other
_JWTR_TO_UNIFIED = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
_JWTRNS_TO_UNIFIED = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}


def _combine_person_versioned_cols(df):
    """Coalesce year-versioned ACS person column pairs into a single column each:
      - RELP (≤2018) + RELSHIPP (2019+) → `RELP`. RELSHIPP's 20-38 codes are
        projected onto RELP's 0-17 codebook; same-sex/opposite-sex spouse and
        partner distinctions are lost (not available pre-2019 anyway).
      - JWTR (≤2018) + JWTRNS (2019+) → `JWTR`. Codes 2-5 mean different
        transit modes in each era; both are collapsed to bus (2) + rail (3).
    Drops the source columns so downstream padding and schema only see the
    unified names."""
    if "RELSHIPP" in df.columns:
        mapped = df["RELSHIPP"].map(_RELSHIPP_TO_RELP).astype("Int8")
        if "RELP" in df.columns:
            df = df.assign(RELP=df["RELP"].fillna(mapped))
        else:
            df = df.assign(RELP=mapped)
        df = df.drop(columns=["RELSHIPP"])
    if "RELP" in df.columns:
        df = df.assign(RELP=df["RELP"].replace(_RELP_NORMALIZE))
    if "JWTRNS" in df.columns:
        mapped = df["JWTRNS"].map(_JWTRNS_TO_UNIFIED).astype("Int8")
        if "JWTR" in df.columns:
            df = df.assign(JWTR=df["JWTR"].fillna(mapped))
        else:
            df = df.assign(JWTR=mapped)
        df = df.drop(columns=["JWTRNS"])
    if "JWTR" in df.columns:
        df = df.assign(JWTR=df["JWTR"].replace(_JWTR_TO_UNIFIED))
    return df


def _pad_missing(df, cols):
    import pandas as pd

    missing = [c for c in cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = pd.NA
    return df


def _add_state_year(load: Callable, state: str, year: str):
    """Materialize a partition, tag it with `state` (postal abbrev) and `year`
    (int), and pad in any catalog-declared columns the partition's year
    doesn't carry. RELP/RELSHIPP and JWTR/JWTRNS are unified first."""
    df = load()
    df = _combine_person_versioned_cols(df)
    df = _pad_missing(df, _PERSON_FULL_COLS)
    if "state" not in df.columns:
        df = df.assign(state=state)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    return df


def _process_household(load: Callable, state: str, year: str):
    """Materialize a household partition keyed by SERIALNO."""
    df = load()
    df = _combine_household_versioned_cols(df)
    df = _pad_missing(df, _HOUSEHOLD_FULL_COLS)
    if "state" not in df.columns:
        df = df.assign(state=state)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    if df.index.name != "SERIALNO":
        df = df.set_index("SERIALNO")
    return df


def _subset(lf: LazyFrame, target_pids: frozenset[str]) -> LazyFrame:
    """Return a LazyFrame containing only the partitions in `target_pids`."""
    if not lf.partitioned:
        return lf
    parts = {pid: p for pid, p in lf.items() if pid in target_pids}
    return LazyDataset(lf.merged_load, parts)


class _AcsTaskView(TabularView):
    """Base class for the four PrivPGD folktables-task views."""

    dataset = "acs"
    deps = {"table": ["person"]}

    _features: ClassVar[list[str]] = []
    _target_col: ClassVar[str] = ""
    _filter_extra_cols: ClassVar[list[str]] = []
    _target_fn: ClassVar[Callable[["pd.Series"], "pd.Series"]] = staticmethod(
        lambda s: s
    )
    _filter_fn: ClassVar[Callable[["pd.DataFrame"], "pd.Series"] | None] = None

    def __init__(
        self,
        states: tuple[str, ...] = _DEFAULT_STATES,
        years: tuple[int, ...] = _DEFAULT_YEARS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._states = tuple(s.lower() for s in states)
        self._years = tuple(years)
        self._targets = frozenset(
            f"{s}_{y}" for s in self._states for y in self._years
        )

    @property
    def _needed_cols(self) -> list[str]:
        return sorted(
            set(self._features)
            | {self._target_col, "SERIALNO"}
            | set(self._filter_extra_cols)
        )

    def query(self, name: str, **tables: LazyFrame):
        assert name == "table"
        person = tables["person"]

        if not person.partitioned:
            return _process_target(
                lambda **kw: person(**kw),
                self._features,
                self._target_col,
                self._target_fn,
                self._filter_fn,
                self._needed_cols,
            )

        keys = set(person.keys())
        missing = self._targets - keys
        assert not missing, (
            f"View {self.name} expects partitions {sorted(missing)} in "
            f"acs.person; available: {sorted(keys)[:5]}…"
        )

        return {
            pid: gen_closure(
                _process_target,
                fun,
                self._features,
                self._target_col,
                self._target_fn,
                self._filter_fn,
                self._needed_cols,
            )
            for pid, fun in person.items()
            if pid in self._targets
        }

    def split_keys(self, keys, req_splits, splits, random_state):
        return super().split_keys(
            _subset(keys, self._targets), req_splits, splits, random_state
        )

    def filter_table(self, name: str, keys: LazyFrame, **tables: LazyFrame):
        return super().filter_table(
            name,
            _subset(keys, self._targets),
            **{k: _subset(v, self._targets) for k, v in tables.items()},
        )


class AcsIncomeView(_AcsTaskView):
    """ACSIncome: predict person income > 50k from 10 demographic/work features."""

    name = "acs_income"
    parameters = get_relative_fn("parameters_income.yml")

    _features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP",
                 "SEX", "RAC1P"]
    _target_col = "PINCP"
    _filter_extra_cols = ["AGEP", "PINCP", "WKHP"]
    _target_fn = staticmethod(_label_income)
    _filter_fn = staticmethod(_filter_income)


class AcsEmploymentView(_AcsTaskView):
    """ACSEmployment: predict employment status (ESR == 1) from 16 features."""

    name = "acs_employment"
    parameters = get_relative_fn("parameters_employment.yml")

    _features = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG",
                 "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"]
    _target_col = "ESR"
    _filter_extra_cols = ["AGEP"]
    _target_fn = staticmethod(_label_eq_one)
    _filter_fn = staticmethod(_filter_employment)


class AcsPublicCoverageView(_AcsTaskView):
    """ACSPublicCoverage: predict public health insurance (PUBCOV == 1)."""

    name = "acs_public_coverage"
    parameters = get_relative_fn("parameters_public_coverage.yml")

    _features = ["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIG",
                 "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "PINCP",
                 "ESR", "ST", "FER", "RAC1P"]
    _target_col = "PUBCOV"
    _filter_extra_cols = ["AGEP", "PINCP"]
    _target_fn = staticmethod(_label_eq_one)
    _filter_fn = staticmethod(_filter_public_coverage)


class AcsTravelTimeView(_AcsTaskView):
    """ACSTravelTime: predict commute > 20 min (JWMNP > 20) for employed adults."""

    name = "acs_travel_time"
    parameters = get_relative_fn("parameters_travel_time.yml")

    # Uses 2018-era columns: RELP (not RELSHIPP), JWTR (not JWTRNS).
    _features = ["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "MIG", "RELP",
                 "RAC1P", "PUMA", "ST", "CIT", "OCCP", "JWTR", "POWPUMA",
                 "POVPIP"]
    _target_col = "JWMNP"
    _filter_extra_cols = ["AGEP", "ESR"]
    _target_fn = staticmethod(_label_travel_time)
    _filter_fn = staticmethod(_filter_travel_time)


# Each year is split into this many SERIALNO-hash chunks so the grouped-by-year
# view emits ~6× as many partitions, parallelizing per-partition processing
# without going back to the original 459-way (state, year) partitioning. The
# hash assigns every SERIALNO (household) to one chunk, so all of a
# household's persons stay together in the same partition.
_CHUNKS_PER_YEAR = 6


def _serialno_chunk_idx(serialno, n_chunks: int) -> int:
    """Deterministic chunk assignment for a single SERIALNO. Stable across
    runs/processes (uses md5, not Python `hash`, whose seed varies)."""
    return int(hashlib.md5(str(serialno).encode()).hexdigest()[:8], 16) % n_chunks


def _serialno_chunk_mask(serialnos, chunk_idx: int, n_chunks: int):
    """Boolean mask: True for SERIALNOs whose hash assigns them to `chunk_idx`."""
    return serialnos.map(lambda s: _serialno_chunk_idx(s, n_chunks)) == chunk_idx


def _state_prefix_serialno(s, state: str):
    """Prefix SERIALNO with state code. Pre-2018 PUMS SERIALNOs are only unique
    within a state-year, so the prefix is required when concatenating across
    states so household-level keys stay collision-free."""
    return state + "_" + s.astype("string")


def _concat_year_for_view(state_funs, year: str, chunk_idx: int, n_chunks: int):
    """Concatenate all states' person partitions for one year, state-prefix
    SERIALNO so it's unique across states, then keep rows whose SERIALNO
    hashes to `chunk_idx`. Households stay together in the same chunk.
    States are loaded in the order given; callers should pass them sorted
    alphabetically for determinism."""
    import pandas as pd

    parts = []
    for state, fun in state_funs:
        df = fun()
        df = _combine_person_versioned_cols(df)
        df = _pad_missing(df, _PERSON_FULL_COLS)
        if "state" not in df.columns:
            df = df.assign(state=state)
        df = df.assign(SERIALNO=_state_prefix_serialno(df["SERIALNO"], state))
        df = df.loc[_serialno_chunk_mask(df["SERIALNO"], chunk_idx, n_chunks).values]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    df.index = df.index.astype("int64").rename("id")
    return df


def _concat_keys_for_year(state_loaders, chunk_idx: int, n_chunks: int):
    """Mirror `_concat_year_for_view` for the keys frame. `acs.keys` is
    SERIALNO-indexed (one row per household) — state-prefix the SERIALNO
    and keep entries whose hash assigns them to `chunk_idx`, matching
    `_concat_year_for_view`."""
    import pandas as pd

    parts = []
    for state, ld in state_loaders:
        df = ld()
        df.index = pd.Index(
            _state_prefix_serialno(df.index.to_series(), state).values,
            name="SERIALNO",
        )
        df = df.loc[_serialno_chunk_mask(df.index.to_series(), chunk_idx, n_chunks).values]
        parts.append(df)
    return pd.concat(parts)


def _is_state_year_pid(pid: str) -> bool:
    """True for `{state}_{year}` source pids (e.g. `ca_2018`); False for the
    regrouped `{year}_{chunk}` pids we emit (e.g. `2018_0`).  Used by
    `_group_by_year` to stay idempotent when called twice on the same lf."""
    first = pid.partition("_")[0]
    # state codes are alphabetic (e.g. `ca`, `ny`); years are 4-digit numbers
    return not (len(first) == 4 and first.isdigit())


def _group_by_year(lf: LazyFrame, table_view: bool) -> LazyFrame:
    """Re-partition a `{state}_{year}`-keyed LazyFrame into ``_CHUNKS_PER_YEAR``
    chunks per year (pids `{year}_{chunk}`). `table_view=True` builds the
    per-row state column and pads the union schema; otherwise just
    concatenates index-only key frames.

    Idempotent: if the pids are already in `{year}_{chunk}` form (e.g.
    `filter_table` receives keys that `split_keys` has already regrouped),
    pass through."""
    if not lf.partitioned:
        return lf
    sample = next(iter(lf.keys()))
    if not _is_state_year_pid(sample):
        return lf
    by_year: dict[str, list] = {}
    for pid, p in lf.items():
        state, year = pid.rsplit("_", 1)
        by_year.setdefault(year, []).append((state, p))
    new_parts = {}
    for year, items in by_year.items():
        items.sort(key=lambda sp: sp[0])  # alphabetical, deterministic
        for chunk_idx in range(_CHUNKS_PER_YEAR):
            pid = f"{year}_{chunk_idx}"
            if table_view:
                new_parts[pid] = LazyPartition(
                    _concat_year_for_view,
                    None,
                    items,
                    year,
                    chunk_idx,
                    _CHUNKS_PER_YEAR,
                )
            else:
                new_parts[pid] = LazyPartition(
                    _concat_keys_for_year, None, items, chunk_idx, _CHUNKS_PER_YEAR
                )
    return LazyDataset(lf.merged_load, new_parts)


class AcsPersonView(TabularView):
    """Full per-person table, regrouped to one partition per year.

    States within a year are concatenated (alphabetical order) into a single
    partition with a fresh `id` index. `state` and `year` columns survive on
    every row. Reduces 459 → ~9 partitions and trims the per-partition write
    overhead. `split_keys` / `filter_table` are overridden to perform the same
    regrouping on the keys side so the `id`s line up."""

    name = "acs_person"
    dataset = "acs"
    deps = {"table": ["person"]}
    parameters = get_relative_fn("parameters_person.yml")

    def query(self, name, **tables: LazyFrame):
        assert name == "table"
        person = tables["person"]
        regrouped = _group_by_year(person, table_view=True)
        if not regrouped.partitioned:
            return regrouped()
        return {pid: regrouped[pid] for pid in regrouped.keys()}

    def split_keys(self, keys, req_splits, splits, random_state):
        return super().split_keys(
            _group_by_year(keys, table_view=False),
            req_splits,
            splits,
            random_state,
        )

    def filter_table(self, name, keys: LazyFrame, **tables: LazyFrame):
        return super().filter_table(
            name,
            _group_by_year(keys, table_view=False),
            **tables,
        )


def _concat_person_relational_for_year(state_funs, year: str, chunk_idx: int, n_chunks: int):
    """Concatenate person partitions across states for one year, state-prefix
    SERIALNO, and keep persons whose SERIALNO hashes to `chunk_idx`. All of
    a household's persons stay in the same chunk."""
    import pandas as pd

    parts = []
    for state, fun in state_funs:
        df = fun()
        df = _combine_person_versioned_cols(df)
        df = _pad_missing(df, _PERSON_FULL_COLS)
        if "state" not in df.columns:
            df = df.assign(state=state)
        df = df.assign(SERIALNO=_state_prefix_serialno(df["SERIALNO"], state))
        df = df.loc[_serialno_chunk_mask(df["SERIALNO"], chunk_idx, n_chunks).values]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    df.index = df.index.astype("int64").rename("id")
    return df


def _concat_household_for_year(state_funs, year: str, chunk_idx: int, n_chunks: int):
    """Concatenate household partitions across states for one year,
    state-prefix the SERIALNO index, and keep households whose SERIALNO
    hashes to `chunk_idx`. Aligned with `_concat_person_relational_for_year`
    via the shared hash, so household chunk[i] always matches person
    chunk[i]'s SERIALNO set."""
    import pandas as pd

    parts = []
    for state, fun in state_funs:
        df = fun()
        df = _combine_household_versioned_cols(df)
        df = _pad_missing(df, _HOUSEHOLD_FULL_COLS)
        if "state" not in df.columns:
            df = df.assign(state=state)
        if df.index.name != "SERIALNO":
            df = df.set_index("SERIALNO")
        df.index = pd.Index(
            _state_prefix_serialno(df.index.to_series(), state).values,
            name="SERIALNO",
        )
        df = df.loc[_serialno_chunk_mask(df.index.to_series(), chunk_idx, n_chunks).values]
        parts.append(df)
    df = pd.concat(parts)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    return df


def _group_relational_by_year(lf: LazyFrame, kind: str) -> LazyFrame:
    """Re-partition a `{state}_{year}`-keyed LazyFrame into ``_CHUNKS_PER_YEAR``
    SERIALNO-hash chunks per year (pids `{year}_{chunk}`). `kind` is
    `"person"`, `"household"`, or `"keys"`. The shared hash function means
    all three sides chunk to the same SERIALNO set per `{year}_{chunk}`,
    so the household-person join (and the keys-table filter) stays valid
    without cross-table loads at partition time.

    Idempotent: pids already in `{year}_{chunk}` form pass through."""
    if not lf.partitioned:
        return lf
    sample = next(iter(lf.keys()))
    if not _is_state_year_pid(sample):
        return lf

    by_year: dict[str, list] = {}
    for pid, p in lf.items():
        state, year = pid.rsplit("_", 1)
        by_year.setdefault(year, []).append((state, p))

    new_parts: dict[str, LazyPartition] = {}
    for year, items in by_year.items():
        items.sort(key=lambda sp: sp[0])  # alphabetical, deterministic
        for chunk_idx in range(_CHUNKS_PER_YEAR):
            pid = f"{year}_{chunk_idx}"
            if kind == "household":
                new_parts[pid] = LazyPartition(
                    _concat_household_for_year,
                    None,
                    items,
                    year,
                    chunk_idx,
                    _CHUNKS_PER_YEAR,
                )
            elif kind == "person":
                new_parts[pid] = LazyPartition(
                    _concat_person_relational_for_year,
                    None,
                    items,
                    year,
                    chunk_idx,
                    _CHUNKS_PER_YEAR,
                )
            elif kind == "keys":
                new_parts[pid] = LazyPartition(
                    _concat_keys_for_year,
                    None,
                    items,
                    chunk_idx,
                    _CHUNKS_PER_YEAR,
                )
            else:
                raise AssertionError(f"unknown kind {kind!r}")
    return LazyDataset(lf.merged_load, new_parts)


class AcsRelationalView(View):
    """Two-table relational view: household (one row per SERIALNO) plus the
    person table (multiple rows per SERIALNO) for hierarchical/relational
    synthesizers like Mare.

    Source `{state}_{year}` partitions are regrouped into
    ``_CHUNKS_PER_YEAR`` SERIALNO-hash chunks per year (pids
    `{year}_{chunk}`), mirroring [[AcsPersonView]]. Each chunk contains
    complete households (all persons of a SERIALNO land in the same chunk),
    and the household table's chunks line up with person via the shared
    hash. SERIALNO is state-prefixed on both sides so the join survives
    cross-state concatenation (pre-2018 PUMS SERIALNOs are only unique
    within a state-year)."""

    name = "acs"
    dataset = "acs"
    deps = {"household": ["household"], "person": ["person"]}
    trn_deps = {"person": ["household"]}
    parameters = get_relative_fn("parameters_relational.yml")

    def query(self, name, **tables: LazyFrame):
        match name:
            case "household":
                src = tables["household"]
                if not src.partitioned:
                    return _process_household(lambda: src(), "unknown", "0")
                regrouped = _group_relational_by_year(src, kind="household")
                return {pid: regrouped[pid] for pid in regrouped.keys()}
            case "person":
                src = tables["person"]
                if not src.partitioned:
                    return _add_state_year(lambda: src(), "unknown", "0")
                regrouped = _group_relational_by_year(src, kind="person")
                return {pid: regrouped[pid] for pid in regrouped.keys()}
            case other:
                raise AssertionError(f"Table {other!r} not part of view {self.name}")

    def split_keys(self, keys, req_splits, splits, random_state):
        return super().split_keys(
            _group_relational_by_year(keys, kind="keys"),
            req_splits,
            splits,
            random_state,
        )

    def filter_table(self, name, keys: LazyFrame, **tables: LazyFrame):
        return super().filter_table(
            name,
            _group_relational_by_year(keys, kind="keys"),
            **tables,
        )
