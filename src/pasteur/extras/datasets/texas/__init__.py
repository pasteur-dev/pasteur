import logging
from typing import TYPE_CHECKING, Callable

import pandas as pd

from ....dataset import Dataset, split_keys
from ....utils import gen_closure, get_relative_fn
from ....utils.progress import piter, process_in_parallel

logger = logging.getLogger(__name__)

LazyFrame = dict[str, Callable[..., pd.DataFrame]]


def _process_old(load: Callable):
    chunk = load()

    # Process new spec units
    spec_unit = chunk["spec_unit"].astype(str)
    get_spec_idx = (
        lambda i: spec_unit.str.slice(i, i + 1).replace("", pd.NA).astype("category")
    )
    spec_units = pd.DataFrame({f"spec_unit_{i + 1}": get_spec_idx(i) for i in range(5)})

    # Drop denormalized facility columns
    return chunk.drop(
        columns=[
            "fac_acute_care_ind",
            "fac_long_term_ac_ind",
            "fac_other_ltc_ind",
            "fac_peds_ind",
            "fac_psych_ind",
            "fac_rehab_ind",
            "fac_snf_ind",
            "fac_teaching_ind",
            "spec_unit",
        ]
    ).join(spec_units)


def _process_new(load1: Callable, load2: Callable):
    chunk1 = load1()
    chunk2 = load2()

    chunk1 = chunk1.drop(
        columns=[
            "filler_space",
            "apr_grouper_error_code",
            "apr_grouper_version_nbr",
            "ms_grouper_error_code",
            "ms_grouper_version_nbr",
        ]
    )

    # Drop the extra columns from the v1 base1
    # if "cert_status" in chunk1:
    #     chunk1.drop(
    #         columns=[
    #             "cert_status",
    #             "oth_icd9_code_1",
    #             "oth_icd9_code_10",
    #             "oth_icd9_code_11",
    #             "oth_icd9_code_12",
    #             "oth_icd9_code_13",
    #             "oth_icd9_code_14",
    #             "oth_icd9_code_15",
    #             "oth_icd9_code_16",
    #             "oth_icd9_code_17",
    #             "oth_icd9_code_18",
    #             "oth_icd9_code_19",
    #             "oth_icd9_code_2",
    #             "oth_icd9_code_20",
    #             "oth_icd9_code_21",
    #             "oth_icd9_code_22",
    #             "oth_icd9_code_23",
    #             "oth_icd9_code_24",
    #             "oth_icd9_code_3",
    #             "oth_icd9_code_4",
    #             "oth_icd9_code_5",
    #             "oth_icd9_code_6",
    #             "oth_icd9_code_7",
    #             "oth_icd9_code_8",
    #             "oth_icd9_code_9",
    #             "poa_provider_indicator",
    #             "princ_diag_code",
    #         ]
    #     )

    return chunk1.join(chunk2.drop(columns=["filler_space"]))


def ingest_worker(pid: str, fun: Callable[..., pd.DataFrame]):
    return (pid, fun())


def _ingest_base(
    base: LazyFrame, base1: LazyFrame, base1_v2: LazyFrame, base2: LazyFrame
) -> LazyFrame:
    old_pids = list(base.keys())
    new_pids1 = list(base1.keys())
    new_pids2 = list(base1_v2.keys())
    assert set(base2.keys()) == set(new_pids1).union(
        new_pids2
    ), "There's a missmatch of base2 and base1 files."

    # jobs = [
    #     *[
    #         {"pid": pid, "fun": gen_closure(_process_old, base[pid])}
    #         for pid in old_pids
    #     ],
    #     *[
    #         {"pid": pid, "fun": gen_closure(_process_new, base1[pid], base2[pid])}
    #         for pid in new_pids1
    #     ],
    #     *[
    #         {"pid": pid, "fun": gen_closure(_process_new, base1_v2[pid], base2[pid])}
    #         for pid in new_pids2
    #     ],
    # ]

    # return {
    #     pid: data
    #     for pid, data in process_in_parallel(
    #         ingest_worker, jobs, {}, 1, "Ingesting chunks"
    #     )
    # }

    return {
        **{pid: gen_closure(_process_old, base[pid]) for pid in old_pids},
        **{pid: gen_closure(_process_new, base1[pid], base2[pid]) for pid in new_pids1},
        **{
            pid: gen_closure(_process_new, base1_v2[pid], base2[pid])
            for pid in new_pids2
        },
    }



class TexasDataset(Dataset):
    name = "texas"
    deps = {
        "base": ["base", "base1", "base1_v2", "base2"],
        "facility": ["facility", "base"],
        "charges": ["charges"],
    }
    key_deps = ["base", "base2"]

    folder_name = "texas"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        import os
        from zipfile import ZipFile

        os.makedirs(bootstrap, exist_ok=True)
        for fn in piter(os.listdir(location), leave=False):
            if not (fn.endswith("_tab") or fn.endswith("-tab-delimited")):
                continue

            with ZipFile(os.path.join(location, fn), "r") as zf:
                logger.info(f"Extracting {fn}...")
                zf.extractall(bootstrap)

    def ingest(self, name: str, **tables: LazyFrame):
        if name == "base":
            return _ingest_base(**tables)
        return pd.DataFrame()

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables[""], ratios, random_state)
