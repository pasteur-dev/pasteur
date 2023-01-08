import logging
from typing import Callable

import pandas as pd

from ....dataset import Dataset
from ....utils import LazyChunk, LazyFrame, gen_closure, get_relative_fn, to_chunked
from ....utils.progress import process_in_parallel

logger = logging.getLogger(__name__)


def _process_old(load: Callable):
    chunk = load()

    # Process new spec units
    spec_unit = chunk["spec_unit"].astype(str)
    get_spec_idx = (
        lambda i: spec_unit.str.slice(i, i + 1).replace("", pd.NA).astype("category")
    )
    spec_units = pd.DataFrame(
        {f"spec_unit_{i + 1}": get_spec_idx(i) for i in range(5)}
    )  # type:ignore

    empty_col = pd.Series(pd.NA, index=chunk.index).astype("category")
    empty_col_names = [
        "poa_e_code_1",
        "poa_e_code_10",
        "poa_e_code_2",
        "poa_e_code_3",
        "poa_e_code_4",
        "poa_e_code_5",
        "poa_e_code_6",
        "poa_e_code_7",
        "poa_e_code_8",
        "poa_e_code_9",
        "poa_oth_diag_code_1",
        "poa_oth_diag_code_10",
        "poa_oth_diag_code_11",
        "poa_oth_diag_code_12",
        "poa_oth_diag_code_13",
        "poa_oth_diag_code_14",
        "poa_oth_diag_code_15",
        "poa_oth_diag_code_16",
        "poa_oth_diag_code_17",
        "poa_oth_diag_code_18",
        "poa_oth_diag_code_19",
        "poa_oth_diag_code_2",
        "poa_oth_diag_code_20",
        "poa_oth_diag_code_21",
        "poa_oth_diag_code_22",
        "poa_oth_diag_code_23",
        "poa_oth_diag_code_24",
        "poa_oth_diag_code_3",
        "poa_oth_diag_code_4",
        "poa_oth_diag_code_5",
        "poa_oth_diag_code_6",
        "poa_oth_diag_code_7",
        "poa_oth_diag_code_8",
        "poa_oth_diag_code_9",
        "poa_princ_diag_code",
        "poa_provider_indicator",
    ]
    empty_df = pd.DataFrame({name: empty_col for name in empty_col_names})

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
    ).join(spec_units).join(empty_df)


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

    if "princ_icd9_code" not in chunk1:
        empty_col = pd.Series(pd.NA, index=chunk1.index).astype("category")
        empty_col_names = [
            "cert_status",
            "oth_icd9_code_1",
            "oth_icd9_code_10",
            "oth_icd9_code_11",
            "oth_icd9_code_12",
            "oth_icd9_code_13",
            "oth_icd9_code_14",
            "oth_icd9_code_15",
            "oth_icd9_code_16",
            "oth_icd9_code_17",
            "oth_icd9_code_18",
            "oth_icd9_code_19",
            "oth_icd9_code_2",
            "oth_icd9_code_20",
            "oth_icd9_code_21",
            "oth_icd9_code_22",
            "oth_icd9_code_23",
            "oth_icd9_code_24",
            "oth_icd9_code_3",
            "oth_icd9_code_4",
            "oth_icd9_code_5",
            "oth_icd9_code_6",
            "oth_icd9_code_7",
            "oth_icd9_code_8",
            "oth_icd9_code_9",
            "poa_provider_indicator",
            "princ_icd9_code",
        ]
        empty_df = pd.DataFrame({name: empty_col for name in empty_col_names})
        chunk1 = chunk1.join(empty_df)

    return chunk1.join(chunk2.drop(columns=["filler_space"]))


def _ingest_base(
    base: LazyFrame, base1: LazyFrame, base1_v2: LazyFrame, base2: LazyFrame
):
    old_pids = list(base.keys())
    new_pids1 = list(base1.keys())
    new_pids2 = list(base1_v2.keys())
    assert set(base2.keys()) == set(new_pids1).union(
        new_pids2
    ), "There's a missmatch of base2 and base1 files."

    assert base.partitioned
    assert base1.partitioned
    assert base1_v2.partitioned
    assert base2.partitioned

    return {
        **{pid: gen_closure(_process_old, base[pid]) for pid in old_pids},
        **{pid: gen_closure(_process_new, base1[pid], base2[pid]) for pid in new_pids1},
        **{
            pid: gen_closure(_process_new, base1_v2[pid], base2[pid])
            for pid in new_pids2
        },
    }


def _ingest_facility(facility: LazyFrame):
    assert facility.partitioned
    return pd.concat([fac() for fac in facility.values()]).drop_duplicates()


def _add_id(id: str, fun: Callable[..., pd.DataFrame] | pd.DataFrame):
    chunk = fun() if callable(fun) else fun
    chunk.index.name = id
    return chunk

def _extract(loc: str, fn: str, dst: str):
    from zipfile import ZipFile
    import os

    if not (fn.endswith("_tab") or fn.endswith("-tab-delimited")):
        return

    with ZipFile(os.path.join(loc, fn), "r") as zf:
        logger.info(f"Extracting {fn}...")
        zf.extractall(dst)

class TexasDataset(Dataset):
    name = "texas"
    deps = {
        "base": ["base", "base1", "base1_v2", "base2"],
        "facility": ["facility", "base"],
        "charges": ["charges"],
    }
    key_deps = ["base"]

    folder_name = "texas"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        import os

        os.makedirs(bootstrap, exist_ok=True)

        base_args = {'loc': location, 'dst': bootstrap}
        per_call_args = [{"fn": fn} for fn in os.listdir(location)]
        process_in_parallel(_extract, per_call_args, base_args, desc='Unzipping texas files')


    def ingest(self, name: str, **tables: LazyFrame):
        match (name):
            case "base":
                return _ingest_base(**tables)
            case "charges":
                return {
                    pid: gen_closure(_add_id, "charge_id", fun)
                    for pid, fun in tables["charges"].items()
                }
            case "facility":
                return _ingest_facility(tables["facility"])
            case _:
                raise Exception()

    @to_chunked
    def keys(self, base: LazyChunk):
        return base()[[]]
