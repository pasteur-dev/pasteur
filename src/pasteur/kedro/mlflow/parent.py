import json
import logging
import os
import pickle
from typing import Any

import mlflow
from mlflow.entities import Run

from ...metrics.mlflow import ARTIFACT_DIR
from .base import get_run, sanitize_name

logger = logging.getLogger(__name__)

def get_run_artifacts(run: Run):
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=ARTIFACT_DIR
    )

    artifacts = {}
    # Load all artifacts by walking
    for root, _, files in os.walk(artifact_dir):
        if not files:
            continue

        # if dir is <p1>/<p2>/<p3> place artifacts in {p1: {p2: {p3: artifacts}}}
        sub_dict = artifacts
        for sub in root.replace(artifact_dir, "").split("/"):
            if not sub:
                continue
            tmp = sub_dict.get(sub, {})
            sub_dict[sub] = tmp
            sub_dict = tmp

        # load all files, support pickle and json for now
        for name in files:
            fn = os.path.join(root, name)

            with open(fn, "rb") as f:
                if fn.endswith(".json"):
                    art = json.load(f)
                elif fn.endswith(".pkl"):
                    art = pickle.load(f)

            no_ext = name[: name.rindex(".")]
            sub_dict[no_ext] = art

    return artifacts


def get_artifacts(runs: dict[str, Run]):
    return {name: get_run_artifacts(run) for name, run in runs.items()}


def prettify_run_names(run_params: dict[str, dict[str, Any]]):
    """Generates a run name based on parameters that are short for use in graphs.

    All names have the same length.

    Parameters that start with `_`, get priority and only have their value printed.
    Ex. `{"_alg": "privbayes", "e1": "abc"}` becomes `privbayes e_1: abc`.

    Parameters composed of letters and then numbers have their number become an indicator:
    `e1` becomes `e_1`, where `_` indicates subscript. TODO

    Parameters with boolean are only printed when true."""

    ref_run = next(iter(run_params.values()))
    value_params = {k for k in ref_run if k.startswith("_")}
    bool_params = {k for k, v in ref_run.items() if isinstance(v, bool)}

    str_params = {name: [] for name in run_params}
    for param in ref_run:
        # Calculate str length for str_params
        length = max(
            map(lambda x: len(str(x)), [run[param] for run in run_params.values()])
        )

        for name in run_params:
            if param in bool_params:
                s = param if run_params[name][param] else (" " * len(param))
            else:
                val_str = str(run_params[name][param])
                buffer = " " * (length - len(val_str))

                if param in value_params:
                    s = f"{buffer}{val_str}"
                else:
                    s = f"{param}={buffer}{val_str}"

            str_params[name].append(s)
    return {name: " ".join(params) for name, params in str_params.items()}


def log_parent_run(parent: str, run_params: dict[str, dict[str, Any]]):
    from ...metrics.visual import mlflow_log_hists

    query = f"tags.pasteur_id = '{sanitize_name(parent)}' and tags.pasteur_parent = '1'"
    parent_runs = mlflow.search_runs(
        filter_string=query,
        search_all_experiments=True
    )

    if len(parent_runs):
        parent_run_id = parent_runs["run_id"][0]
        logger.info(f"Relaunching parent run for logging:\n{parent}")
        mlflow.start_run(
            parent_run_id,
        )
    else:
        # TODO: Perhaps this should not be true
        assert False, f"Parent run {parent} should exist to create combined report."

    runs = {name: get_run(name, parent) for name in run_params}
    artifacts = get_artifacts(runs)
    pretty = prettify_run_names(run_params)

    assert len(artifacts)
    ref_artifacts = next(iter(artifacts.values()))
    visual = ref_artifacts["visual"]
    meta = ref_artifacts["meta"]
    
    holder = visual["holder"]
    data = {
        "wrk": visual["wrk"],
        "ref": visual["ref"]
    }
    syn_data = {pretty[name]: art["visual"]["syn"] for name, art in artifacts.items()}
    
    mlflow_log_hists(holder, log_artifacts=False, **data, **syn_data)
