import json
import logging
import os
import pickle
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT

from ...utils.mlflow import ARTIFACT_DIR, mlflow_log_perf
from .base import get_git_suffix, get_run, sanitize_name

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
                else:
                    continue

            try:
                no_ext = name[: name.rindex(".")]
            except Exception:
                no_ext = name
            sub_dict[no_ext] = art

    return artifacts


def get_artifacts(runs: dict[str, Run]):
    return {name: get_run_artifacts(run) for name, run in runs.items()}


def prettify_run_names(run_params: dict[str, dict[str, Any]]):
    """Generates a run name based on parameters that are short for use in graphs.

    Parameters of each run are lined up with each other and left-justified.
    The resulting name is stripped to the right, to remove extra space at the end
    if possible. Left spaces remain to maintain structure if the final name is
    left-justified.

    Parameters that start with `_`, get priority and only have their value printed.
    Ex. `{"_alg": "privbayes", "e1": "abc"}` becomes `privbayes e_1: abc`.

    Parameters composed of letters and then numbers have their number become an indicator:
    `e1` becomes `e_1`, where `_` indicates subscript. TODO

    Parameters with boolean are only printed when true."""

    ref_run = next(iter(run_params.values()))
    value_params = {k for k in ref_run if k.startswith("_")}
    bool_params = {k for k, v in ref_run.items() if isinstance(v, bool)}

    # Skip params shared by all runs
    skip_params = {
        k
        for k, v in ref_run.items()
        if all(k in run and run[k] == v for run in run_params.values()) and not k.startswith("_")
    }

    str_params = {name: [] for name in run_params}
    pretty_provided = {}
    for param in ref_run:
        if param in skip_params:
            continue
        # Calculate str length for str_params
        # length = max(
        #     map(lambda x: len(str(x)), [run[param] for run in run_params.values()])
        # )

        for name in run_params:
            try:
                param_str = param[param.rindex(".") + 1 :]
            except Exception:
                param_str = param

            if param in bool_params:
                # s = param_str if run_params[name][param] else (" " * len(param_str))
                s = param_str if run_params[name][param] else ""
            elif param == "_alg":
                # FIXME: dirty hack to add algorithm name
                s = str(name.split(".", 1)[-1].split(" ", 1)[0])
            elif param == "_pretty":
                pretty_provided[name] = str(run_params[name][param])
                continue
            else:
                val_str = str(run_params[name][param])
                # buffer = " " * (length - len(val_str))
                buffer = ""

                if param in value_params:
                    s = f"{val_str}{buffer}"
                else:
                    s = f"{param_str}={val_str}{buffer}"

            str_params[name].append(s)
    return {
        name: " ".join(params).strip() if params and any(params) else "base"
        for name, params in str_params.items()
    } | pretty_provided


def log_parent_run(
    parent: str,
    run_params: dict[str, dict[str, Any]],
    skip_parent: bool = False,
    experiment_id: str | None = None,
):
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT.set(True)

    git = get_git_suffix()
    query = f"tags.pasteur_id = '{sanitize_name(parent)}' and tags.pasteur_parent = '1' and tags.pasteur_git = '{git}'"
    parent_runs = mlflow.search_runs(filter_string=query, search_all_experiments=True)

    if not len(parent_runs):
        logger.info(f"Creating empty mlflow parent run:\n{parent}")
        ctx_mgr = mlflow.start_run(run_name=parent, experiment_id=experiment_id)
        mlflow.set_tag("pasteur_id", parent)
        mlflow.set_tag("pasteur_parent", "1")
        mlflow.set_tag("pasteur_git", git)
    else:
        parent_run_id = parent_runs["run_id"][0]  # type: ignore
        logger.info(f"Relaunching parent run for logging:\n{parent}")
        ctx_mgr = mlflow.start_run(parent_run_id)

    with ctx_mgr:
        runs = {
            name: get_run(
                name,
                parent if not skip_parent else None,
                git if not skip_parent else None,
            )
            for name in run_params
        }
        artifacts = get_artifacts(runs)
        pretty = prettify_run_names(run_params)
        assert len(runs)

        ref_params = next(iter(runs.values())).data.params

        for name, val in ref_params.items():
            for run in runs.values():
                params = run.data.params
                if not name in params or params[name] != val:
                    break
            else:
                # if we iterate over the whole loop else runs
                # log param if it exists and its the same in all runs
                mlflow.log_param(name, val)

        ref_artifacts = next(iter(artifacts.values()))
        # meta = ref_artifacts["meta"]

        perfs = {pretty[n]: a["perf"] for n, a in artifacts.items() if "perf" in a}
        try:
            mlflow_log_perf(**perfs)
        except Exception as e:
            logger.error(f"Error logging performance:\n{e}")

        for name, folder in ref_artifacts["metrics"].items():
            if not "metric" in folder:
                logger.error(
                    f"Metric '{name}' does not have a 'metric' executable, skipping..."
                )
            metric = folder["metric"]

            splits = {}
            for alg_name, artifact in artifacts.items():
                try:
                    splits[pretty[alg_name]] = artifact["metrics"][name]["data"]
                except Exception as e:
                    logger.error(
                        f"Split '{pretty[alg_name]}' metric '{name}' is broken."
                    )

            metric.visualise(data=splits)
            metric.summarize(data=splits)
