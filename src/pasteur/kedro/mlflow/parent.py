import json
import logging
import os
import pickle
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT

from ...utils.mlflow import ARTIFACT_DIR, mlflow_log_perf, mlflow_log_energy
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
                elif fn.endswith(".csv"):
                    import pandas as pd

                    art = pd.read_csv(f)
                elif fn.endswith(".pkl"):
                    try:
                        art = pickle.load(f)
                    except Exception as e:
                        logger.error(
                            f"Error loading pickle artifact.\n'{fn}'", exc_info=True
                        )
                        continue
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
        if all(k in run and run[k] == v for run in run_params.values())
        and not k.startswith("_")
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
                s = param_str if run_params[name][param] else f"no_{param_str}"
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


def _render_params_plot(
    params_by_split: dict[str, int | float],
    artifact_path: str,
):
    """Plot total parameter count per algorithm across sweep steps.

    Mirrors the layout of ``_render_multiplot`` in extras/metrics/distr.py
    (algorithms as lines, x-axis = sweep steps, dots/CI when ``-r N`` runs
    exist) but with a single subplot and saves as a standalone SVG.
    """
    import re
    from io import BytesIO

    import matplotlib.pyplot as plt
    import numpy as np

    from ...utils.styles import use_style

    if not params_by_split:
        return

    parsed: dict[str, tuple[str, str | None, int | None]] = {}
    for name in params_by_split:
        tokens = name.split()
        if not tokens:
            parsed[name] = ("default", None, None)
            continue
        if "=" not in tokens[0] and not re.match(r"^r\d+$", tokens[0]):
            alg = tokens[0]
            rest = tokens[1:]
        else:
            alg = "syn"
            rest = tokens
        run_idx = None
        step_parts = []
        for token in rest:
            if re.match(r"^r\d+$", token):
                run_idx = int(token[1:])
            else:
                step_parts.append(token)
        step = " ".join(step_parts) if step_parts else None
        parsed[name] = (alg, step, run_idx)

    algorithms = list(dict.fromkeys(alg for alg, _, _ in parsed.values()))
    steps = list(dict.fromkeys(step for _, step, _ in parsed.values()))
    has_runs = any(r is not None for _, _, r in parsed.values())
    has_steps = len(steps) > 1 or (len(steps) == 1 and steps[0] is not None)
    if not has_steps:
        steps = [None]

    use_style("mlflow")

    fig_w = max(6, 3 + 1.2 * len(steps))
    fig_h = 4.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    cmap = plt.cm.tab10.colors  # type: ignore[attr-defined]
    n_alg = len(algorithms)

    for alg_idx, alg in enumerate(algorithms):
        color = cmap[alg_idx % len(cmap)]
        pos_list: list[int] = []
        mean_list: list[float] = []
        vals_list: list[list[float]] = []

        for step_idx, step in enumerate(steps):
            run_values: list[float] = []
            for name, (a, s, _r) in parsed.items():
                if a == alg and s == step:
                    val = params_by_split.get(name)
                    if val is None:
                        continue
                    fval = float(val)
                    if np.isnan(fval):
                        continue
                    run_values.append(fval)
            if not run_values:
                continue
            pos_list.append(step_idx)
            mean_list.append(float(np.mean(run_values)))
            vals_list.append(run_values)

        if not pos_list:
            continue

        ax.plot(
            pos_list,
            mean_list,
            color=color,
            label=alg,
            marker="o",
            markersize=5,
            linewidth=1.5,
            zorder=4,
        )

        if has_runs:
            offset = (alg_idx - (n_alg - 1) / 2) * 0.06
            for pos, vals in zip(pos_list, vals_list):
                if len(vals) > 1:
                    q25, q75 = np.percentile(vals, [25, 75])
                    vmin, vmax = float(np.min(vals)), float(np.max(vals))
                    ax.plot(
                        [pos + offset, pos + offset],
                        [vmin, vmax],
                        color=color,
                        linewidth=1,
                        alpha=0.4,
                        zorder=2,
                    )
                    ax.plot(
                        [pos + offset, pos + offset],
                        [q25, q75],
                        color=color,
                        linewidth=4,
                        alpha=0.3,
                        zorder=2,
                    )
                jitter = np.linspace(-0.03, 0.03, len(vals)) + offset
                ax.scatter(
                    [pos + j for j in jitter],
                    vals,
                    color=color,
                    alpha=0.5,
                    s=15,
                    zorder=3,
                )

    ax.set_xticks(range(len(steps)))
    step_labels = [s if s is not None else "default" for s in steps]
    if any(len(str(lb)) > 10 for lb in step_labels) or len(step_labels) > 6:
        ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(step_labels, fontsize=8)

    ax.set_title("Model Parameter Count", fontweight="bold", fontsize=11)
    ax.set_ylabel("Parameters", fontsize=9)
    if any(v > 0 for v in params_by_split.values()):
        ax.set_yscale("log")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, linewidth=0.5, which="both")

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    svg = buf.read().decode("utf-8")
    plt.close(fig)
    mlflow.log_text(svg, artifact_path)


def log_parent_run(
    parent: str,
    run_params: dict[str, dict[str, Any]],
    skip_parent: bool = False,
    experiment_id: str | None = None,
):
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT.set(True)

    git = get_git_suffix()
    query = f'tags.pasteur_id = "{sanitize_name(parent)}" and tags.pasteur_parent = "1" and tags.pasteur_git = "{git}"'
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
            name: run
            for name in run_params
            if (
                run := get_run(
                    name,
                    parent if not skip_parent else None,
                    git if not skip_parent else None,
                )
            )
            is not None
        }
        # Filter out runs that weren't found in MLflow
        missing = [name for name, run in runs.items() if run is None]
        if missing:
            logger.warning(f"MLflow runs not found (skipping): {missing}")
        runs = {name: run for name, run in runs.items() if run is not None}
        if not runs:
            logger.error("No MLflow runs found, skipping parent logging")
            return
        artifacts = get_artifacts(runs)
        pretty = prettify_run_names(run_params)

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

        # Log time metrics in nodes
        try:
            perfs = {pretty[n]: a["perf"] for n, a in artifacts.items() if "perf" in a}
            mlflow_log_perf(**perfs)
        except Exception:
            logger.error(f"Error logging performance.", exc_info=True)

        # Log energy
        try:
            energy = {
                pretty[n]: a["energy"] for n, a in artifacts.items() if "energy" in a
            }
            if energy:
                mlflow_log_energy(**energy)
        except Exception:
            logger.error(f"Error logging energy info.", exc_info=True)

        # Copy top-level files from each run's artifact dir into model/<pretty>/
        for name, run in runs.items():
            try:
                from tempfile import TemporaryDirectory

                with TemporaryDirectory() as run_artifact_dir:
                    mlflow.artifacts.download_artifacts(
                        run_id=run.info.run_id, artifact_path="./", dst_path=run_artifact_dir
                    )
                    for entry in os.listdir(run_artifact_dir):
                        full_path = os.path.join(run_artifact_dir, entry)
                        if os.path.isfile(full_path):
                            mlflow.log_artifact(
                                full_path, artifact_path=f"model/{pretty[name]}"
                            )
            except Exception:
                logger.error(
                    f"Error logging model files for '{name}'.", exc_info=True
                )

        # Log model parameter count plot
        try:
            params_by_split: dict[str, int | float] = {}
            for n, a in artifacts.items():
                tp = (
                    a.get("model", {}).get("total_params")
                    if isinstance(a, dict)
                    else None
                )
                if tp is not None:
                    params_by_split[pretty[n]] = tp
            if params_by_split:
                _render_params_plot(params_by_split, "params.svg")
        except Exception:
            logger.error(f"Error rendering params plot.", exc_info=True)

        for name, folder in ref_artifacts["metrics"].items():
            try:
                # For some reason testing if "metric" is in folder can return
                # true but then folder["metric"] raises an error
                metric = folder["metric"]
            except Exception:
                logger.error(
                    f"Metric '{name}' does not have a 'metric' executable, skipping..."
                )
                continue

            splits = {}
            for alg_name, artifact in artifacts.items():
                try:
                    splits[pretty[alg_name]] = artifact["metrics"][name]["data"]
                except Exception as e:
                    logger.error(
                        f"Split '{pretty[alg_name]}' metric '{name}' is broken."
                    )

            try:
                metric.visualise(data=splits)
            except Exception as e:
                logger.error(f"Error visualising metric '{name}'.", exc_info=True)
            try:
                metric.summarize(data=splits)
            except Exception as e:
                logger.error(f"Error summarizing metric '{name}'.", exc_info=True)
