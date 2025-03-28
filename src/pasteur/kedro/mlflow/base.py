from typing import Any

import mlflow
from mlflow.entities import Run, RunStatus

from ...utils.parser import dict_to_flat_params


def flatten_dict(d: dict, recursive: bool = True, sep: str = ".") -> dict:
    def expand(key, value):
        if isinstance(value, dict):
            new_value = (
                flatten_dict(value, recursive=recursive, sep=sep)
                if recursive
                else value
            )
            return [(f"{key}{sep}{k}", v) for k, v in new_value.items()]
        else:
            return [(f"{key}", value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


_git_id = None


def _get_git_suffix():
    # FIXME: Dirty global var

    global _git_id
    if _git_id is not None:
        return _git_id
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        _git_id = f" (git:{sha[:8]})"
        return _git_id
    except Exception:
        return ""


def get_run_name(pipeline: str, params: dict[str, Any]):
    run_name = pipeline
    for param, val in dict_to_flat_params(params).items():
        if param.startswith("_"):
            continue
        run_name += f" {param}={val}"

    return run_name + _get_git_suffix()


def get_parent_name(
    pipeline: str,
    algs: list[str],
    hyperparams: list[str],
    iterators: list[str],
    params: list[str],
):
    algs_str = ""
    if algs:
        algs_str = " -a [" + ", ".join(algs) + "]"
    hyper_str = "".join(map(lambda x: f" -h {x}", hyperparams))
    iter_str = "".join(map(lambda x: f" -i {x}", iterators))
    param_str = "".join(map(lambda x: f" {x}", params))
    return f"{pipeline}{algs_str}{hyper_str}{iter_str}{param_str}{_get_git_suffix()}"


def sanitize_name(name: str):
    # todo: properly escape
    return name.replace('"', '\\"').replace("'", "\\'")


def get_run_id(name: str, parent: str | None, finished: bool = True):
    filter_string = f"tags.pasteur_id = '{sanitize_name(name)}'"
    if parent:
        filter_string += f" and tags.pasteur_pid = '{sanitize_name(parent)}'"
    if finished:
        filter_string += (
            f" and attribute.status = '{RunStatus.to_string(RunStatus.FINISHED)}'"
        )
    tmp = mlflow.search_runs(
        experiment_ids=[exp.experiment_id for exp in mlflow.search_experiments()],
        filter_string=filter_string,
    )
    if len(tmp):
        return tmp["run_id"][0]
    return None


def check_run_done(name: str, parent: str | None):
    return bool(get_run_id(name, parent))


def get_run(name: str, parent: str | None) -> Run:
    return mlflow.get_run(get_run_id(name, parent))


def remove_runs(parent: str, delete_parent: bool = False):
    """Removes runs with provided parent"""

    # Delete children
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_pid = '{sanitize_name(parent)}'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)

    # Delete parent
    if not delete_parent:
        return

    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_id = '{sanitize_name(parent)}' and tags.pasteur_parent = '1'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)
