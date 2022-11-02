import mlflow
from mlflow.entities import RunStatus

from ...utils import dict_to_flat_params


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


def _get_git_suffix():
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return f" (git:{sha[:8]})"
    except:
        return ""


def get_run_name(pipeline: str, params: dict[str]):
    run_name = pipeline
    for param, val in dict_to_flat_params(params).items():
        if param.startswith("_"):
            continue
        run_name += f" {param}={val}"

    return run_name + _get_git_suffix()


def get_parent_name(
    pipeline: str, hyperparams: list[str], iterators: list[str], params: list[str]
):
    hyper_str = "".join(map(lambda x: f" -h {x}", hyperparams))
    iter_str = "".join(map(lambda x: f" -i {x}", iterators))
    param_str = "".join(map(lambda x: f" {x}", params))
    return f"{pipeline}{hyper_str}{iter_str}{param_str}{_get_git_suffix()}"


def sanitize_name(name: str):
    # todo: properly escape
    return name.replace('"', '\\"').replace("'", "\\'")


def check_run_done(name: str, parent: str):
    return (
        len(
            mlflow.search_runs(
                search_all_experiments=True,
                filter_string=f"tags.pasteur_id = '{sanitize_name(name)}' and "
                + f"tags.pasteur_pid = '{sanitize_name(parent)}' and "
                + f"attribute.status = '{RunStatus.to_string(RunStatus.FINISHED)}'",
            )
        )
        > 0
    )


def remove_runs(parent: str):
    """Removes runs with provided parent"""

    # Delete children
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_pid = '{sanitize_name(parent)}'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)

    # Delete parent
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.pasteur_id = '{sanitize_name(parent)}' and tags.pasteur_parent = '1'",
    )
    for id in runs["run_id"]:
        mlflow.delete_run(id)
