import logging
from typing import Any, Iterable

import click
from kedro.framework.session import KedroSession

from ..utils.parser import eval_params, merge_params, str_params_to_dict
from .runner import SimpleRunner

logger = logging.getLogger(__name__)


@click.command()
@click.argument("pipeline", type=str, default=None)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
def pipe(pipeline, params):
    """pipe(line) is a modified version of run with minified logging and shorter syntax"""

    param_dict = str_params_to_dict(params)

    with KedroSession.create(extra_params=param_dict) as session:
        session.run(
            tags=[],
            runner=SimpleRunner(pipeline, " ".join(params)),  # SequentialRunner(True),
            node_names="",
            from_nodes="",
            to_nodes="",
            from_inputs="",
            to_outputs="",
            load_versions={},
            pipeline_name=pipeline,
        )


def _process_iterables(iterables: dict[str, Iterable]):
    sentinel = object()
    iterator_dict = {n: iter(v) for n, v in iterables.items()}
    value_dict = {n: next(v, None) for n, v in iterator_dict.items()}

    has_combs = True
    while has_combs:
        yield value_dict

        has_combs = False
        for name, it in iterator_dict.items():
            val: Any = next(it, sentinel)

            if val is sentinel:
                new_it = iter(iterables[name])
                iterator_dict[name] = new_it
                value_dict[name] = next(new_it, None)
            else:
                value_dict[name] = val
                has_combs = True
                break


@click.command()
@click.argument("pipeline", type=str, default=None)
@click.option("--alg", "-a", multiple=True)
@click.option("--iterator", "-i", multiple=True)
@click.option("--hyperparameter", "-h", multiple=True)
@click.option("--clear-cache", "-c", is_flag=True)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
def sweep(pipeline, alg, iterator, hyperparameter, params, clear_cache):
    """Similar to pipe, sweep allows in addition a hyperparameter sweep.

    By using `-i` an iterator can be defined (ex. `-i i="range(5)"`), which will
    make the pipeline run for each value of i. Then i can be used in expressions
    with other variables that are passed as arguments (ex. `j="0.2*i"`).

    If an iterator is also a hyperparameter (ex. `-h e1="[0.1,0.2,0.3]"`)
    then `-h` can be used, which will both sweep and pass the variable as an
    override at the same time (it is equal to `-i val=<iterable> val=val`).

    If `alg` is provided, `pipeline` is treated as the sweep view and the algorithms
    provided are sweeped. Example:
    ```
    s tab_adult -a privbayes -a aim
    ```
    runs the following for each parameter combination:
    ```
    tab_adult.ingest
    tab_adult.aim.synth
    tab_adult.privbayes.synth
    ```

    Ingest is ran for each parameter combination, so if a parameter override
    changes the view it is honored."""

    from .mlflow import (
        check_run_done,
        get_parent_name,
        get_run_name,
        log_parent_run,
        remove_runs,
    )

    parent_name = get_parent_name(pipeline, alg, hyperparameter, iterator, params)
    if clear_cache:
        with KedroSession.create() as session:
            session.load_context()
            logger.warning(f"Removing runs from mlflow with parent:\n{parent_name}")
            remove_runs(parent_name)

    iterable_dict = eval_params(iterator)
    hyperparam_dict = eval_params(hyperparameter)

    mlflow_dict = {
        "_mlflow_parent_name": parent_name,
    }

    runs = {}

    if alg:
        view = pipeline
        pipelines = {f"{view}.ingest": None, **{f"{view}.{a}.synth": a for a in alg}}
    else:
        pipelines = {pipeline: ""}

    for iters in _process_iterables(iterable_dict | hyperparam_dict):
        param_dict = eval_params(params, iters)
        hyper_dict = {n: iters[n] for n in hyperparam_dict}
        vals = param_dict | hyper_dict
        extra_params = merge_params(vals | mlflow_dict)

        for pipeline, alg in pipelines.items():
            with KedroSession.create(extra_params=extra_params) as session:
                session.load_context()

                run_name = get_run_name(pipeline, vals)
                if alg:
                    # if alg exists add its name
                    runs[run_name] = {"_alg": alg, **vals}
                elif alg is not None:
                    # ingest pipeline has None and should be skipped from cross-eval
                    runs[run_name] = vals

                if check_run_done(run_name, parent_name):
                    logger.warning(f"Run '{run_name}' is complete, skipping...")
                    continue

                session.run(
                    tags=[],
                    runner=SimpleRunner(
                        pipeline, " ".join(f"{n}={v}" for n, v in vals.items())
                    ),
                    node_names="",
                    from_nodes="",
                    to_nodes="",
                    from_inputs="",
                    to_outputs="",
                    load_versions={},
                    pipeline_name=pipeline,
                )

    with KedroSession.create() as session:
        session.load_context()
        log_parent_run(parent_name, runs)


@click.command()
@click.option("--user", "-u", type=str, default=None)
@click.option(
    "--download-dir",
    "-d",
    type=str,
    default=None,
    help="Specify a different download dir. By default `raw_location` is used.",
)
@click.argument(
    "datasets",
    nargs=-1,
    type=str,
)
@click.option(
    "--accept",
    "-a",
    is_flag=True,
    help="By passing this option, you accept to the terms of the data that will be downloaded. Pasteur doesn't provide or license any of the datasets.",
)
def download(
    user: str | None,
    download_dir: str | None,
    datasets: list[str],
    accept: bool = False,
):
    """Downloads all Pasteur datasets from their creators, provided the user
    agrees to their access requirements, and has credentials, if required.

    Uses `wget` and `boto3` to download files.

    Only downloads missing files, can be ran to verify dataset is downloaded correctly."""
    from ..extras.download import get_description, main

    # Setup logging and params with kedro
    with KedroSession.create() as session:
        ctx = session.load_context()

        logger.info(get_description())
        if not datasets:
            return

        download_dir = download_dir or ctx.params.get("raw_location", None)
        assert download_dir, f"Download dir is empty"

        if not accept:
            logger.error(
                "You have to accept to the license of the data stores you're about to download from (--accept/-a)."
            )
        else:
            main(download_dir, datasets, user)


@click.command()
@click.argument(
    "datasets",
    nargs=-1,
    type=str,
)
def bootstrap(
    datasets: tuple[str],
):
    """Preprocesses downloaded datasets which require it so they can be loaded by kedro."""
    from os import path

    from ..dataset import Dataset
    from ..module import get_module_dict
    from ..kedro.pipelines.main import NAME_LOCATION
    from ..utils.progress import logging_redirect_pbar

    # Setup logging and params with kedro
    with KedroSession.create() as session:
        ctx = session.load_context()

        dataset_modules = get_module_dict(Dataset, ctx.pasteur.modules)  # type: ignore

        if not datasets:
            datasets = tuple(dataset_modules)

        for dataset in datasets:
            if dataset not in dataset_modules:
                logger.error(f"Module for dataset `{dataset}` not currently loaded.")
                return

        params = ctx.params
        raw_location = params["raw_location"]
        base_location = params["base_location"]

        with logging_redirect_pbar():
            for name in datasets:
                ds = dataset_modules[name]
                if not ds.bootstrap:
                    continue
                assert (
                    ds.folder_name
                ), "Folder name for a dataset shouldn't be null when bootstrap is supplied."

                ds_raw_location = params.get(NAME_LOCATION.format(ds.folder_name), path.join(raw_location, ds.folder_name))
                bootstrap_location = path.join(base_location, "bootstrap", ds.folder_name)  # type: ignore
                logger.info(f"Initializing dataset \"{name}\" in:\n{bootstrap_location}")
                ds.bootstrap(ds_raw_location, bootstrap_location)


@click.group(name="Pasteur")
def cli():
    """Command line tools for manipulating a Kedro project."""


cli.add_command(bootstrap)
cli.add_command(download)
cli.add_command(pipe)
cli.add_command(sweep)

cli.add_command(download, "dl")
cli.add_command(pipe, "p")
cli.add_command(sweep, "s")
