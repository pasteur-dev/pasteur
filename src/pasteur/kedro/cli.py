import logging
from typing import Any, Iterable

import click
from kedro.framework.session import KedroSession

from ..utils.parser import eval_params, merge_params, str_params_to_dict
from .runner import SimpleRunner
from ..utils.progress import init_pool

logger = logging.getLogger(__name__)


@click.command()
@click.argument("pipeline", type=str, default=None)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
@click.option(
    "--all",
    is_flag=True,
    help="Also runs dataset ingestion, which is skipped by default.",
)
@click.option(
    "--synth", is_flag=True, help="Skips running split ingestion, only runs synthesis."
)
@click.option(
    "--metrics", is_flag=True, help="Useful for testing metrics, runs only metrics."
)
@click.option(
    "-r",
    "--refresh-processes",
    type=int,
    default=1,
    help="Restarts processes after `n` tasks. Lower numbers help with memory leaks but slower. Set to 0 to disable. Check `pasteur.utils.leaks` to fix.",
)
@click.option("-w", "--max-workers", type=int, default=None)
def pipe(pipeline, params, all, synth, metrics, max_workers, refresh_processes):
    """pipe(line) is a modified version of run with minified logging and shorter syntax"""

    from .pipelines.meta import (
        TAG_CHANGES_HYPERPARAMETER,
        TAG_CHANGES_PER_ALGORITHM,
        TAG_METRICS,
    )

    assert sum([all, synth, metrics]) <= 1

    param_dict = str_params_to_dict(params)

    with KedroSession.create(extra_params=param_dict) as session:
        if "ingest" in pipeline:
            logger.debug("Skipping tags for ingest pipeline.")
            tags = []
        elif all:
            logger.info("Nodes for ingesting the dataset will be run.")
            tags = []
        elif synth:
            logger.warning(
                "Skipping ingest nodes which are affected by hyperparameters, results may be invalid."
            )
            tags = [TAG_CHANGES_PER_ALGORITHM]
        elif metrics:
            logger.warning("Only running metrics nodes.")
            tags = [TAG_METRICS]
        else:
            logger.debug(
                "Skipping dataset ingestion. In case of error, run the pipeline with the name of the dataset."
            )
            tags = [TAG_CHANGES_HYPERPARAMETER, TAG_CHANGES_PER_ALGORITHM]

        # TODO: Allow for using a config value
        if refresh_processes == 0:
            refresh_processes = None

        session.run(
            tags=tags,
            runner=SimpleRunner(
                pipeline,
                " ".join(params),
                max_workers=max_workers,
                refresh_processes=refresh_processes,
            ),  # SequentialRunner(True),
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
    tab_adult.aim
    tab_adult.privbayes
    ```
    Where the first algorithm runs for all nodes that are affected by hyperparameters,
    ie executing preprocessing.

    Ingest is ran for each parameter combination, so if a parameter override
    changes the view it is honored."""

    from .mlflow import (
        check_run_done,
        get_parent_name,
        get_run_name,
        log_parent_run,
        remove_runs,
    )
    from .pipelines.meta import TAG_CHANGES_HYPERPARAMETER, TAG_CHANGES_PER_ALGORITHM

    # Create pipelines
    if alg:
        view = pipeline
        pipelines_tags = [
            (
                f"{view}.{alg[0]}",
                [TAG_CHANGES_HYPERPARAMETER, TAG_CHANGES_PER_ALGORITHM],
            ),
            *[(f"{view}.{a}", [TAG_CHANGES_PER_ALGORITHM]) for a in alg[1:]],
        ]
    elif "ingest" in pipeline:
        pipelines_tags = [(pipeline, [])]
    else:
        pipelines_tags = [
            (pipeline, [TAG_CHANGES_HYPERPARAMETER, TAG_CHANGES_PER_ALGORITHM])
        ]

    # Configure iterators
    iterable_dict = eval_params(iterator)
    hyperparam_dict = eval_params(hyperparameter)

    # Configure parent
    parent_name = get_parent_name(pipeline, alg, hyperparameter, iterator, params)
    mlflow_dict = {
        "_mlflow_parent_name": parent_name,
    }

    if clear_cache:
        with KedroSession.create() as session:
            session.load_context()
            logger.warning(f"Removing runs from mlflow with parent:\n{parent_name}")
            remove_runs(parent_name)

    runs = {}
    for iters in _process_iterables(iterable_dict | hyperparam_dict):
        param_dict = eval_params(params, iters)
        hyper_dict = {n: iters[n] for n in hyperparam_dict}
        vals = param_dict | hyper_dict
        extra_params = merge_params(vals | mlflow_dict)

        for pipeline, tags in pipelines_tags:
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
                    tags=tags,
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

    if len(runs) <= 1:
        logger.info("Only 1 run executed, skipping summary.")
        return

    # with KedroSession.create() as session:
    #     session.load_context()
    #     log_parent_run(parent_name, runs)


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
    from ..kedro.pipelines.main import NAME_LOCATION
    from ..module import get_module_dict
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

        with logging_redirect_pbar(), init_pool():
            for name in datasets:
                ds = dataset_modules[name]
                if not ds.bootstrap:
                    continue
                assert (
                    ds.folder_name
                ), "Folder name for a dataset shouldn't be null when bootstrap is supplied."

                ds_raw_location = params.get(
                    NAME_LOCATION.format(ds.folder_name),
                    path.join(raw_location, ds.folder_name),
                )
                bootstrap_location = path.join(base_location, "bootstrap", ds.folder_name)  # type: ignore
                logger.info(f'Initializing dataset "{name}" in:\n{bootstrap_location}')
                ds.bootstrap(ds_raw_location, bootstrap_location)


@click.command()
@click.argument(
    "dataset",
    type=str,
)
@click.argument(
    "output",
    type=str,
)
def export(
    dataset: str,
    output: str,
):
    """Exports a kedro dataset with name `dataset` into file `output`. Format is chosen based on filename."""
    import pyarrow as pa
    import pyarrow.csv as csv

    # Setup logging and params with kedro
    with KedroSession.create() as session:
        ctx = session.load_context()

        ds = ctx.catalog.load(dataset)
        if callable(ds):
            ds = ds()

        logger.info(f"Starting export of '{dataset}' to '{output}'.")
        if output.endswith(".csv.gz"):
            table = pa.Table.from_pandas(ds)
            with pa.CompressedOutputStream(output, "gzip") as out:
                csv.write_csv(table, out)
        elif output.endswith(".csv"):
            table = pa.Table.from_pandas(ds)
            csv.write_csv(table, output)
        elif output.endswith(".pq") or output.endswith(".parquet"):
            ds.to_parquet(output)
        else:
            assert (
                False
            ), f"Unsupported file format: '{output[output.index('.'):]}' of file '{output}'"
        logger.info("Finished export.")


@click.group(name="Pasteur")
def cli():
    """Command line tools for manipulating a Kedro project."""


cli.add_command(bootstrap)
cli.add_command(export)

cli.add_command(download)
cli.add_command(pipe)
cli.add_command(sweep)

cli.add_command(download, "dl")
cli.add_command(pipe, "p")
cli.add_command(sweep, "s")
