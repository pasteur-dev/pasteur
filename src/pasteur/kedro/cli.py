"""In this module, all Pasteur related cli commands are defined.

You can access them through `pasteur <command>` or `kedro <command>`."""

import logging
from typing import Any, Iterable

import click
from kedro.framework.session import KedroSession

from pasteur.kedro.mlflow.base import get_git_suffix

from ..utils.parser import eval_params, merge_params, str_params_to_dict
from ..utils.progress import init_pool
from .runner import SimpleRunner

logger = logging.getLogger(__name__)


def create_session(
    cls,
    project_path: str | None = None,
    save_on_close: bool = True,
    env: str | None = None,
    runtime_params: dict[str, Any] | None = None,
    conf_source: str | None = None,
    session_id: str | None = None,
) -> KedroSession:
    # We have to stub this to change session_id

    import getpass
    import os

    from kedro.framework.project import validate_settings
    from kedro.framework.session.session import _describe_git, _jsonify_cli_context
    from kedro.io.core import generate_timestamp

    if session_id:
        logger.info(f'Reusing session id: "{session_id}"')

    validate_settings()

    session = cls(
        project_path=project_path,
        session_id=generate_timestamp() if session_id is None else session_id,
        save_on_close=save_on_close,
        conf_source=conf_source,
    )

    # have to explicitly type session_data otherwise mypy will complain
    # possibly related to this: https://github.com/python/mypy/issues/1430
    session_data: dict[str, Any] = {
        "project_path": session._project_path,
        "session_id": session.session_id,
    }

    ctx = click.get_current_context(silent=True)
    if ctx:
        session_data["cli"] = _jsonify_cli_context(ctx)

    env = env or os.getenv("KEDRO_ENV")
    if env:
        session_data["env"] = env

    if runtime_params:
        session_data["runtime_params"] = runtime_params

    try:
        session_data["username"] = getpass.getuser()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Unable to get username. Full exception: %s", exc
        )

    session_data.update(**_describe_git(session._project_path))
    session._store.update(session_data)

    return session


@click.command
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
@click.option("--pre", is_flag=True, help="Only runs split preprocessing.")
@click.option(
    "--synth",
    is_flag=True,
    help="Skips running split preprocessing, only runs synthesis.",
)
@click.option(
    "--metrics", is_flag=True, help="Useful for testing metrics, runs only metrics."
)
@click.option(
    "--sample",
    is_flag=True,
    help="Loads existing model, runs refresh (e.g. mirror descent), then samples.",
)
@click.option(
    "-r",
    "--refresh-processes",
    type=int,
    default=1,
    help="Restarts processes after `n` tasks. Lower numbers help with memory leaks but slower. Set to 0 to disable. Check `pasteur.utils.leaks` to fix.",
)
@click.option("-w", "--max-workers", type=int, default=None)
@click.option(
    "-c",
    "--continue-from",
    type=str,
    default=None,
    help="Node name to continue from, all previous nodes are skipped (nodes in the same topological generation are also skipped).",
)
@click.option(
    "-s",
    "--session-id",
    type=str,
    default=None,
    help="Session ID to use. Allows reusing artifacts from a previous run.",
)
@click.option(
    "--stochastic",
    is_flag=True,
    help="Override random_state with a random seed.",
)
@click.pass_context
def pipe(
    ctx,
    pipeline,
    params,
    all,
    pre,
    synth,
    metrics,
    sample,
    max_workers,
    refresh_processes,
    continue_from,
    session_id,
    stochastic,
):
    """pipe(line) is a modified version of run with minified logging and shorter syntax"""

    from .pipelines.meta import (
        TAG_ALWAYS,
        TAG_CHANGES_HYPERPARAMETER,
        TAG_CHANGES_PER_ALGORITHM,
        TAG_REVERSE,
        TAG_SAMPLE,
        TAG_METRICS,
    )

    assert sum([all, pre, synth, metrics, sample]) <= 1

    param_dict = str_params_to_dict(params)

    if stochastic:
        import random
        param_dict["random_state"] = random.randint(0, 2**31 - 1)
        logger.info(f"Stochastic mode: random_state={param_dict['random_state']}")

    cmd: str = ctx.info_name
    if cmd.startswith("i"):
        match cmd:
            case "iv" | "ingest_view":
                pipeline = f"ingest_view.{pipeline}"
            case "id" | "ingest_dataset":
                pipeline = f"ingest_dataset.{pipeline}"
            case "i" | "ingest":
                pipeline = f"{pipeline}.ingest"

    with create_session(
        KedroSession, runtime_params=param_dict, env="base", session_id=session_id
    ) as session:
        if "ingest" in pipeline:
            logger.debug("Skipping tags for ingest pipeline.")
            tags = []
        elif all:
            logger.info("Nodes for ingesting the dataset will be run.")
            tags = []
        elif pre:
            logger.info("Only nodes for preprocessing the view will be run.")
            tags = [TAG_ALWAYS, TAG_CHANGES_HYPERPARAMETER]
        elif synth:
            logger.warning(
                "Skipping ingest nodes which are affected by hyperparameters, results may be invalid."
            )
            tags = [TAG_ALWAYS, TAG_CHANGES_PER_ALGORITHM]
        elif metrics:
            logger.warning("Only running metrics nodes.")
            tags = [TAG_METRICS]

            # Disable load versions for metrics, due to missing .e.g, models
            from pasteur.kedro.hooks import pasteur

            if pasteur:
                pasteur.load_any = True
        elif sample:
            logger.info(
                "Loading existing model, refreshing, sampling, and running metrics."
            )
            tags = [TAG_ALWAYS, TAG_SAMPLE, TAG_REVERSE, TAG_METRICS]

            from pasteur.kedro.hooks import pasteur

            if pasteur:
                pasteur.load_any = True
                pasteur.refresh = True
        else:
            logger.debug(
                "Skipping dataset ingestion. In case of error, run the pipeline with the name of the dataset."
            )
            tags = [TAG_ALWAYS, TAG_CHANGES_HYPERPARAMETER, TAG_CHANGES_PER_ALGORITHM]

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
                resume_node=continue_from,
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
@click.option(
    "--all",
    is_flag=True,
    help="Also runs dataset ingestion, which is skipped by default.",
)
@click.option("--pre", is_flag=True, help="Only runs split preprocessing.")
@click.option(
    "--synth",
    is_flag=True,
    help="Skips running split preprocessing, only runs synthesis.",
)
@click.option(
    "--metrics", is_flag=True, help="Useful for testing metrics, runs only metrics."
)
@click.option(
    "--sample",
    is_flag=True,
    help="Loads existing model, runs refresh (e.g. mirror descent), then samples.",
)
@click.option("--skip-parent", "-p", is_flag=True)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
@click.option(
    "--refresh-processes",
    type=int,
    default=1,
    help="Restarts processes after `n` tasks. Lower numbers help with memory leaks but slower. Set to 0 to disable. Check `pasteur.utils.leaks` to fix.",
)
@click.option("-w", "--max-workers", type=int, default=None)
@click.option(
    "-r",
    "--runs",
    type=int,
    default=1,
    help="Run each pipeline multiple times. Appends rN to the run name in mlflow.",
)
@click.option(
    "-s",
    "--stochastic",
    is_flag=True,
    help="Override random_state with a random seed for each run.",
)
@click.pass_context
def sweep(
    ctx,
    pipeline,
    alg,
    iterator,
    hyperparameter,
    skip_parent,
    all,
    pre,
    synth,
    metrics,
    sample,
    params,
    clear_cache,
    max_workers,
    refresh_processes,
    runs,
    stochastic,
):
    """Similar to pipe, sweep allows in addition a hyperparameter sweep.

    By using `-i` an iterator can be defined (e.g., `-i i="range(5)"`), which will
    make the pipeline run for each value of i. Then i can be used in expressions
    with other variables that are passed as arguments (ex. `j="0.2*i"`).

    If an iterator is also a hyperparameter (e.g., `-h e1="[0.1,0.2,0.3]"`)
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

    import builtins

    builtins_all = builtins.all

    from .mlflow import (
        check_run_done,
        get_parent_name,
        get_run_name,
        log_parent_run,
        remove_runs,
    )
    from .pipelines.meta import (
        TAG_ALWAYS,
        TAG_CHANGES_HYPERPARAMETER,
        TAG_CHANGES_PER_ALGORITHM,
        TAG_REVERSE,
        TAG_SAMPLE,
        TAG_METRICS,
    )

    assert sum([all, pre, synth, metrics, sample]) <= 1

    # Create pipelines
    load_any = False
    refresh = False
    if all:
        logger.info("Nodes for ingesting the dataset will be run.")
        default_tags = []
    elif pre:
        logger.info("Only nodes for preprocessing the view will be run.")
        default_tags = [TAG_ALWAYS, TAG_CHANGES_HYPERPARAMETER]
    elif synth:
        logger.warning(
            "Skipping ingest nodes which are affected by hyperparameters, results may be invalid."
        )
        default_tags = [TAG_ALWAYS, TAG_CHANGES_PER_ALGORITHM]
    elif metrics:
        logger.warning("Only running metrics nodes.")
        default_tags = [TAG_METRICS]
        load_any = True
    elif sample:
        logger.info(
            "Loading existing model, refreshing, sampling, and running metrics."
        )
        default_tags = [TAG_ALWAYS, TAG_SAMPLE, TAG_REVERSE, TAG_METRICS]
        load_any = True
        refresh = True
    else:
        default_tags = [
            TAG_ALWAYS,
            TAG_CHANGES_HYPERPARAMETER,
            TAG_CHANGES_PER_ALGORITHM,
        ]

    if alg:
        view = pipeline
        pipelines_tags = [
            (
                f"{view}.{alg[0]}",
                list(default_tags),
            ),
            *[
                (
                    f"{view}.{a}",
                    [t for t in default_tags if t != TAG_CHANGES_HYPERPARAMETER],
                )
                for a in alg[1:]
            ],
        ]
    else:
        pipelines_tags = [
            (
                pipeline,
                list(default_tags),
            )
        ]

    # Configure iterators
    iterable_dict = eval_params(iterator)
    hyperparam_dict = eval_params(hyperparameter)

    # Configure parent
    parent_name = get_parent_name(pipeline, alg, hyperparameter, iterator, params)
    if runs > 1:
        parent_name += f" -r {runs}"
    mlflow_dict = {
        "_mlflow_parent_name": parent_name,
    }

    if clear_cache:
        with KedroSession.create(env="base") as session:
            session.load_context()
            logger.warning(f"Removing runs from mlflow with parent:\n{parent_name}")
            remove_runs(parent_name, delete_parent=False)

    # Apply load_any / refresh hooks (same as pipe)
    if load_any or refresh:
        from pasteur.kedro.hooks import pasteur as pasteur_hook

        if pasteur_hook:
            if load_any:
                pasteur_hook.load_any = True
            if refresh:
                pasteur_hook.refresh = True

    # TODO: Allow for using a config value
    if refresh_processes == 0:
        refresh_processes = None

    import random

    num_runs = runs
    # Determine base seed for reproducible multi-run sweeps
    if stochastic:
        base_seed = random.randint(0, 2**31 - 1)
        logger.info(f"Stochastic mode: base random_state={base_seed}")
    else:
        base_seed = None

    # Read configured random_state for deriving follow-up run seeds
    if num_runs > 1 and not stochastic:
        with KedroSession.create(env="base") as session:
            ctx = session.load_context()
            configured_seed = ctx.params.get("random_state", 0)
    else:
        configured_seed = None

    run_results = {}
    ingested = False
    runtime_params = {}

    for run_idx in range(num_runs):
        for iters in _process_iterables(iterable_dict | hyperparam_dict):
            param_dict = eval_params(params, iters)
            hyper_dict = {n: iters[n] for n in hyperparam_dict}
            vals = param_dict | hyper_dict
            runtime_params = merge_params(vals | mlflow_dict)

            alg_only_hyper = builtins_all([n.startswith("alg") for n in vals])

            for i, (pipeline, tags) in enumerate(pipelines_tags):
                tags = list(tags)
                params_skipped = False
                if (alg_only_hyper and ingested) or i:
                    params_skipped = True
                    if TAG_CHANGES_HYPERPARAMETER in tags:
                        tags.remove(TAG_CHANGES_HYPERPARAMETER)

                suffix_dict = (
                    {"_mlflow_run_suffix": f"r{run_idx + 1}"}
                    if num_runs > 1
                    else {}
                )
                # Determine seed for this run
                seed_dict = {}
                if stochastic:
                    seed_dict["random_state"] = base_seed + run_idx
                elif num_runs > 1 and run_idx > 0:
                    # Vary configured seed for follow-up runs
                    seed_dict["random_state"] = configured_seed + run_idx

                run_runtime_params = merge_params(
                    vals | mlflow_dict | suffix_dict | seed_dict
                )

                with KedroSession.create(
                    runtime_params=run_runtime_params, env="base"
                ) as session:
                    session.load_context()

                    run_name = get_run_name(pipeline, run_runtime_params)
                    run_vals = (
                        {**vals, "_run": f"r{run_idx + 1}"}
                        if num_runs > 1
                        else vals
                    )
                    if alg:
                        # if alg exists add its name
                        run_results[run_name] = {"_alg": alg, **run_vals}
                    elif alg is not None:
                        # ingest pipeline has None and should be skipped from cross-eval
                        run_results[run_name] = run_vals

                    if check_run_done(
                        run_name,
                        None if skip_parent else parent_name,
                        None if skip_parent else get_git_suffix(),
                    ):
                        logger.warning(f"Run '{run_name}' is complete, skipping...")
                        continue

                    if params_skipped:
                        logger.warning(
                            "Skipping ingestion since hyperparameters are the same"
                        )

                    session.run(
                        tags=tags,
                        runner=SimpleRunner(
                            pipeline,
                            " ".join(f"{n}={v}" for n, v in vals.items()),
                            max_workers=max_workers,
                            refresh_processes=refresh_processes,
                        ),
                        node_names="",
                        from_nodes="",
                        to_nodes="",
                        from_inputs="",
                        to_outputs="",
                        load_versions={},
                        pipeline_name=pipeline,
                    )
                    ingested = True

    if len(run_results) <= 1:
        logger.info("Only 1 run executed, skipping summary.")
        return

    with KedroSession.create(runtime_params=runtime_params, env="base") as session:
        ctx = session.load_context()
        experiment_id = getattr(ctx, "mlflow").get_experiment_id(pipeline.split(".")[0])
        log_parent_run(
            parent_name, run_results, skip_parent=skip_parent, experiment_id=experiment_id
        )


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

    Only downloads missing files, can be ran to verify dataset is downloaded correctly.
    """
    from ..dataset import Dataset
    from ..extras.download import datasets as EXTRA_DATASETS
    from ..module import get_module_dict
    from ..utils.download import get_description, main

    # Setup logging and params with kedro
    with KedroSession.create(env="base") as session:
        ctx = session.load_context()

        dataset_modules = get_module_dict(Dataset, getattr(ctx, "pasteur").modules)
        all_datasets = dict(EXTRA_DATASETS)
        for name, ds in dataset_modules.items():
            if isinstance(ds.raw_sources, dict):
                all_datasets.update(ds.raw_sources)
            elif ds.raw_sources is not None:
                all_datasets[name] = ds.raw_sources

        logger.info(get_description(all_datasets))
        if not datasets:
            return

        sel_datasets = {}
        for ds in datasets:
            if ds not in all_datasets:
                logger.error(f"Raw sources for {ds} not found.")
                return
            sel_datasets[ds] = all_datasets[ds]

        download_dir = download_dir or getattr(ctx, "pasteur").raw_location
        assert download_dir, f"Download dir is empty"

        if not accept:
            logger.error(
                "You have to accept to the license of the data stores you're about to download from (--accept/-a)."
            )
        else:
            main(download_dir, sel_datasets, user)


@click.command()
@click.argument(
    "datasets",
    nargs=-1,
    type=str,
)
def bootstrap(
    datasets: tuple[str, ...],
):
    """Preprocesses downloaded datasets which require it so they can be loaded by kedro."""
    from os import path

    from ..dataset import Dataset
    from ..kedro.pipelines.main import NAME_LOCATION
    from ..module import get_module_dict
    from ..utils.progress import logging_redirect_pbar

    # Setup logging and params with kedro
    with KedroSession.create(env="base") as session:
        ctx = session.load_context()

        dataset_modules = get_module_dict(Dataset, ctx.pasteur.modules)  # type: ignore

        if not datasets:
            datasets = tuple(dataset_modules)

        for dataset in datasets:
            if dataset not in dataset_modules:
                logger.error(f"Module for dataset `{dataset}` not currently loaded.")
                return

        locations = ctx.config_loader.get("locations")
        raw_location = locations["raw"]
        base_location = locations["base"]
        bootstrap_location = locations.get(
            "bootstrap", path.join(base_location, "bootstrap")
        )

        with logging_redirect_pbar(), init_pool():
            for name in datasets:
                ds = dataset_modules[name]
                if not ds.bootstrap:
                    continue
                assert (
                    ds.folder_name
                ), "Folder name for a dataset shouldn't be null when bootstrap is supplied."

                ds_raw_location = locations.get(
                    NAME_LOCATION.format(ds.folder_name),
                    path.join(raw_location, ds.folder_name),
                )
                bootstrap_location_ds = path.join(bootstrap_location, ds.folder_name)
                logger.info(
                    f'Initializing dataset "{name}" in:\n{bootstrap_location_ds}'
                )
                ds.bootstrap(ds_raw_location, bootstrap_location_ds)


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
    with KedroSession.create(env="base") as session:
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

from ..litmus.cli import litmus

cli.add_command(litmus)

cli.add_command(download, "dl")
cli.add_command(pipe, "p")
cli.add_command(sweep, "s")

# TODO: fix styling in help menu
cli.add_command(pipe, "ingest_dataset")
cli.add_command(pipe, "id")
cli.add_command(pipe, "ingest_view")
cli.add_command(pipe, "iv")
cli.add_command(pipe, "ingest")
cli.add_command(pipe, "i")
