from typing import Iterable
import click
from kedro.framework.cli.project import project_group
from kedro.framework.cli.utils import CONTEXT_SETTINGS
from kedro.framework.session import KedroSession

from .utils import str_params_to_dict
from .kedro.runner import SimpleRunner


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@project_group.command()
@click.argument("pipeline", type=str, default=None)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
def p(pipeline, params):
    """p(ipeline) is a modified version of run with minified logging and shorter syntax"""

    param_dict = str_params_to_dict(params)

    with KedroSession.create(env=None, extra_params=param_dict) as session:
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
    null = object()
    iterator_dict = {n: iter(v) for n, v in iterables.items()}
    value_dict = {n: next(v, None) for n, v in iterator_dict.items()}

    has_combs = True
    while has_combs:
        yield value_dict

        has_combs = False
        for name, it in iterator_dict.items():
            val = next(it, null)

            if val is null:
                new_it = iter(iterables[name])
                iterator_dict[name] = new_it
                value_dict[name] = next(new_it, None)
            else:
                value_dict[name] = val
                has_combs = True
                break


@project_group.command()
@click.argument("pipeline", type=str, default=None)
@click.option("--iterator", "-i", multiple=True)
@click.option("--hyperparameter", "-h", multiple=True)
@click.argument(
    "params",
    nargs=-1,
    type=str,
)
def s(pipeline, iterator, hyperparameter, params):
    """Similar to p, s(weep) allows in addition a hyperparameter sweep.
    
    By using `-i` an iterator can be defined (ex. `-i i="range(5)"`), which will
    make the pipeline run for each value of i. Then i can be used in expressions
    with other variables that are passed as arguments (ex. `j="0.2*i"`).
    
    If an iterator is also a hyperparameter (ex. `-h e1="[0.1,0.2,0.3]"`)
    then `-h` can be used, which will both sweep and pass the variable as an
    override at the same time (it is equal to `-i val=<iterable> val=val`). """

    iterable_dict = str_params_to_dict(iterator)
    hyperparam_dict = str_params_to_dict(hyperparameter)

    for vals in _process_iterables(iterable_dict | hyperparam_dict):
        param_dict = str_params_to_dict(params, vals)
        hyper_dict = {n: vals[n] for n in hyperparam_dict}
        vals = param_dict | hyper_dict

        with KedroSession.create(env=None, extra_params=vals) as session:
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
