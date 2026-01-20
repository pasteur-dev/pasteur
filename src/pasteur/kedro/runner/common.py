import logging
from typing import Any, Callable, cast

from kedro.io import DataCatalog
from kedro.pipeline.node import Node
from pluggy import PluginManager
from kedro.pipeline import Pipeline, transcoding
from rich import get_console

from ...utils.perf import PerformanceTracker
from ...utils.progress import (
    RICH_TRACEBACK_ARGS,
    process_in_parallel,
    set_node_name,
    IS_AGENT,
)

logger = logging.getLogger(__name__)


def _collect_inputs_from_hook(  # noqa: PLR0913
    node: Node,
    catalog: Any,
    inputs: dict[str, Any],
    is_async: bool,
    hook_manager: PluginManager,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    inputs = inputs.copy()  # shallow copy to prevent in-place modification by the hook
    hook_response = hook_manager.hook.before_node_run(
        node=node,
        catalog=catalog,
        inputs=inputs,
        is_async=is_async,
        session_id=session_id,
        run_id=run_id,
    )

    additional_inputs = {}
    if (
        hook_response is not None
    ):  # all hooks on a _NullPluginManager will return None instead of a list
        for response in hook_response:
            if response is not None and not isinstance(response, dict):
                response_type = type(response).__name__
                raise TypeError(
                    f"'before_node_run' must return either None or a dictionary mapping "
                    f"dataset names to updated values, got '{response_type}' instead."
                )
            additional_inputs.update(response or {})

    return additional_inputs


def _call_node_run(  # noqa: PLR0913
    node: Node,
    catalog: Any,
    inputs: dict[str, Any],
    is_async: bool,
    hook_manager: PluginManager,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    try:
        outputs = node.run(inputs)
    except Exception as exc:
        hook_manager.hook.on_node_error(
            error=exc,
            node=node,
            catalog=catalog,
            inputs=inputs,
            is_async=is_async,
            session_id=session_id,
        )
        raise exc
    hook_manager.hook.after_node_run(
        node=node,
        catalog=catalog,
        inputs=inputs,
        outputs=outputs,
        is_async=is_async,
        session_id=session_id,
        run_id=run_id,
    )
    return outputs


def _task_worker(fun: Callable[..., dict[str, Any]], catalog: DataCatalog):
    res = fun()
    for name, data in res.items():
        catalog.save(name, data)


def run_expanded_node(
    node: Node,
    catalog: DataCatalog,
    hook_manager: PluginManager,
    session_id: str | None = None,
    run_id: str | None = None,
) -> Node:
    """Handles expanded output's option of returning a set of callables.
    Callables are processed by the process pool and the result is untangled
    by `ExpandedNode` into the dictionary that gets saved in the catalog.

    It also handles printing exceptions."""
    node_name = node.name.split("(")[0]
    set_node_name(node_name)
    try:
        t = PerformanceTracker.get("nodes")
        t.log_to_file()
        t.start(node_name)

        # Readd mlflow tracking
        if run_id:
            import mlflow

            if not mlflow.active_run():
                mlflow.start_run(run_id=run_id)

        inputs = {}
        session_id = cast(str, session_id)

        for name in node.inputs:
            hook_manager.hook.before_dataset_loaded(node=node, dataset_name=name)  # type: ignore
            inputs[name] = catalog.load(name)
            hook_manager.hook.after_dataset_loaded(node=node, dataset_name=name, data=inputs[name])  # type: ignore

        is_async = False

        additional_inputs = _collect_inputs_from_hook(
            node,
            catalog,
            inputs,
            is_async,
            hook_manager,
            session_id=session_id,
            run_id=run_id,
        )
        inputs.update(additional_inputs)

        outputs = _call_node_run(
            node,
            catalog,
            inputs,
            is_async,
            hook_manager,
            session_id=session_id,
            run_id=run_id,
        )
    except Exception as e:
        print_exc = not (isinstance(e, RuntimeError) and str(e) == "subprocess failed")
        if print_exc:
            if not IS_AGENT:
                # Prevent printing traceback for subprocesses that crash
                get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Saving "{node_name}" failed with error:\n{type(e).__name__}: {e}',
                exc_info=IS_AGENT,
            )
        logger.info(
            f'To continue from this node, add `-c "{node.name.split("(", 1)[0]}" -s "{session_id}"` to a pipeline run\'s arguments.'
        )
        raise e

    # Clear outputs
    for name in node.outputs:
        d = catalog[name]
        if hasattr(d, "reset"):
            getattr(d, "reset")()

    if isinstance(outputs, set):
        # TODO: Fix hooks
        # When a set is received, process it in parallel
        try:
            process_in_parallel(
                _task_worker,
                per_call_args=[{"fun": fun} for fun in outputs],
                base_args={"catalog": catalog},
                desc=f"Processing tasks ({node.name.split('(')[0]:>25s})",
            )
        except Exception as e:
            logger.info(
                f'To continue from this node, add `-c "{node.name.split("(", 1)[0]}" -s "{session_id}"` to a pipeline run\'s arguments.'
            )
            raise e
    else:
        try:
            for name, data in outputs.items():
                hook_manager.hook.before_dataset_saved(node=node, dataset_name=name, data=data)  # type: ignore
                catalog.save(name, data)
                hook_manager.hook.after_dataset_saved(node=node, dataset_name=name, data=data)  # type: ignore
        except Exception as e:
            print_exc = not (
                isinstance(e, RuntimeError) and str(e) == "subprocess failed"
            )
            if print_exc and not IS_AGENT:
                # Prevent printing traceback for subprocesses that crash
                get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Saving "{node_name}" failed with error:\n{type(e).__name__}: {e}',
                exc_info=print_exc and IS_AGENT,
            )
            # TODO: Handle dataset errors better
    for name in node.confirms:
        catalog.confirm(name)

    t.stop(node.name.split("(")[0])
    return node


def resume_from(pipeline: Pipeline, node: str | None):
    # Since the list is always the same, resume from the last node that was ran
    groups = pipeline.grouped_nodes

    if node is None:
        return [n for group in groups for n in group]

    out = []
    started = False

    for g in groups:
        group_started = started
        for n in g:
            if group_started:
                out.append(n)
                continue

            # Append us, skip the rest of the group
            # assuming it ran. Fair assumption that skips extra
            # steps but could cause failures down the pipeline
            if n.name == node:
                started = True
                out.append(n)

    return out


def resume_from_dependencies(pipeline: Pipeline, node: str):
    # For the parallel runner, get a bit more complicated.
    # Only run nodes that are children of the resume node
    dependencies: dict[Node, set[Node]] = pipeline.node_dependencies.copy()

    # Find resume node
    resume_node = None
    for n in dependencies:
        if n.name.startswith(node):
            resume_node = n
            break
    assert resume_node is not None, f"Could not find node {node}"

    keep = {resume_node}
    dependencies[resume_node] = set()
    stack: set[Node] = {resume_node}

    while stack:
        n = stack.pop()

        for child, parents in dependencies.items():
            if n in parents and child not in keep:
                keep.add(child)
                stack.add(child)

    new_dependencies = {}
    for n in keep:
        new_dependencies[n] = dependencies[n].intersection(keep)

    return new_dependencies, list(new_dependencies)
