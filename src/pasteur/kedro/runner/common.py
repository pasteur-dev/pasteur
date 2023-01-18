import logging
from typing import Any, Callable, cast

from kedro.io import DataCatalog
from kedro.pipeline.node import Node
from kedro.runner.runner import _call_node_run, _collect_inputs_from_hook
from pluggy import PluginManager
from rich import get_console

from ...utils.perf import PerformanceTracker
from ...utils.progress import RICH_TRACEBACK_ARGS, process_in_parallel, set_node_name

logger = logging.getLogger(__name__)


def _task_worker(fun: Callable[..., dict[str, Any]], catalog: DataCatalog):
    res = fun()
    for name, data in res.items():
        catalog.save(name, data)


def run_expanded_node(
    node: Node,
    catalog: DataCatalog,
    hook_manager: PluginManager,
    session_id: str | None = None,
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

        inputs = {}
        session_id = cast(str, session_id)

        for name in node.inputs:
            hook_manager.hook.before_dataset_loaded(dataset_name=name)  # type: ignore
            inputs[name] = catalog.load(name)
            hook_manager.hook.after_dataset_loaded(dataset_name=name, data=inputs[name])  # type: ignore

        is_async = False

        additional_inputs = _collect_inputs_from_hook(
            node, catalog, inputs, is_async, hook_manager, session_id=session_id
        )
        inputs.update(additional_inputs)

        outputs = _call_node_run(
            node, catalog, inputs, is_async, hook_manager, session_id=session_id
        )
    except Exception as e:
        if not (isinstance(e, RuntimeError) and str(e) == "subprocess failed"):
            # Prevent printing traceback for subprocesses that crash
            get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Node "{node_name}" failed with error:\n{type(e).__name__}: {e}'
            )
        raise

    # Clear outputs
    for name in node.outputs:
        d = catalog._get_dataset(name)
        if hasattr(d, "reset"):
            getattr(d, "reset")()

    if isinstance(outputs, set):
        # TODO: Fix hooks
        # When a set is received, process it in parallel
        process_in_parallel(
            _task_worker,
            per_call_args=[{"fun": fun} for fun in outputs],
            base_args={"catalog": catalog},
            desc=f"Processing tasks ({node.name.split('(')[0]:>25s})",
        )
    else:
        try:
            for name, data in outputs.items():
                hook_manager.hook.before_dataset_saved(dataset_name=name, data=data)  # type: ignore
                catalog.save(name, data)
                hook_manager.hook.after_dataset_saved(dataset_name=name, data=data)  # type: ignore
        except Exception as e:
            if not (isinstance(e, RuntimeError) and str(e) == "subprocess failed"):
                # Prevent printing traceback for subprocesses that crash
                get_console().print_exception(**RICH_TRACEBACK_ARGS)
            logger.error(
                f'Saving "{node_name}" failed with error:\n{type(e).__name__}: {e}'
            )
            # TODO: Handle dataset errors better
    for name in node.confirms:
        catalog.confirm(name)

    t.stop(node.name.split("(")[0])
    return node
