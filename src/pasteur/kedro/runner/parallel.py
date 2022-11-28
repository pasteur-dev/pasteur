import logging
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import chain

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner.parallel_runner import ParallelRunner
from kedro.runner.runner import run_node
from pluggy import PluginManager
from rich import get_console

from ...utils.perf import PerformanceTracker
from ...utils.progress import (
    MULTIPROCESS_ENABLE,
    RICH_TRACEBACK_ARGS,
    is_jupyter,
    logging_redirect_pbar,
    piter,
    init_pool,
    close_pool,
    tqdm,
    set_node_name,
)

logger = logging.getLogger(__name__)


def _logging_thread_fun(q):
    try:
        while True:
            record = q.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
    except EOFError:
        pass


def _replace_loggers_with_queue(q):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.root)

    for l in loggers:
        l.propagate = True
        l.handlers = []
        l.level = logging.NOTSET

    from logging.handlers import QueueHandler

    logging.root.handlers.append(QueueHandler(q))


def _replace_logging(fun, *args, _q=None, **kwargs):
    if _q is not None:
        _replace_loggers_with_queue(_q)
    
    node_name = kwargs["node"].name.split("(")[0]
    set_node_name(node_name)

    try:
        return fun(*args, **kwargs), PerformanceTracker.get_trackers()
    except Exception as e:
        get_console().print_exception(**RICH_TRACEBACK_ARGS)
        logger.error(
            f"Node \"{args[0].name.split('(')[0]}\" failed with error:\n{type(e).__name__}: {e}"
        )
        raise


def _get_required_workers_count(pipeline: Pipeline, max_workers: int | None = None):
    """
    Calculate the max number of processes required for the pipeline,
    limit to the number of CPU cores.
    """
    # Number of nodes is a safe upper-bound estimate.
    # It's also safe to reduce it by the number of layers minus one,
    # because each layer means some nodes depend on other nodes
    # and they can not run in parallel.
    # It might be not a perfect solution, but good enough and simple.
    required_processes = len(pipeline.nodes) - len(pipeline.grouped_nodes) + 1

    if not max_workers:
        return required_processes
    return min(required_processes, max_workers)


class SimpleParallelRunner(ParallelRunner):
    def __init__(
        self,
        pipe_name: str | None = None,
        params_str: str | None = None,
        max_workers: int | None = None,
    ):
        assert MULTIPROCESS_ENABLE
        self.pipe_name = pipe_name
        self.params_str = params_str
        self.max_workers = max_workers

        super().__init__(is_async=False)

    @property
    def _logger(self):
        return logging.getLogger("dummy")

    def _run(  # pylint: disable=too-many-locals,useless-suppression
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str,
    ) -> None:
        """The abstract interface for running pipelines.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            AttributeError: When the provided pipeline is not suitable for
                parallel execution.
            RuntimeError: If the runner is unable to schedule the execution of
                all pipeline nodes.
            Exception: In case of any downstream node failure.

        """
        # pylint: disable=import-outside-toplevel,cyclic-import

        nodes = pipeline.nodes
        self._validate_catalog(catalog, pipeline)
        self._validate_nodes(nodes)

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        node_dependencies = pipeline.node_dependencies
        todo_nodes = set(node_dependencies.keys())
        done_nodes = set()  # type: set[Node]
        futures = set()
        done = None

        init_pool(self.max_workers)
        max_threads = _get_required_workers_count(pipeline, self.max_workers)

        use_pbar = not is_jupyter()

        from kedro.framework.project import LOGGING

        with logging_redirect_pbar(), ThreadPoolExecutor(
            max_workers=max_threads,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),),
        ) as pool:
            # set up logging handler
            log_queue = self._manager.Queue()
            lp = threading.Thread(target=_logging_thread_fun, args=(log_queue,))
            lp.start()

            desc = f"Executing pipeline {self.pipe_name}"
            desc += f" with overrides `{self.params_str}`" if self.params_str else ""

            pbar: piter = None  # type: ignore
            if not is_jupyter() or not use_pbar:
                logger.info(desc)
            if use_pbar:
                pbar = piter(total=len(node_dependencies), desc=desc, leave=True)
            last_index = 0

            failed = False
            interrupted = False
            while not failed:
                if use_pbar:
                    n = len(done_nodes) - last_index
                    pbar.update(n)
                    last_index = len(done_nodes)

                ready = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    futures.add(
                        pool.submit(
                            _replace_logging,
                            run_node,
                            node=node,
                            catalog=catalog,
                            is_async=False,
                            hook_manager=hook_manager,
                            session_id=session_id,
                            _q=log_queue,
                        )
                    )

                if not futures:
                    if todo_nodes:
                        debug_data = {
                            "todo_nodes": todo_nodes,
                            "done_nodes": done_nodes,
                            "ready_nodes": ready,
                            "done_futures": done,
                        }
                        debug_data_str = "\n".join(
                            f"{k} = {v}" for k, v in debug_data.items()
                        )
                        raise RuntimeError(
                            f"Unable to schedule new tasks although some nodes "
                            f"have not been run:\n{debug_data_str}"
                        )
                    break  # pragma: no cover

                # Set pbar description
                if use_pbar:
                    if len(futures) == 1:
                        pbar.set_description(
                            f"Executing node {len(done_nodes) + 1:2d}/{len(nodes)}"
                        )
                    else:
                        node_str = ", ".join(
                            str(len(done_nodes) + 1 + i) for i in range(len(futures))
                        )

                        pbar.set_description(
                            f"Executing nodes {{{node_str}}}/{len(nodes)}"
                        )

                try:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    interrupted = True
                    failed = True

                    print("crashed")

                    done = set()
                    for future in futures:
                        future.cancel()
                    pool.shutdown()

                for future in chain(done):
                    try:
                        node, trackers = future.result()
                        # Merge performance tracking from the thread
                        # to the outside one.
                        PerformanceTracker.merge_trackers(trackers)
                        done_nodes.add(node)

                        # Print message
                        if (not use_pbar or not is_jupyter()) and not failed:
                            node_name = node.name.split("(")[0]
                            logger.info(
                                f"Completed node {len(done_nodes):2d}/{len(nodes):2d}: {node_name}"
                            )

                        # Decrement load counts, and release any datasets we
                        # have finished with. This is particularly important
                        # for the shared, default datasets we created above.
                        for data_set in node.inputs:
                            load_counts[data_set] -= 1
                            if (
                                load_counts[data_set] < 1
                                and data_set not in pipeline.inputs()
                            ):
                                catalog.release(data_set)
                        for data_set in node.outputs:
                            if (
                                load_counts[data_set] < 1
                                and data_set not in pipeline.outputs()
                            ):
                                catalog.release(data_set)

                    except Exception:
                        for future in futures:
                            future.cancel()

                        # Log to console
                        if not interrupted:
                            logger.error(
                                f"One (or more) of the nodes failed, exiting..."
                            )
                        failed = True

            # Close pool
            close_pool()

            # Remove logging queue
            log_queue.put(None)
            lp.join()

            if interrupted:
                logger.error(f"Received KeyboardInterrupt, exiting...")
            if failed:
                # Remove unfinished pbar
                if use_pbar:
                    pbar.leave = False
                    pbar.close()

                # exception needs to be raised for the `on_pipeline_error`
                # hook to run
                # Remove system exception hook to avoid printing exception to
                # the console
                import sys

                sys.excepthook = lambda *_: None
                raise Exception()

    def __str__(self) -> str:
        return f"<ParallelRunner {self.pipe_name} {self.params_str}>"
