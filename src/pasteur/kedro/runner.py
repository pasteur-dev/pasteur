"""``SimpleRunner`` is a modification of ``SequentialRunner`` that uses a TQDM
loading bar (friendlier for jupyter). It also force enables async save of datasets.

The TQDM loading bar is only activated if the pipeline is large enough.
"""

from collections import Counter
from itertools import chain

from kedro.io import AbstractDataSet, DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner.runner import AbstractRunner, run_node
from pluggy import PluginManager
import logging

from ..progress import piter, logging_redirect_pbar, is_jupyter

logger = logging.getLogger(__name__)

PROGRESS_BAR_MIN_PIPE_LEN = 9


def simplify_logging():
    # Add basic formatting
    logFormatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    rootLogger = logging.getLogger()
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.handlers.clear()
    rootLogger.addHandler(consoleHandler)

    # Disable var overloading warning and node print
    for package in (
        "kedro.pipeline.node",
        "kedro.config.common",
        "kedro.framework.session.session",
        "kedro.extras.extensions.ipython",
        "IPKernelApp",
        "dummy",
    ):
        logger = logging.getLogger(package)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    # Disable all customised loaders
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(consoleHandler)
    return loggers


class SimpleRunner(AbstractRunner):
    """``SimpleRunner`` is a modification of ``SequentialRunner`` that uses a TQDM
    loading bar. It also force enables async save of datasets.
    """

    def __init__(
        self,
        pipe_name: str | None = None,
        params_str: str | None = None,
    ):
        self.pipe_name = pipe_name
        self.params_str = params_str

        super().__init__(is_async=True)

    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        return MemoryDataSet()

    @property
    def _logger(self):
        return logging.getLogger("dummy")

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """The method implementing sequential pipeline running.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            Exception: in case of any downstream node failure.
        """
        nodes = pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))

        use_pbar = len(nodes) >= PROGRESS_BAR_MIN_PIPE_LEN and not is_jupyter()

        with logging_redirect_pbar():
            desc = f"Executing pipeline {self.pipe_name}"
            desc += f" with overrides `{self.params_str}`" if self.params_str else ""

            if not is_jupyter() or not use_pbar:
                logger.info(desc)
            if use_pbar:
                pbar = piter(nodes, desc=desc, leave=True)
            else:
                pbar = nodes
            for exec_index, node in enumerate(pbar):
                node_name = node.name.split("(")[0]
                if use_pbar:
                    pbar.set_description(f"Executing {node_name}")
                try:
                    run_node(node, catalog, hook_manager, self._is_async, session_id)
                    done_nodes.add(node)
                except Exception:
                    self._suggest_resume_scenario(pipeline, done_nodes)
                    raise

                # decrement load counts and release any data sets we've finished with
                for data_set in node.inputs:
                    load_counts[data_set] -= 1
                    if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                        catalog.release(data_set)
                for data_set in node.outputs:
                    if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                        catalog.release(data_set)

                if not use_pbar or not is_jupyter():
                    logger.info(
                        f"Completed node {exec_index + 1:2d}/{len(nodes):2d}: {node_name}"
                    )

            if use_pbar and exec_index == len(nodes) - 1:
                pbar.set_description(desc.replace("Executing", "Executed"))