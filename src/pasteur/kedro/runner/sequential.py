"""``SimpleRunner`` is a modification of ``SequentialRunner`` that uses a TQDM
loading bar (friendlier for jupyter). It also force enables async save of datasets.

The TQDM loading bar is only activated if the pipeline is large enough.
"""

import logging
from collections import Counter
from itertools import chain

from kedro.io import AbstractDataSet, DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner.runner import AbstractRunner
from pluggy import PluginManager

from ...utils.progress import is_jupyter, logging_redirect_pbar, piter, set_node_name
from .common import run_expanded_node

logger = logging.getLogger(__name__)


class SimpleSequentialRunner(AbstractRunner):
    """``SimpleRunner`` is a modification of ``SequentialRunner`` that uses a TQDM
    loading bar. It also force enables async save of datasets.
    """

    def __init__(
        self,
        pipe_name: str | None = None,
        params_str: str | None = None,
        max_workers: int | None = None,
        refresh_processes: int | None = None
    ):
        self.pipe_name = pipe_name
        self.params_str = params_str

        super().__init__(is_async=False)

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

        use_pbar = not is_jupyter()

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
                    set_node_name(node_name)
                    run_expanded_node(node, catalog, hook_manager, session_id)
                    done_nodes.add(node)
                except KeyboardInterrupt as e:
                    import sys

                    logger.error(f"Received KeyboardInterrupt, exiting...")
                    sys.excepthook = lambda *_: None

                    if use_pbar:
                        pbar.leave = False
                        pbar.close()

                    raise Exception() from e  # raising KeyboardInterrupt results in Aborted! being printed.
                except Exception as e:
                    import sys

                    sys.excepthook = lambda *_: None  # exception was printed already

                    if use_pbar:
                        pbar.leave = False
                        pbar.close()

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

    def __str__(self) -> str:
        return f"<SequentialRunner {self.pipe_name} {self.params_str}>"
