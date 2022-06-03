from copy import deepcopy
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from kedro.framework.context import KedroContext

import great_expectations.exceptions as ge_exceptions
from great_expectations.data_context import DataContext
from great_expectations.core.batch import (
    BatchDefinition,
    BatchRequest,
    RuntimeBatchRequest,
)
from great_expectations.data_context.types.base import (
    DataContextConfig,
)
from great_expectations.core.batch_spec import (
    BatchMarkers,
    BatchSpec,
    RuntimeDataBatchSpec,
)
from great_expectations.core.batch import BatchMarkers
from great_expectations.datasource.data_connector import DataConnector
from great_expectations.execution_engine import ExecutionEngine
from great_expectations.core.id_dict import IDDict


log = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
kedro_context = None
kedro_assets = None


def dataset_to_datasource(datasets: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """This function maps kedro datasets to datasources, assets, and tags used by ge"""
    datasources = {}
    for dataset in datasets:
        datasource, split, asset = dataset.split(".")
        is_decoded = "decoded_" in asset
        if is_decoded:
            asset = asset.replace("decoded_", "")

        if datasource not in datasources:
            datasources[datasource] = {}

        if asset not in datasources[datasource]:
            datasources[datasource][asset] = {}

        datasources[datasource][asset][dataset] = {
            "split": split,
            "is_decoded": is_decoded,
        }

    return datasources


class KedroDataConnector(DataConnector):
    """The KedroDataConnector allows for opening kedro catalog datasets, which are
    generated dynamically or statically (from catalog.yml and from python code),
    into great_expectations

    It returns a `RuntimeDataBatchSpec`, confusing the execution engine into
    loading the dataframe.
    """

    def __init__(
        self,
        name: str,
        datasource_name: str,
        execution_engine: ExecutionEngine,
        batch_spec_passthrough: Optional[dict] = None,
    ) -> None:
        super().__init__(
            name, datasource_name, execution_engine, batch_spec_passthrough
        )

        self._name = datasource_name
        global kedro_assets
        self._assets = kedro_assets[datasource_name]

    def get_batch_data_and_metadata(
        self,
        batch_definition: BatchDefinition,
    ) -> Tuple[Any, BatchSpec, BatchMarkers,]:  # batch_data
        batch_spec: RuntimeDataBatchSpec = self.build_batch_spec(
            batch_definition=batch_definition,
        )
        batch_data, batch_markers = self._execution_engine.get_batch_data_and_markers(
            batch_spec=batch_spec
        )
        self._execution_engine.load_batch_data(batch_definition.id, batch_data)
        return (
            batch_data,
            batch_spec,
            batch_markers,
        )

    def build_batch_spec(
        self, batch_definition: BatchDefinition
    ) -> RuntimeDataBatchSpec:
        global kedro_context
        batch_spec: BatchSpec = super().build_batch_spec(
            batch_definition=batch_definition
        )
        asset = self._assets[batch_definition.data_asset_name]
        found = False
        for name, batch in asset.items():
            if batch == batch_definition["batch_identifiers"]:
                found = True
                break
        if not found:
            logger.warning(
                f"Batch matching IDs {str(batch_spec['batch_identifiers'])} not found, loading {name}."
            )
        batch_spec["batch_data"] = kedro_context.catalog.load(name)
        return RuntimeDataBatchSpec(batch_spec)

    def get_available_data_asset_names(self) -> List[str]:
        return list(self._assets.keys())

    def get_batch_definition_list_from_batch_request(
        self,
        batch_request: Union[BatchRequest, RuntimeBatchRequest],
    ) -> List[BatchDefinition]:
        asset = self._assets[batch_request.data_asset_name]
        return [
            BatchDefinition(
                datasource_name=self.datasource_name,
                data_connector_name=self.name,
                data_asset_name=batch_request.data_asset_name,
                batch_identifiers=IDDict(batch),
                batch_spec_passthrough=batch_request.batch_spec_passthrough,
            )
            for name, batch in asset.items()
            if kedro_context.catalog.exists(name)
        ]

    def _generate_batch_spec_parameters_from_batch_definition(
        self, batch_definition: BatchDefinition
    ) -> dict:
        return dict()


class KedroDataContext(DataContext):
    def __init__(
        self, context: KedroContext, dataset_to_datasource=dataset_to_datasource
    ):
        self._context = context
        self._dataset_to_datasource = dataset_to_datasource
        global kedro_context
        if not kedro_context:
            kedro_context = context
        super().__init__(context_root_dir=context.project_path)

    def _load_project_config(self):
        """
        Loads the project config from kedro template files
        """
        expectations_yaml = self._context.config_loader.get(
            "expectations*", "*expectations*", "**/*expectations*"
        )
        expectations_yaml = deepcopy(expectations_yaml)

        # Create a dataset filter to avoid loading invalid datasets such as params
        dataset_filter = expectations_yaml.pop("kedro_dataset_filter", [])
        dataset_filter.extend(["params:", "parameters"])

        # Change notebook dir to kedro one
        self.GE_UNCOMMITTED_DIR = expectations_yaml.pop("notebook_dir", "notebooks")
        self.GE_EDIT_NOTEBOOK_DIR = self.GE_UNCOMMITTED_DIR

        try:
            config = DataContextConfig.from_commented_map(
                commented_map=expectations_yaml
            )

            datasets = self._context.catalog.list()
            datasets = [
                name for name in datasets if not any(f in name for f in dataset_filter)
            ]
            global kedro_assets
            kedro_assets = dataset_to_datasource(datasets)

            datasources_spec = {
                name: {
                    "name": name,
                    "class_name": "Datasource",
                    "module_name": "great_expectations.datasource",
                    "execution_engine": {
                        "module_name": "great_expectations.execution_engine",
                        "class_name": "PandasExecutionEngine",
                    },
                    "data_connectors": {
                        "kedro": {
                            "class_name": "KedroDataConnector",
                            "module_name": "pasteur.pipelines.measure.expectations.data_context",
                            # "batch_identifiers": ["split", "is_decoded"],
                        },
                    },
                }
                for name in kedro_assets.keys()
            }

            config["datasources"] = {**config["datasources"], **datasources_spec}

            return config
        except ge_exceptions.InvalidDataContextConfigError:
            # Just to be explicit about what we intended to catch
            raise

    def _save_project_config(self):
        """Save the current project to expanded config file."""
        MOD_CONFIG = "expectations.yml.new"
        config_filepath = os.path.join(self._context.project_path, MOD_CONFIG)
        # Do not overwrite config file
        i = 1
        while os.path.exists(config_filepath):
            config_filepath = f"{config_filepath}-{i}"
            i += 1

        log.info(f"Saving modified config (expanded) to {config_filepath}")

        with open(config_filepath, "w") as outfile:
            self.config.to_yaml(outfile)
