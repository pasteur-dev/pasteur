from math import isnan
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

import mlflow
from kedro.extras.datasets.pandas.csv_dataset import CSVDataSet
from mlflow.tracking import MlflowClient

# from kedro_mlflow.io.artifacts.mlflow_artifact_dataset import MlflowArtifactDataSet


class MlflowSDMetricsDataset(CSVDataSet):
    """Wraps a DataFrame SDMetrics result as a CSV artifact and logs the results as metrics"""

    def __init__(self, prefix, local_path, artifact_path):
        super().__init__(filepath=local_path)
        self.run_id = None
        self.prefix = prefix
        self.artifact_path = artifact_path
        self._logging_activated = True

    @property
    def _logging_activated(self):
        return self.__logging_activated

    @_logging_activated.setter
    def _logging_activated(self, flag):
        if not isinstance(flag, bool):
            raise ValueError(f"_logging_activated must be a boolean, got {type(flag)}")
        self.__logging_activated = flag

    def _save(self, data: pd.DataFrame):
        # Log metrics
        for _, row in data.iterrows():
            metric = row["metric"]
            score = row["raw_score"]
            if score and not isnan(score):
                mlflow.log_metric(f"{self.prefix}.{metric}", score)

        # _get_save_path needs to be called before super, otherwise
        # it will throw exception that file under path already exist.
        if hasattr(self, "_version"):
            # all kedro datasets inherits from AbstractVersionedDataSet
            local_path = self._get_save_path()
        elif hasattr(self, "_filepath"):
            # in case custom datasets inherits from AbstractDataSet without versioning
            local_path = self._filepath  # pragma: no cover
        elif hasattr(self, "_path"):
            # special datasets with a folder instead of a specifi files like PartitionedDataSet
            local_path = Path(self._path)

        # it must be converted to a string with as_posix()
        # for logging on remote storage like Azure S3
        local_path = local_path.as_posix()

        super()._save(data)

        if self._logging_activated:
            if self.run_id:
                # if a run id is specified, we have to use mlflow client
                # to avoid potential conflicts with an already active run
                mlflow_client = MlflowClient()
                mlflow_client.log_artifact(
                    run_id=self.run_id,
                    local_path=local_path,
                    artifact_path=self.artifact_path,
                )
            else:
                mlflow.log_artifact(local_path, self.artifact_path)

    def _load(self) -> Any:  # pragma: no cover
        if self.run_id:
            # if no run_id is specified, we take the artifact from the local path rather that the active run:
            # there are a lot of chances that it has not been saved yet!

            mlflow_client = MlflowClient()

            if hasattr(self, "_version"):
                # all kedro datasets inherits from AbstractVersionedDataSet
                local_path = self._get_load_path()
            elif hasattr(self, "_filepath"):
                # in case custom datasets inherits from AbstractDataSet without versioning
                local_path = self._filepath  # pragma: no cover
            elif hasattr(self, "_path"):
                # special datasets with a folder instead of a specifi files like PartitionedDataSet
                local_path = Path(self._path)

            artifact_path = (
                (self.artifact_path / local_path.name).as_posix()
                if self.artifact_path
                else local_path.name
            )

            mlflow_client.download_artifacts(
                run_id=self.run_id,
                path=artifact_path,
                dst_path=local_path.parent.as_posix(),  # must be a **local** **directory**
            )

        # finally, read locally
        return super()._load()
