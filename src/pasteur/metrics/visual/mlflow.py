from .holder import HistHolder, VizData
import mlflow


def mlflow_log_hists(holder: HistHolder, **data: VizData):
    if not mlflow.active_run():
        return
    path_prefix = "histograms/"
    if holder.table != "table":
        path_prefix += f"{holder.table}/"

    vizs = holder.visualise(data)

    for name, viz in vizs.items():
        name = name.lower()
        if isinstance(viz, dict):
            for i, (n, v) in enumerate(viz.items()):
                mlflow.log_figure(v, f"{path_prefix}{name}_{i}_{n}.png")
        else:
            mlflow.log_figure(viz, f"{path_prefix}{name}.png")