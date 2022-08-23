from .holder import HistHolder, VizData
import mlflow


def mlflow_log_hists(holder: HistHolder, **data: VizData):
    if not mlflow.active_run():
        return

    vizs = holder.visualise(data)

    for name, viz in vizs.items():
        if isinstance(viz, list):
            for i, v in enumerate(viz):
                mlflow.log_figure(v, f"metrics/hist/{holder.table}/{name}_{i}.png")
        else:
            mlflow.log_figure(viz, f"metrics/hist/{holder.table}/{name}.png")
