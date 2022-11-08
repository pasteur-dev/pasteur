from ...utils.mlflow import (
    gen_html_figure_container,
    load_matplotlib_style,
    mlflow_log_artifacts,
)
from .holder import HistHolder, VizData

_SAVE_HTML = True


def mlflow_log_hists(holder: HistHolder, log_artifacts: bool = True, **data: VizData):
    import matplotlib.pyplot as plt
    import mlflow

    if not mlflow.active_run():
        return

    if log_artifacts:
        mlflow_log_artifacts("visual", holder=holder, **data)

    path_prefix = "histograms/"
    if holder.table != "table":
        path_prefix += f"{holder.table}/"

    load_matplotlib_style()
    vizs = holder.visualise(data)

    for name, viz in vizs.items():
        name = name.lower()
        if isinstance(viz, dict):
            if _SAVE_HTML:
                html = gen_html_figure_container(viz)
                mlflow.log_text(html, f"{path_prefix}{name}.html")
            else:
                for i, (n, v) in enumerate(viz.items()):
                    mlflow.log_figure(v, f"{path_prefix}{name}_{i}_{n}.png")

            for v in viz.values():
                plt.close(v)
        else:
            mlflow.log_figure(viz, f"{path_prefix}{name}.png")
            plt.close(viz)
