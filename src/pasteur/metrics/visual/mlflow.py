from .holder import HistHolder, VizData
from ..mlflow import gen_html_figure_container, load_matplotlib_style

_SAVE_HTML = True


def mlflow_log_hists(holder: HistHolder, **data: VizData):
    import mlflow
    import matplotlib.pyplot as plt

    if not mlflow.active_run():
        return
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