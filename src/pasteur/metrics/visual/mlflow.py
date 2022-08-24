from io import BytesIO
import mlflow
from matplotlib.figure import Figure

from .holder import HistHolder, VizData

_SAVE_HTML = True


def _gen_html(viz: dict[str, Figure]):
    import base64

    style = """
    <style>
        .flex {
            display: flex;
            flex-wrap: wrap;
        }
    </style>
    """

    header = """
    <div class="flex">
    """

    footer = """
    </div>
    """

    img_html = (
        lambda name, x: '<img class="'
        + name
        + '" src="data:image/png;base64,'
        + x
        + '">'
    )

    imgs = []
    for name, fig in viz.items():
        with BytesIO() as buff:
            fig.savefig(buff, format="png")

            buff.seek(0)
            bytes = buff.read()

        bytes_base64 = base64.b64encode(bytes)
        enc = bytes_base64.decode()

        img = img_html(name, enc)
        imgs.append(img)

    return (
        "<html><head>"
        + style
        + "</head><body>"
        + header
        + "\n".join(imgs)
        + footer
        + "</body></html>"
    )


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
            if _SAVE_HTML:
                html = _gen_html(viz)
                mlflow.log_text(html, f"{path_prefix}{name}.html")
            else:
                for i, (n, v) in enumerate(viz.items()):
                    mlflow.log_figure(v, f"{path_prefix}{name}_{i}_{n}.png")
        else:
            mlflow.log_figure(viz, f"{path_prefix}{name}.png")
