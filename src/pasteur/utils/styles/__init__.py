""" Simple style for matplotlib that looks good in mlflow. """

from os import path


def use_style(name: str):
    import matplotlib.pyplot as plt

    dirname = path.dirname(__file__)
    mplstyle = path.join(dirname, f"{name}.mplstyle")
    assert path.isfile(mplstyle), f"Style {mplstyle} not found."
    plt.style.use(mplstyle)


__all__ = ["use_style"]
