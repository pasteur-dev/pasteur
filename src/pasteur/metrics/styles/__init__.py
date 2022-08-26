from os import path

import matplotlib.pyplot as plt


def use_style(name: str):
    dirname = path.dirname(__file__)
    mplstyle = path.join(dirname, f"{name}.mplstyle")
    assert path.isfile(mplstyle), f"Style {mplstyle} not found."
    plt.style.use(mplstyle)


__all__ = ["use_style"]
