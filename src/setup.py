import os

import numpy as np
from setuptools import Extension, find_packages, setup


def get_c_files(dr: str):
    return [
        os.path.join(dr, fn)
        for fn in os.listdir(dr)
        if fn.endswith(".c") or fn.endswith(".h")
    ]


entry_points = {
    "console_scripts": ["pasteur = pasteur.__main__:main"],
    "kedro.hooks": [
        "pasteur = pasteur.kedro.hooks:pasteur",
        "pasteur_mlflow = pasteur.kedro.hooks:mlflow",
    ],
    "kedro.project_commands": ["pasteur = pasteur.cli:cli"],
}

# get the dependencies and installs
with open("requirements.txt", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="pasteur",
    version="0.1",
    packages=find_packages(exclude=["tests", "project"]),
    entry_points=entry_points,
    install_requires=requires,
    extras_require={
        "docs": [
            "docutils<0.18.0",
            "sphinx~=3.4.3",
            "sphinx_rtd_theme==0.5.1",
            "nbsphinx==0.8.1",
            "recommonmark==0.7.1",
            "sphinx-autodoc-typehints==1.11.1",
            "sphinx_copybutton==0.3.1",
            "ipykernel>=5.3, <7.0",
            "Jinja2<3.1.0",
        ]
    },
    ext_modules=[
        Extension(
            "pasteur.marginal.native",
            get_c_files("pasteur/marginal/native/"),
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
)
