import os

import numpy as np
from setuptools import Extension, setup


def get_c_files(dr: str):
    return [
        os.path.join(dr, fn)
        for fn in os.listdir(dr)
        if fn.endswith(".c") or fn.endswith(".h")
    ]

setup(
    ext_modules=[
        Extension(
            "pasteur.marginal.native",
            get_c_files("src/pasteur/marginal/native/"),
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
)
