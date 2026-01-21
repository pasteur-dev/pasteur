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
    # We don't ship wheels, marginal calculation is compiled for the target
    # machine, for now with `-march=native`.
    ext_modules=[
        Extension(
            "pasteur.marginal.native",
            get_c_files("src/pasteur/marginal/native/"),
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native"],
            # For debugging:
            # extra_compile_args=["-O0", "-g", "-march=native"],
            # Run with AGENT=1 _MULTIPROCESS=0 gdb --args python -m pasteur...
            # Then, `run` and `bt full` on crash.
        )
    ],
)
