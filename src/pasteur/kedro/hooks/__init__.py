""" Pasteur ships with two hooks: one that dynamically creates the synthetic
data pipelines and datasets, and one that manages Mlflow runs.

The classes of those hooks are provided through `PasteurHook`, `MlflowTrackingHook`.

In addition, instances of those hooks are provided in `pasteur` and `mlflow`,
which are registered as Kedro entrypoints. """

from ..utils import get_pasteur_modules

modules = get_pasteur_modules()
if modules:
    from .pasteur import PasteurHook
    from ..mlflow import MlflowTrackingHook

    pasteur = PasteurHook(modules)
    mlflow = MlflowTrackingHook()

__all__ = ["PasteurHook", "MlflowTrackingHook", "mlflow", "pasteur"]