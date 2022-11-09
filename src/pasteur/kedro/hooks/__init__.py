from ..utils import get_pasteur_modules

modules = get_pasteur_modules()
if modules:
    from .pasteur import PasteurHook
    from ..mlflow import MlflowTrackingHook
    hooks = [PasteurHook(modules), MlflowTrackingHook()]
else:
    hooks = []

__all__ = ["PasteurHook", "MlflowTrackingHook", "hooks"]