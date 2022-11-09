from .kedro.utils import get_pasteur_modules

if get_pasteur_modules():
    from .kedro.cli import cli
else:
    cli = None

__all__ = ["cli"]