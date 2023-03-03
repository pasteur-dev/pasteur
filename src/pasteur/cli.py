from .kedro.utils import get_pasteur_modules

if get_pasteur_modules():
    from .kedro.cli import cli
else:
    cli = None

    import logging

    logger = logging.getLogger(__name__)
    logger.warn(
        "Pasteur project not found in the current directory "
        + "(settings.py file doesn't contain `PASTEUR_MODULES = ...`). "
        + "Disabling Pasteur commands."
    )

__all__ = ["cli"]
