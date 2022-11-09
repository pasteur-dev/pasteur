from typing import Callable

def get_pasteur_modules() -> Callable | list[type] | None:
    """Grabs pasteur modules from kedro settings.
    
    If they're not defined, returns None. 
    If they're None, returns recommended modules.
    Otherwise, returns the value. """

    from kedro.framework.project import settings
    from ..extras import get_recommended_modules
    from dynaconf import Validator

    PASTEUR_MODULES = "PASTEUR_MODULES"
    sentinel = "_pasteur_disabled_xyz"

    for v in settings.validators:
        if PASTEUR_MODULES in v.names:
            break
    else:
        # Add validator for modules only if it doesn't exist
        settings.validators.register(
            Validator(PASTEUR_MODULES, default=sentinel),
        )
        settings.validators.validate()

    if settings.PASTEUR_MODULES == sentinel:
        return None
    if settings.PASTEUR_MODULES is None:
        return get_recommended_modules
    return settings.PASTEUR_MODULES
