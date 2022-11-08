def get_module_dict(parent: type, modules: list[type]):
    """Filters the list `modules` for modules which extend """
    out = {}
    for module in modules:
        if not module.issubclass(parent):
            continue

        assert hasattr(module, "name"), f"Module class attr {module.__name__}.name doesn't exist."
        assert isinstance(module.name, str) and module.name, f"{module.__name__} is not str or is empty."
        assert module.name not in out, "There are multiple modules of the same type with the same name."
        out[module.name] = module
    return out
