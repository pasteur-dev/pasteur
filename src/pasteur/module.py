from collections import defaultdict
from typing import Generic, TypeVar


class Module:
    """A Pasteur module extends a base Module class (such as Datset) and defines a name.

    Each base class, name combination registered in the system is considered unique*.
    Example: there should only be one registered Dataset named "adult".

    *the exception to this is metrics, where the name corresponds to a column type
    and there can be multiple visualizations for a certain column type.
    """

    name: str


class ModuleClass:
    """Modules which need to be instantiated multiple times extend from ModuleClass and define
    a Factory to act as their module"""

    name: str
    _factory: type["ModuleFactory"]

    def __init__(self, *args, _from_factory: bool = False, **kwargs) -> None:
        assert (
            _from_factory
        ), "Module should not be instantiated, call `cls.get_factory()`"

    @classmethod
    def get_factory(cls, *args, **kwargs):
        """Returns a factory that registers this module to the system.

        Any `*args` and `**kwargs` passed to this function will be saved and passed
        to the module's `__init__()` method when calling `build()`."""
        return cls._factory(cls, *args, **kwargs)


A = TypeVar("A", bound=ModuleClass)


class ModuleFactory(Module, Generic[A]):
    """Some modules (such as transformers) require multiple instances in the system. In this case,
    it's not possible to provide a module instance for them.

    `ModuleFactory` is used to provide a wrapper instance to that module class."""

    def __init__(self, cls: type[A], *args, name: str | None = None, **kwargs) -> None:
        self._cls = cls
        self.name = name or cls.name
        self.args = args
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        """Build is used to create the new instance. You can override this
        function to customize instance creation."""
        return self._cls(*args, *self.args, _from_factory=True, **kwargs, **self.kwargs)


M = TypeVar("M", bound=Module)


def get_module_dict(parent: type[M], modules: list[Module]) -> dict[str, M]:
    """Filters the list `modules` for modules which extend"""
    out = {}
    for module in modules:
        if isinstance(module, parent):
            assert (
                module.name not in out
            ), f'Module name "{module.name}" registered twice.'
            out[module.name] = module
    return out


def get_module_dict_multiple(
    parent: type[M], modules: list[Module]
) -> dict[str, list[M]]:
    """Same as `get_module_dict`, except for that it returns a list.
    Multiple modules can use the same name."""
    out = defaultdict(list)
    for module in modules:
        if isinstance(module, parent):
            out[module.name].append(module)
    return out
