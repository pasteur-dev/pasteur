from typing import TypeVar, Generic


class Module:
    """A Pasteur module extends a base Module class (such as Synth) and defines a name.

    Each base class, name combination registered in the system is considered unique.
    Example: there should only be one registered Dataset named "adult".
    """

    name: str


class ModuleClass:
    """Modules which can not be instantiated extend from ModuleClass and define a Factory to act as their module"""

    name: str
    _factory: type["ModuleFactory"]

    def __init__(self, *args, _from_factory: bool = False, **kwargs) -> None:
        assert (
            _from_factory
        ), "Module should not be instantiated, call `cls.get_factory()`"

    @classmethod
    def get_factory(cls, *args, **kwargs):
        return cls._factory(cls, *args, **kwargs)


A = TypeVar("A", bound=ModuleClass)


class ModuleFactory(Module, Generic[A]):
    """Some modules (such as transformers) require multiple instances in the system. In this case,
    it's not possible to provide a module instance for them.

    `BaseFactory` is used to provide a wrapper instance to that module class."""

    def __init__(self, cls: type[A], *args, name: str | None = None, **_) -> None:
        self._cls = cls
        self.name = name or cls.name

    def build(self, *args, **kwargs):
        """Build is used to create the new instance. You can override this
        function to customize instance creation."""
        return self._cls(*args, _from_factory=True, **kwargs)


M = TypeVar("M", bound=Module)


def get_module_dict(parent: type[M], modules: list[Module]) -> dict[str, M]:
    """Filters the list `modules` for modules which extend"""
    out = {}
    for module in modules:
        if isinstance(module, parent):
            out[module.name] = module
    return out


def instantiate_dict(d: dict[str, type[A]]) -> dict[str, A]:
    return {k: v() for k, v in d.items()}
