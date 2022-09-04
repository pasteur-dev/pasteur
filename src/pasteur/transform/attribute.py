import numpy as np
from typing import Any, NamedTuple, Literal

IDX_DTYPES = np.uint8 | np.uint16 | np.uint32


def get_type(domain: int):
    if domain <= 1 << 8:
        return np.uint8
    if domain <= 1 << 16:
        return np.uint16
    return np.uint32


class Level(list["Level | IDX_DTYPES"]):
    def __init__(self, type: Literal["cat", "ord"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type

    def __str__(self) -> str:
        base = super().__str__()
        if self.type == "cat":
            return "{" + base[1:-1] + "}"
        return base

    def __repr__(self) -> str:
        base = super().__repr__()
        if self.type == "cat":
            return "{" + base[1:-1] + "}"
        return base

    @property
    def max_height(self) -> int:
        return max(lvl.max_height + 1 if isinstance(lvl, Level) else 0 for lvl in self)

    def _get_groups_by_level(self, lvl: int) -> list[list[int]]:
        groups = []
        for l in self:
            if isinstance(l, Level):
                g = l._get_groups_by_level(lvl - 1)

                if lvl == 0:
                    groups.append(g)
                else:
                    groups.extend(g)
            else:
                groups.append(l)
        return groups

    def _get_max_n(self):
        return max(lvl._get_max_n() if isinstance(lvl, Level) else lvl for lvl in self)

    def get_groups(self, height: int) -> list[list[int]]:
        return self._get_groups_by_level(self.max_height - height)

    def get_dict_mapping(self, height: int) -> dict[int, int]:
        groups = self.get_groups(height)
        mapping = {}
        for i, g in enumerate(groups):
            if isinstance(g, list):
                for j in g:
                    mapping[j] = i
            else:
                mapping[g] = i

        return mapping

    def get_mapping(self, height: int) -> np.array:
        domain = self._get_max_n() + 1
        a = np.ndarray((domain), dtype=get_type(domain))

        dmap = self.get_dict_mapping(height)
        for i, j in dmap.items():
            a[i] = j
        return a

    @staticmethod
    def from_str(
        a: str, nullable: bool = False, ukn_val: Any | None = None
    ) -> tuple["Level", list[str | int]]:
        i = 0
        names = []
        name = ""

        stack = [[]]
        is_ord = [False]
        bracket_closed = False

        if nullable:
            stack[-1].append(i)
            i += 1
            names.append(None)
        if ukn_val is not None:
            stack[-1].append(i)
            i += 1
            names.append(ukn_val)

        for j, c in enumerate(a):
            # Check brackets close correctly, after a bracket(s) closes a comma should follow
            if bracket_closed:
                assert (
                    c in "]},"
                ), f"',' should follow after a bracket closing (']', '}}'): {a[:j+1]}<"

            if c in "]}," and not bracket_closed:
                stack[-1].append(i)
                i += 1

                # Try to convert to int
                try:
                    name = int(name)
                except:
                    pass

                names.append(name)
                name = ""

            if c not in "[]{},":
                name += c

            if c in ["]", "}"]:
                bracket_closed = True
            elif c == ",":
                bracket_closed = False

            match c:
                case "{":
                    stack.append([])
                    is_ord.append(False)
                case "}":
                    children = stack.pop()
                    assert not is_ord.pop(), "Unmatched '[' bracket, found '}'"
                    stack[-1].append(Level("cat", children))
                case "[":
                    stack.append([])
                    is_ord.append(True)
                case "]":
                    children = stack.pop()
                    assert is_ord.pop(), "Unmatched '{' bracket, found ']'"
                    stack[-1].append(Level("ord", children))

        lvl_attrs = stack[0]
        if len(lvl_attrs) == 1:
            lvl = lvl_attrs[0]
        else:
            lvl = Level("cat", lvl_attrs)

        return lvl, names


class OrdLevel(Level):
    def __init__(self, *args, **kwargs):
        super().__init__("ord", *args, **kwargs)


class CatLevel(Level):
    def __init__(self, *args, **kwargs):
        super().__init__("cat", *args, **kwargs)


class Column(NamedTuple):
    name: str
    type: Literal["idx", "num"]
    na: bool
    lvl: Level | None = None


class Attribute(NamedTuple):
    name: str
    cols: dict[str, Column]

    @property
    def na(self):
        if len(self.cols) == 0:
            return False

        na = next(iter(self.cols.values()))
        for col in self.cols.values():
            assert col.na == na, f"Attribute {self.name} has mixed nullability values."

        return na


class Attributes(dict[str, Attribute]):
    pass
