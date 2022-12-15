import numpy as np
from typing import Any, Literal, TypeVar
from copy import copy


def get_dtype(domain: int):
    # uint16 is 2x as fast as uint32 (5ms -> 3ms), use with marginals.
    # Marginal domain can not exceed max(uint16) size 65535 + 1
    if domain <= 1 << 8:
        return np.uint8
    if domain <= 1 << 16:
        return np.uint16
    if domain <= 1 << 32:
        return np.uint32
    return np.uint64


LI = TypeVar("LI", "Level", int)


class Level(list[LI]):
    def __init__(self, type: Literal["cat", "ord"], arr: list["Level" | Any]):
        lvls = []
        for a in arr:
            if isinstance(a, Level):
                lvls.append(a)
            else:
                lvls.append(str(a))

        super().__init__(lvls)
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
    def height(self) -> int:
        if not self:
            return 0
        return max(lvl.height if isinstance(lvl, Level) else 0 for lvl in self) + 1

    @property
    def size(self) -> int:
        return sum(lvl.size if isinstance(lvl, Level) else 1 for lvl in self)

    def get_domain(self, height: int):
        return len(self.get_groups(height))

    def _get_groups_by_level(self, lvl: int, ofs: int = 0):
        groups: list[list | int] = []
        for l in self:
            if isinstance(l, Level):
                g, ofs = l._get_groups_by_level(lvl - 1, ofs)

                if lvl == 0:
                    groups.append(g)
                else:
                    groups.extend(g)
            else:
                groups.append(ofs)
                ofs += 1
        return groups, ofs

    def get_groups(self, height: int) -> list[list | int]:
        return self._get_groups_by_level(self.height - 1 - height)[0]

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

    def get_mapping(self, height: int) -> np.ndarray:
        domain = self.size
        a = np.ndarray((domain), dtype=get_dtype(domain))

        dmap = self.get_dict_mapping(height)
        for i, j in dmap.items():
            a[i] = j
        return a

    def get_human_values(self) -> list[str]:
        out = []
        for lvl in self:
            if isinstance(lvl, Level):
                out.extend(lvl.get_human_values())
            else:
                out.append(str(lvl))
        return out

    @staticmethod
    def from_str(a: str, nullable: bool = False, ukn_val: Any | None = None) -> "Level":
        stack = [[]]
        is_ord = [False]
        bracket_closed = False

        if nullable:
            stack[-1].append(str(None))
        if ukn_val is not None:
            stack[-1].append(str(ukn_val))

        name = ""
        for j, c in enumerate(a):
            # Check brackets close correctly, after a bracket(s) closes a comma should follow
            if bracket_closed:
                assert (
                    c in "]},"
                ), f"',' should follow after a bracket closing (']', '}}'): {a[:j+1]}<"

            if c in "]}," and not bracket_closed:
                stack[-1].append(str(name))
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

        return lvl


class Value:
    name: str | None = None
    common: int = 0


class IdxValue(Value):
    def get_domain(self, height: int = 0) -> int:
        """Returns the domain of the attribute in the given height."""
        raise NotImplementedError()

    def get_mapping(self, height: int) -> np.ndarray:
        """Returns a numpy array that associates discrete values with groups at
        the given height."""
        raise NotImplementedError()

    @property
    def height(self) -> int:
        """Returns the maximum height of this column."""
        return 0

    @property
    def domain(self):
        return self.get_domain(0)

    def is_ordinal(self) -> bool:
        """Returns whether this column is ordinal, other than for the elements
        it shares in common with the other attributes."""
        return False

    def downsample(self, column: np.ndarray, height: int):
        if height == 0:
            return column
        return self.get_mapping(height)[column]

    def upsample(self, column: np.ndarray, height: int, deterministic: bool = True):
        if height == 0:
            return column

        assert (
            deterministic
        ), "Current column doesn't contain a histogram, can't upsample"

        d = self.get_domain(height)
        mapping = self.get_mapping(height)

        # create reverse mapping
        reverse_map = np.empty((d,), dtype=get_dtype(self.get_domain(0)))
        for i in range(d):
            c = (mapping == i).argmax()
            reverse_map[i] = c

        return reverse_map[column]

    def select_height(self) -> int:
        return 0


class LevelValue(IdxValue):
    """A specific type of IdxColumn, which contains a hierarchical attribute
    structure based on a tree."""

    def __init__(self, lvl: Level, common: int = 0) -> None:
        self.head = lvl
        self.common = common

    def __str__(self) -> str:
        return "Idx" + str(self.head)

    def __repr__(self) -> str:
        return "Idx" + repr(self.head)

    def get_domain(self, height: int):
        return self.head.get_domain(height)

    def get_mapping(self, height: int):
        return self.head.get_mapping(height)

    def is_ordinal(self) -> bool:
        if self.head.type == "ord" and self.head.size == len(self.head):
            return True
        if (
            self.head.type == "cat"
            and len(self.head) == self.common + 1
            and self.head[self.common].type == "ord"
        ):
            return True
        return False

    @property
    def height(self):
        return self.head.height


class CatValue(LevelValue):
    """Initializer for LevelColumn, which initializes a single level Categorical column."""

    def __init__(self, vals, na: bool = False, ukn_val: Any | None = None):
        arr = []
        common = 0
        if na:
            arr.append(None)
            common += 1
        if ukn_val is not None:
            arr.append(ukn_val)
            common += 1
        arr.extend(vals)

        super().__init__(Level("cat", arr))


class OrdValue(LevelValue):
    """Initializer for LevelColumn, which initializes a single level Ordinal column, which might have common values."""

    def __init__(self, vals, na: bool = False, ukn_val: Any | None = None):
        lvl = Level("ord", vals)
        common = 0

        if na or ukn_val is not None:
            arr = []
            if na:
                common += 1
                arr.append(None)
            if ukn_val is not None:
                common += 1
                arr.append(ukn_val)
            arr.append(lvl)

            lvl = Level("cat", arr)

        super().__init__(lvl, common)


class NumValue(Value):
    """Numerical Column, its value can be represented with a number, which might be NaN.

    TODO: handle multiple common values (1 is assumed to be NA), appropriately."""

    def __init__(
        self,
        bins: int | None = None,
        min: int | float | None = None,
        max: int | float | None = None,
    ) -> None:
        self.bins = bins
        self.min = min
        self.max = max

    def __str__(self) -> str:
        return f"Num[{self.bins if self.bins is not None else 'NA'},{self.min if self.min else float('nan'):.2f}<x<{self.max if self.max else float('nan'):.2f}]"

    def __repr__(self) -> str:
        return str(self)


V = TypeVar("V", bound=Value)


class Attribute:
    def __init__(
        self,
        name: str,
        vals: dict[str, V],
        na: bool = False,
        ukn_val: bool = False,
    ) -> None:
        self.name = name
        self.na = na
        self.ukn_val = ukn_val
        self.common = self.na + self.ukn_val

        self.update_vals(vals)

    def update_vals(self, vals: dict[str, V]):
        self.vals: dict[str, Value] = {}
        for name, val in vals.items():
            val = copy(val)
            val.name = name
            val.common = self.common
            self.vals[name] = val

    def __str__(self) -> str:
        return f"Attr[na={int(self.na)},ukn={int(self.ukn_val)}]{self.vals}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, col: str) -> Value:
        return self.vals[col]


Attributes = dict[str, Attribute]


class OrdAttribute(Attribute):
    def __init__(
        self, name: str, vals: list[Any], na: bool = False, ukn_val: Any | None = None
    ) -> None:
        cols = {name: OrdValue(vals, na, ukn_val)}

        super().__init__(name, cols, na, ukn_val is not None)


class CatAttribute(Attribute):
    def __init__(
        self, name: str, vals: list[Any], na: bool = False, ukn_val: Any | None = None
    ) -> None:
        cols = {name: OrdValue(vals, na, ukn_val)}

        super().__init__(name, cols, na, ukn_val is not None)


class NumAttribute(Attribute):
    def __init__(
        self,
        name: str,
        bins: int,
        min: int | float | None,
        max: int | float | None,
        nullable: bool = False,
    ) -> None:
        super().__init__(name, {name: NumValue(bins, min, max)}, nullable, False)
