import numpy as np
from typing import Any, Literal
from copy import copy


def get_dtype(domain: int):
    # uint16 is 2x as fast as uint32 (5ms -> 3ms), use with marginals.
    # Marginal domain can not exceed max(uint16) size 65535 + 1
    if domain <= 1 << 8:
        return np.uint8
    if domain <= 1 << 16:
        return np.uint16
    return np.uint32


class Level:
    pass


class LeafLevel(Level, str):
    pass


class NodeLevel(Level, list[Level]):
    def __init__(self, type: Literal["cat", "ord"], arr: list[Level | Any]):
        lvls = []
        for a in arr:
            if isinstance(a, NodeLevel):
                lvls.append(a)
            else:
                lvls.append(LeafLevel(a))

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
        return max(lvl.height if isinstance(lvl, NodeLevel) else 0 for lvl in self) + 1

    @property
    def size(self) -> int:
        return sum(lvl.size if isinstance(lvl, NodeLevel) else 1 for lvl in self)

    def get_domain(self, height: int):
        return len(self.get_groups(height))

    def _get_groups_by_level(self, lvl: int, ofs: int = 0) -> list[list[int] | int]:
        groups = []
        for l in self:
            if isinstance(l, NodeLevel):
                g, ofs = l._get_groups_by_level(lvl - 1, ofs)

                if lvl == 0:
                    groups.append(g)
                else:
                    groups.extend(g)
            else:
                groups.append(ofs)
                ofs += 1
        return groups, ofs

    def get_groups(self, height: int) -> list[list[int] | int]:
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

    def get_mapping(self, height: int) -> np.array:
        domain = self.size
        a = np.ndarray((domain), dtype=get_dtype(domain))

        dmap = self.get_dict_mapping(height)
        for i, j in dmap.items():
            a[i] = j
        return a

    def get_human_values(self) -> list[str]:
        out = []
        for lvl in self:
            if isinstance(lvl, NodeLevel):
                out.extend(lvl.get_human_values())
            else:
                out.append(str(lvl))
        return out

    @staticmethod
    def from_str(
        a: str, nullable: bool = False, ukn_val: Any | None = None
    ) -> "NodeLevel":
        i = 0

        stack = [[]]
        is_ord = [False]
        bracket_closed = False

        if nullable:
            stack[-1].append(LeafLevel(None))
        if ukn_val is not None:
            stack[-1].append(LeafLevel(ukn_val))

        name = ""
        for j, c in enumerate(a):
            # Check brackets close correctly, after a bracket(s) closes a comma should follow
            if bracket_closed:
                assert (
                    c in "]},"
                ), f"',' should follow after a bracket closing (']', '}}'): {a[:j+1]}<"

            if c in "]}," and not bracket_closed:
                stack[-1].append(LeafLevel(name))
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
                    stack[-1].append(NodeLevel("cat", children))
                case "[":
                    stack.append([])
                    is_ord.append(True)
                case "]":
                    children = stack.pop()
                    assert is_ord.pop(), "Unmatched '{' bracket, found ']'"
                    stack[-1].append(NodeLevel("ord", children))

        lvl_attrs = stack[0]
        if len(lvl_attrs) == 1:
            lvl = lvl_attrs[0]
        else:
            lvl = NodeLevel("cat", lvl_attrs)

        return lvl


class Column:
    type: Literal["idx", "num"]
    name: str | None = None
    na: bool = False

    # Column type specific hints
    lvl: NodeLevel
    bins: int | None
    max: float | int | None
    max: float | int | None


class IdxColumn:
    type = "idx"

    def __init__(self, lvl: NodeLevel) -> None:
        self.lvl = lvl

    def __str__(self) -> str:
        return "Idx" + str(self.lvl)

    def __repr__(self) -> str:
        return "Idx" + repr(self.lvl)


class CatColumn(IdxColumn):
    def __init__(self, vals, na: bool = False, ukn_val: Any | None = None):
        arr = []
        if na:
            arr.append(None)
        if ukn_val is not None:
            arr.append(ukn_val)
        arr.extend(vals)

        super().__init__(NodeLevel("cat", arr))


class OrdColumn(IdxColumn):
    def __init__(self, vals, na: bool = False, ukn_val: Any | None = None):
        lvl = NodeLevel("ord", vals)

        if na or ukn_val is not None:
            arr = []
            if na:
                arr.append(None)
            if ukn_val is not None:
                arr.append(ukn_val)
            arr.append(lvl)

            lvl = NodeLevel("cat", arr)

        super().__init__(lvl)


class NumColumn(Column):
    type = "num"

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


class Attribute:
    def __init__(
        self,
        name: str,
        cols: dict[str, Column],
        na: bool = False,
        ukn_val: bool = False,
    ) -> None:
        self.name = name
        self.na = na
        self.ukn_val = ukn_val
        self.common = self.na + self.ukn_val

        self.update_cols(cols)

    def update_cols(self, cols: dict[str, Column]):
        self.cols: dict[str, Column] = {}
        for name, col in cols.items():
            col = copy(col)
            col.na = self.na
            col.name = name
            self.cols[name] = col

    def __str__(self) -> str:
        return f"Attr[na={int(self.na)},ukn={int(self.ukn_val)}]{self.cols}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, col: str) -> Column:
        return self.cols[col]


Attributes = dict[str, Attribute]


class OrdAttribute(Attribute):
    def __init__(
        self, name: str, vals: list[Any], na: bool = False, ukn_val: Any | None = None
    ) -> None:
        cols = {name: OrdColumn(vals, na, ukn_val)}

        super().__init__(name, cols, na, ukn_val is not None)


class CatAttribute(Attribute):
    def __init__(
        self, name: str, vals: list[Any], na: bool = False, ukn_val: Any | None = None
    ) -> None:
        cols = {name: CatColumn(vals, na, ukn_val)}

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
        super().__init__(name, {name: NumColumn(bins, min, max)}, nullable, False)
