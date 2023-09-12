"""This module implements the base abstractions of Attribute and Value, 
which are used to encapsulate the information of complex types.

Values are separated into CatValues (Categorical; defined by an index) and 
NumValues (Numerical). In addition, we define StratifiedValue as a special
CatValue that holds a Stratification.

The Stratification represents a tree, created through Grouping nodes.
Groupings are essentially lists with special functions.
We separate groupings based on whether they are categorical (child order irrelevant)
or ordinal (nearby children are more similar to each other).

The leafs of the tree represent the values the Value can take.
They take the form of strings that describe each value, but are essentially placeholders.
The Value is an integer. To map leafs to integers, the Tree is searched with Depth
First Search, respecting child order to be deterministic.

An Attribute holds multiple values and a set of common conditions. When a common
condition is active, all of the Attribute's Values are expected to have the same
value."""

from copy import copy
from typing import Any, Literal, Mapping, TypeVar

import numpy as np


def get_dtype(domain: int):
    """Returns the smallest NumPy unsigned integer dtype that will fit integers
    up to `domain` - 1."""

    # uint16 is 2x as fast as uint32 (5ms -> 3ms), use with marginals.
    # Marginal domain can not exceed max(uint16) size 65535 + 1
    if domain <= 1 << 8:
        return np.uint8
    if domain <= 1 << 16:
        return np.uint16
    if domain <= 1 << 32:
        return np.uint32
    return np.uint64


GI = TypeVar("GI", "Grouping", int)


class Grouping(list[GI]):
    """An enchanced form of list that holds the type of grouping (categorical, ordinal),
    and implements helper functions and an enchanced string representation."""

    def __init__(
        self,
        type: Literal["cat", "ord"],
        arr: list["Grouping | Any"],
        title: str | None = None,
    ):
        lvls = []
        for a in arr:
            if isinstance(a, Grouping):
                lvls.append(a)
            else:
                lvls.append(str(a))

        self.pref = f"`{title}`" if title else ""

        super().__init__(lvls)
        self.type = type

    def __repr__(self) -> str:
        base = super().__repr__()
        if self.type == "cat":
            return self.pref + "{" + base[1:-1] + "}"
        return self.pref + base

    @property
    def height(self) -> int:
        if not self:
            return 0
        return max(g.height if isinstance(g, Grouping) else 0 for g in self) + 1

    @property
    def size(self) -> int:
        return sum(g.size if isinstance(g, Grouping) else 1 for g in self)

    def get_domain(self, height: int):
        return len(self.get_groups(height))

    def _get_groups_by_level(self, lvl: int, ofs: int = 0):
        groups: list[list | int] = []
        for l in self:
            if isinstance(l, Grouping):
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
            if isinstance(lvl, Grouping):
                out.extend(lvl.get_human_values())
            else:
                out.append(str(lvl))
        return out

    @staticmethod
    def from_str(
        a: str, nullable: bool = False, ukn_val: Any | None = None
    ) -> "Grouping":
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
                    stack[-1].append(Grouping("cat", children))
                case "[":
                    stack.append([])
                    is_ord.append(True)
                case "]":
                    children = stack.pop()
                    assert is_ord.pop(), "Unmatched '{' bracket, found ']'"
                    stack[-1].append(Grouping("ord", children))

        lvl_attrs = stack[0]
        if len(lvl_attrs) == 1:
            lvl = lvl_attrs[0]
        else:
            lvl = Grouping("cat", lvl_attrs)

        return lvl


class Value:
    """Base value class"""

    name: str
    common: int = 0


class SeqValue(Value):
    table: str

    def __init__(self, name: str, table: str) -> None:
        self.name = name
        self.table = table


class CatValue(Value):
    """Class for a Categorical Value.

    Each Categorical Value is represented by an unsigned integer.
    It can also group its different values together based on an integer parameter
    named height.
    The implementation of this class remains abstract, and is expanded in
    the StratifiedValue class."""

    def get_domain(self, height: int = 0) -> int:
        """Returns the domain of the attribute in the given height."""
        raise NotImplementedError()

    def get_mapping(self, height: int) -> np.ndarray:
        """Returns a numpy array that associates discrete values with groups at
        the given height."""
        raise NotImplementedError()

    @property
    def height(self) -> int:
        """Returns the maximum height of this value."""
        return 0

    @property
    def domain(self):
        return self.get_domain(0)

    def is_ordinal(self) -> bool:
        """Returns whether this value is ordinal, other than for the elements
        it shares in common with the other attributes."""
        return False

    def downsample(self, value: np.ndarray, height: int):
        """Receives an array named `value` and downsamples it based on the provided
        height, by grouping certain values together. The proper implementation
        is provided by pasteur.hierarchy."""
        if height == 0:
            return value
        return self.get_mapping(height)[value]

    def upsample(self, value: np.ndarray, height: int, deterministic: bool = True):
        """Does the opposite of downsample. If deterministic is True, for each
        group at a given height one of its values is chosen arbitrarily to represent
        all children of the group.

        If deterministic is False, the group is sampled based on this Value's
        histogram (not implemented in this class; see pasteur.hierarchy)."""
        if height == 0:
            return value

        assert (
            deterministic
        ), "Current value doesn't contain a histogram, can't upsample"

        d = self.get_domain(height)
        mapping = self.get_mapping(height)

        # create reverse mapping
        reverse_map = np.empty((d,), dtype=get_dtype(self.get_domain(0)))
        for i in range(d):
            c = (mapping == i).argmax()
            reverse_map[i] = c

        return reverse_map[value]

    def select_height(self) -> int:
        return 0


IdxValue = CatValue


class StratifiedValue(CatValue):
    """A version of CategoricalValue which uses a Stratification to represent
    the domain knowledge of the Value.

    Each unique value is mapped to a tree
    with nodes where the child order matters.
    By traversing the tree in DFS, each leaf is mapped to an integer."""

    def __init__(self, head: Grouping, common: int = 0) -> None:
        self.head = head
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


class GenerationValue(StratifiedValue):
    table: str
    max_len: int

    def __init__(self, table: str, max_len: int) -> None:
        self.table = table
        self.max_len = max_len
        super().__init__(Grouping("ord", list(range(max_len + 1))), 0)


def _create_strat_value_cat(vals, na: bool = False, ukn_val: Any | None = None):
    arr = []
    common = 0
    if na:
        arr.append(None)
        common += 1
    if ukn_val is not None:
        arr.append(ukn_val)
        common += 1
    arr.extend(vals)

    return StratifiedValue(Grouping("cat", arr))


def _create_strat_value_ord(vals, na: bool = False, ukn_val: Any | None = None):
    g = Grouping("ord", vals)
    common = 0

    if na or ukn_val is not None:
        arr = []
        if na:
            common += 1
            arr.append(None)
        if ukn_val is not None:
            common += 1
            arr.append(ukn_val)
        arr.append(g)

        g = Grouping("cat", arr)

    return StratifiedValue(g, common)


class NumValue(Value):
    """Numerical Value: its value can be represented with a number, which might be NaN.

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
    """Attribute class which holds multiple values in a dictionary."""

    def __init__(
        self,
        name: str,
        vals: dict[str, V],
        na: bool = False,
        ukn_val: bool = False,
        common: str | int | None = None,
        unroll: bool = False,
        unroll_with: tuple[str, ...] = tuple(),
        partition: bool = False,
        partition_with: tuple[str, ...] = tuple(),
    ) -> None:
        self.name = name
        self.na = na
        self.ukn_val = ukn_val
        if common is None:
            self.common = self.na + self.ukn_val
        else:
            self.common = common

        self.unroll = unroll
        self.unroll_with = unroll_with

        self.partition = partition
        self.partition_with = partition_with

        self.update_vals(vals)

    def update_vals(self, vals: dict[str, V]):
        self.vals: dict[str, Value] = {}
        for name, val in vals.items():
            val = copy(val)
            val.name = name
            val.common = self.common
            self.vals[name] = val

    def __str__(self) -> str:
        flags = []
        if self.na:
            flags.append("NA")
        if self.ukn_val:
            flags.append("UKN")
        if self.unroll:
            if self.unroll_with:
                flags.append(f"UNROLL({','.join(self.unroll_with)})")
            else:
                flags.append("UNROLL")
        if self.partition:
            if self.partition_with:
                flags.append(f"PARTN({','.join(self.partition_with)})")
            else:
                flags.append("PARTN")
        
        return f"Attr[{','.join(flags)}]{self.vals}"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, col: str) -> Value:
        return self.vals[col]


Attributes = Mapping[str | tuple[str], Attribute]


def OrdAttribute(
    name: str,
    vals: list[Any],
    na: bool = False,
    ukn_val: Any | None = None,
    partition: bool = False,
    partition_with: tuple[str, ...] = tuple(),
):
    """Returns an Attribute holding a single Stratified Value where its children
    are ordinal, based on the provided data."""
    cols = {name: _create_strat_value_ord(vals, na, ukn_val)}
    return Attribute(
        name,
        cols,
        na,
        ukn_val is not None,
        partition=partition,
        partition_with=partition_with,
    )


def CatAttribute(
    name: str,
    vals: list[Any],
    na: bool = False,
    ukn_val: Any | None = None,
    partition: bool = False,
    partition_with: tuple[str, ...] = tuple(),
):
    """Returns an Attribute holding a single Stratified Value where its children
    are categorical, based on the provided data."""
    cols = {name: _create_strat_value_cat(vals, na, ukn_val)}
    return Attribute(
        name,
        cols,
        na,
        ukn_val is not None,
        partition=partition,
        partition_with=partition_with,
    )


def NumAttribute(
    name: str,
    bins: int,
    min: int | float | None,
    max: int | float | None,
    nullable: bool = False,
):
    """Returns an Attribute holding a single NumValue with the provided data."""
    return Attribute(name, {name: NumValue(bins, min, max)}, nullable, False)


def SeqAttribute(name: str, table: str):
    """Returns an Attribute holding a single SeqValue with the provided data."""
    return Attribute(name, {name: SeqValue(name, table)}, False, False)


def GenAttribute(name: str, table: str, max_len: int):
    """Returns an Attribute holding a single GenerationValue with the provided data."""
    return Attribute(name, {name: GenerationValue(table, max_len)}, False, False)


__all__ = [
    "get_dtype",
    "Grouping",
    "Value",
    "CatValue",
    "NumValue",
    "StratifiedValue",
    "Attribute",
    "Attributes",
    "OrdAttribute",
    "CatAttribute",
    "NumAttribute",
]
