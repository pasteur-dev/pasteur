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

from typing import Any, Literal, Mapping, TypeVar, Sequence

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


GI = TypeVar("GI", "Grouping", str)


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


class SeqValue(Value):
    table: str
    order: int | None

    def __init__(self, name: str, table: str, order: int | None = None) -> None:
        self.name = name
        self.table = table
        self.order = order

    def __str__(self) -> str:
        return f"Seq[{self.table},ord={self.order if self.order is not None else 'NA'}]"

    def __repr__(self) -> str:
        return str(self)


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

    def __init__(self, name: str, head: Grouping) -> None:
        self.name = name
        self.head = head

    def __str__(self) -> str:
        return "Idx" + str(self.head)

    def __repr__(self) -> str:
        return "Idx" + repr(self.head)

    def get_domain(self, height: int):
        return self.head.get_domain(height)

    def get_mapping(self, height: int):
        return self.head.get_mapping(height)

    def is_ordinal(self) -> bool:
        return self.head.type == "ord" and self.head.size == len(self.head)

    @property
    def height(self):
        return self.head.height


class GenerationValue(StratifiedValue):
    table: str
    max_len: int

    def __init__(self, name: str, table: str, max_len: int) -> None:
        self.table = table
        self.max_len = max_len
        super().__init__(name, Grouping("ord", list(range(max_len + 1))))


def _create_strat_value_cat(
    name: str, vals, na: bool = False, ukn_val: Any | None = None
):
    arr = []
    if na:
        arr.append(None)
    if ukn_val is not None:
        arr.append(ukn_val)
    arr.extend(vals)

    return StratifiedValue(name, Grouping("cat", arr))


def _create_strat_value_ord(
    name: str, vals, na: bool = False, ukn_val: Any | None = None
):
    g = Grouping("ord", vals)

    if na or ukn_val is not None:
        arr = []
        if na:
            arr.append(None)
        if ukn_val is not None:
            arr.append(ukn_val)
        arr.append(g)

        g = Grouping("cat", arr)

    return StratifiedValue(name, g)


OrdValue = _create_strat_value_ord


class NumValue(Value):
    """Numerical Value: its value can be represented with a number, which might be NaN."""

    def __init__(
        self,
        name: str,
        bins: np.ndarray | Sequence[float] | int,
        nullable: bool = False,
        min: int | float | None = None,
        max: int | float | None = None,
    ) -> None:
        self.name = name

        if isinstance(bins, np.ndarray) or isinstance(bins, Sequence):
            self.bins = np.array(bins)
        else:
            assert (
                bins is not None and min is not None and max is not None
            ), f"Please provide 'min' and 'max' when 'bins' is an int."
            self.bins = np.linspace(min, max, bins + 1)
        self.nullable = nullable

    def __str__(self) -> str:
        return f"Num[{', '.join([f'{f:.2f}' for f in self.bins])}]"

    def __repr__(self) -> str:
        return str(self)


class StratifiedNumValue(StratifiedValue):
    name_cnt: str

    def __init__(
        self,
        name: str,
        name_cnt: str,
        head: Grouping,
        null: None | Sequence[bool] = None,
    ) -> None:
        self.name_cnt = name_cnt
        if null:
            assert len(null) == head.get_domain(0)
            self.null = null
        else:
            self.null = [False for _ in range(head.get_domain(0))]

        super().__init__(name, head)

    def __str__(self) -> str:
        return "NumIdx" + str(self.head)

    def __repr__(self) -> str:
        return "NumIdx" + repr(self.head)


def _groups_match(main: Grouping, other: Grouping):
    """Checks that `other` mirrors the structure of `main`."""
    if len(main) != len(other):
        return False

    if main.type != other.type:
        return False

    for a, b in zip(main, other):
        if not isinstance(a, Grouping):
            continue

        if not isinstance(b, Grouping):
            return False

        if not _groups_match(a, b):
            return False

    return True


def CommonValue(
    name: str, na: bool = False, ukn_val: Any | None = None, normal_name: str = "Normal"
):

    vals = []
    if na:
        vals.append(None)
    if ukn_val is not None:
        vals.append(ukn_val)
    vals.append(normal_name)

    return StratifiedValue(name, Grouping("cat", vals))


class Attribute:
    """Attribute class which holds multiple values in a dictionary."""

    def __init__(
        self,
        name: str | tuple[str, ...],
        vals: Sequence[Value],
        common: CatValue | None = None,
        unroll: bool = False,
        along: tuple[str | tuple[str, ...], ...] = tuple(),
        partition: bool = False,
    ) -> None:
        self.name = name
        self.common = common
        self.unroll = bool(unroll)
        self.along = along
        self.partition = partition
        self.vals = {k.name: k for k in vals}

        # Perform a check for a valid common value
        if common:
            # Categorical Value check
            # Check that domains match for all categorical values with common value
            for v in vals:
                if not isinstance(v, CatValue):
                    continue

                if isinstance(common, StratifiedValue) and isinstance(
                    v, StratifiedValue
                ):
                    # For stratified values we traverse the tree and check if it matches
                    assert _groups_match(common.head, v.head)
                else:
                    for h in range(common.height + 1):
                        assert v.get_domain(v.height - h) == common.get_domain(h)

    def _str_pref(self):
        flags = []
        if self.unroll:
            if self.along:
                flags.append(
                    f"UNROLL({','.join(['_'.join(k) if not isinstance(k, str) else k for k in self.along])})"
                )
            else:
                flags.append("UNROLL")
        if self.partition:
            flags.append("PARTN")
        if self.common:
            flags.append(f"COMMON({self.common.name}:{self.common})")

        return f"Attr[{','.join(flags)}]"

    def __str__(self) -> str:
        return self._str_pref() + str(self.vals)

    def __repr__(self) -> str:
        return self._str_pref() + repr(self.vals)

    def __getitem__(self, col: str) -> Value:
        return self.vals[col]


Attributes = Mapping[str | tuple[str, ...], Attribute]


def OrdAttribute(
    name: str,
    vals: list[Any],
    na: bool = False,
    ukn_val: Any | None = None,
    partition: bool = False,
):
    """Returns an Attribute holding a single Stratified Value where its children
    are ordinal, based on the provided data."""
    return Attribute(
        name,
        [_create_strat_value_ord(name, vals, na, ukn_val)],
        partition=partition,
    )


def CatAttribute(
    name: str,
    vals: list[Any],
    na: bool = False,
    ukn_val: Any | None = None,
    partition: bool = False,
):
    """Returns an Attribute holding a single Stratified Value where its children
    are categorical, based on the provided data."""
    return Attribute(
        name,
        [_create_strat_value_cat(name, vals, na, ukn_val)],
        partition=partition,
    )


def NumAttribute(
    name: str,
    bins: int,
    min: int | float | None,
    max: int | float | None,
    nullable: bool = False,
):
    """Returns an Attribute holding a single NumValue with the provided data."""
    return Attribute(name, [NumValue(name, bins, nullable, min, max)])


def SeqAttribute(name: str, table: str, order: int | None = None):
    """Returns an Attribute holding a single SeqValue with the provided data."""
    return Attribute(name, [SeqValue(name, table, order)])


def GenAttribute(name: str, table: str, max_len: int):
    """Returns an Attribute holding a single GenerationValue with the provided data."""
    return Attribute(name, [GenerationValue(name, table, max_len)])


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
