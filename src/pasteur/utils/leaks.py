""" Simple wrapper that helps with finding memory leaks.

>>> from pasteur.utils.leaks import clear, check, graph
>>> clear()
>>> a = suspicious_fun()
>>> del a # remove result of function
>>> check() # prints new objects, should be empty or contain misc. python objects

Rerun a few times, if `check()` prints new objects, especially big ones, with specific
frequency, run `graph('suspicious_class')` to produce a graph of what keeps the
object in memory.

Example output of check:
```
set                     5365       +32
CategoricalDtype          72       +32
Index                     75       +31
ObjectEngine              61       +30
ReferenceType          16491       +20
slice                    154       +19
BlockPlacement            62       +19
cell                   17802       +17
method                  1105       +16
CategoricalAccessor       32       +16
CategoricalBlock          32       +16
Flags                     34       +10
Series                    30        +9
SingleBlockManager        30        +9
NumericBlock              30        +3
Int64Index                 7        +3
Int64Engine                5        +2
DataFrame                  4        +1 < suspicious, +1 per iteration
BlockManager               4        +1 < suspicious
ExtensionBlock             2        +1 < suspicious
IntegerArray               2        +1 < suspicious
```"""

import logging

try:
    import objgraph
except:
    print("Module 'objgraph' is required to use this module.")
    raise

logger = logging.getLogger(__name__)


def clear():
    objgraph.growth()


def check(info: str):
    result = objgraph.growth(1000)
    if not result:
        return

    width = max(len(name) for name, _, _ in result)
    for name, count, delta in result:
        info += "\n%-*s%9d %+9d" % (width, name, count, delta)

    logger.warning(info)


def graph(name: str):
    import random
    from io import StringIO
    from urllib.parse import quote

    with StringIO() as output:
        objgraph.show_chain(
            objgraph.find_backref_chain(
                random.choice(objgraph.by_type(name)), objgraph.is_proper_module
            ),
            output=output,
        )

        print("https://dreampuf.github.io/GraphvizOnline/#" + quote(output.getvalue()))
