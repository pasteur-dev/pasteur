from collections import defaultdict
from typing import Sequence
import networkx as nx
from IPython.core.display import display, SVG


def display_graph(g, prog="dot", graph={}, nodes={}, edges={}):
    display_pydot(nx.nx_pydot.to_pydot(g), prog, graph, nodes, edges)


def display_pydot(g, prog="dot", graph={}, nodes={}, edges={}):
    process_args = lambda args, pref: [f"{pref}{k}={v}" for k, v in args.items()]
    args = (
        process_args(graph, "-G")
        + process_args(edges, "-E")
        + process_args(nodes, "-N")
    )

    display(SVG(g.create(format="svg", prog=[prog, *args])))


def display_induced_graph(g, condensed=True):
    import pydot

    if g.is_directed():
        graph_type = "digraph"
    else:
        graph_type = "graph"
    strict = nx.number_of_selfloops(g) == 0 and not g.is_multigraph() and not condensed

    graph_defaults = g.graph.get("graph", {})
    o = pydot.Dot(g.name, graph_type=graph_type, strict=strict, **graph_defaults)

    # Keep how many nodes each attribute has for a cleaner appearance
    attr_counts = defaultdict(int)
    attr_vals = defaultdict(set)
    marked_vals = defaultdict(lambda: False)
    for node_name, d in g.nodes(data=True):
        attr_counts[(d["table"], d["order"], d["attr"])] += 1
        attr_vals[(d["table"], d["order"], d["attr"])].add(d["value"])
        marked_vals[(d["table"], d["order"], d["attr"], d["value"])] |= d.get(
            "marked", False
        )

    table_subs: dict[tuple, pydot.Graph] = {}
    attr_subs: dict[tuple, pydot.Graph] = {}
    val_names: dict[tuple, str] = {}  # pick random name for condensed node

    for node_name, d in g.nodes(data=True):
        # Set up clusters
        if (d["table"], d["order"]) not in table_subs:
            if d["table"] and d["order"] is not None:
                label = f"{d['table']}[{d['order']}]"
            elif d["table"]:
                label = d["table"]
            else:
                label = ""

            name = f"{d['table']}[{d['order']}]"
            if label:
                sub = pydot.Cluster(name, label=label)
            else:
                sub = pydot.Subgraph(name, label=label)
            table_subs[(d["table"], d["order"])] = sub
            o.add_subgraph(sub)

        if (d["table"], d["order"], d["attr"]) not in attr_subs:
            name = d["attr"]

            if attr_counts[(d["table"], d["order"], d["attr"])] <= 1 or (
                condensed and len(attr_vals[d["table"], d["order"], d["attr"]]) == 1
            ):
                sub = pydot.Subgraph(name, label="")
            else:
                sub = pydot.Cluster(name, label=d["attr"])
            attr_subs[(d["table"], d["order"], d["attr"])] = sub
            table_subs[(d["table"], d["order"])].add_subgraph(sub)

        # Setup attribute
        if len(attr_vals[d["table"], d["order"], d["attr"]]) == 1:
            if attr_counts[(d["table"], d["order"], d["attr"])] > 1 and not condensed:
                label = ""
            else:
                label = d["value"]
        else:
            label = d["value"].replace(d["attr"] + "_", "")

        if not condensed:
            label += f"[{d['height']}]"
        new_data = {"label": label}

        if d.get("marked", False) or (
            condensed and marked_vals[d["table"], d["order"], d["attr"], d["value"]]
        ):
            new_data["color"] = "green"

        if (
            not condensed
            or (d["table"], d["order"], d["attr"], d["value"]) not in val_names
        ):
            val_names[(d["table"], d["order"], d["attr"], d["value"])] = node_name
            attr_subs[(d["table"], d["order"], d["attr"])].add_node(
                pydot.Node(node_name, **new_data)
            )

    for a, b, data in g.edges(data=True):
        new_data = {}
        if data.get("immorality", False):
            new_data["color"] = "red"
        if data.get("immoral", False):
            new_data["color"] = "blue"
        if data.get("triangulated", False):
            new_data["color"] = "green"

        if (
            g.nodes[a]["table"] != g.nodes[b]["table"]
            or g.nodes[a]["order"] != g.nodes[b]["order"]
        ):
            dst = o
        elif g.nodes[a]["attr"] != g.nodes[b]["attr"]:
            dst = table_subs[(g.nodes[a]["table"], g.nodes[a]["order"])]
        else:
            dst = attr_subs[
                (g.nodes[a]["table"], g.nodes[a]["order"], g.nodes[a]["attr"])
            ]

        if condensed:
            if ah := g.nodes[a]["height"]:
                new_data["taillabel"] = str(ah)
            if bh := g.nodes[b]["height"]:
                new_data["headlabel"] = str(bh)

            a = val_names[
                (
                    g.nodes[a]["table"],
                    g.nodes[a]["order"],
                    g.nodes[a]["attr"],
                    g.nodes[a]["value"],
                )
            ]
            b = val_names[
                (
                    g.nodes[b]["table"],
                    g.nodes[b]["order"],
                    g.nodes[b]["attr"],
                    g.nodes[b]["value"],
                )
            ]
            if a == b:
                # Do not add edge to self
                continue

        dst.add_edge(pydot.Edge(a, b, **new_data))

    display_pydot(o, edges={"labeldistance": 1.5})


def display_junction_tree(
    junction: nx.Graph,
    g: nx.Graph | nx.DiGraph,
    messages: Sequence[Sequence] | None = None,
):
    import pydot

    if messages:
        message_order = {}
        for i, generation in enumerate(messages):
            for message in generation:
                message_order[message] = i + 1
    else:
        message_order = None

    o = pydot.Dot(graph_type="graph")

    for cl, data in junction.nodes(data=True):
        cl_attrs = defaultdict(dict)
        for node in cl:
            d = g.nodes[node]
            if d["value"] not in cl_attrs[(d["table"], d["order"], d["attr"])]:
                cl_attrs[(d["table"], d["order"], d["attr"])][d["value"]] = d["height"]
            else:
                cl_attrs[(d["table"], d["order"], d["attr"])][d["value"]] = min(
                    d["height"],
                    cl_attrs[(d["table"], d["order"], d["attr"])][d["value"]],
                )

        label = '<<TABLE CELLBORDER="1" BORDER="0">'
        for attr, vals in cl_attrs.items():
            if len(vals) > 1:
                label += f'<TR><TD COLSPAN="2"><B>{attr[-1]}</B></TD></TR>'
                for val, h in sorted(vals.items(), key=lambda a: a[0]):
                    label += f'<TR><TD ALIGN="LEFT">{val}</TD><TD>{h}</TD></TR>'
            else:
                val, h = next(iter(vals.items()))
                label += f'<TR><TD ALIGN="LEFT"><B>{val}</B></TD><TD>{h}</TD></TR>'

        label += "</TABLE>>"

        o.add_node(pydot.Node(str(cl), label=label, shape="plaintext"))

    for a, b, d in junction.edges(data=True):
        new_data = {"label": f"{d['common']} ({d['domain']:,d})"}

        if message_order:
            new_data["dir"] = "both"
            new_data["taillabel"] = f"<<B>{message_order[(b, a)]}</B>>"
            new_data["headlabel"] = f"<<B>{message_order[(a, b)]}</B>>"

        o.add_edge(pydot.Edge(str(a), str(b), **new_data))

    display_pydot(o, edges={"labeldistance": 1.5})
