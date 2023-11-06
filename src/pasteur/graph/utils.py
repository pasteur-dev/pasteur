from collections import defaultdict
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


def enchanced_display(g):
    import pydot

    if g.is_directed():
        graph_type = "digraph"
    else:
        graph_type = "graph"
    strict = nx.number_of_selfloops(g) == 0 and not g.is_multigraph()

    graph_defaults = g.graph.get("graph", {})
    o = pydot.Dot(g.name, graph_type=graph_type, strict=strict, **graph_defaults)

    # Keep how many nodes each attribute has for a cleaner appearance
    attr_counts = defaultdict(int)
    attr_vals = defaultdict(set)
    for node_name, d in g.nodes(data=True):
        attr_counts[(d["table"], d["order"], d["attr"])] += 1
        attr_vals[(d["table"], d["order"], d["attr"])].add(d["value"])

    table_subs: dict[tuple, pydot.Graph] = {}
    attr_subs: dict[tuple, pydot.Graph] = {}

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

            if attr_counts[(d["table"], d["order"], d["attr"])] > 1:
                sub = pydot.Cluster(name, label=d["attr"])
            else:
                sub = pydot.Subgraph(name, label="")
            attr_subs[(d["table"], d["order"], d["attr"])] = sub
            table_subs[(d["table"], d["order"])].add_subgraph(sub)

        # Setup attribute
        if len(attr_vals[d["table"], d["order"], d["attr"]]) == 1:
            if attr_counts[(d["table"], d["order"], d["attr"])] > 1:
                label = ""
            else:
                label = d["value"]
        else:
            label = d["value"].replace(d["attr"] + "_", "")

        label += f"[{d['height']}]"
        new_data = {"label": label}

        if d.get("marked", False):
            new_data["color"] = "green"
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
        dst.add_edge(pydot.Edge(a, b, **new_data))

    display_pydot(o, edges={"labeldistance": 1.5})
