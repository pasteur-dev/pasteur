import networkx as nx
from IPython.core.display import display, SVG


def display_graph(g, prog="dot", graph={}, nodes={}, edges={}):
    process_args = lambda args, pref: [f"{pref}{k}={v}" for k, v in args.items()]
    args = (
        process_args(graph, "-G")
        + process_args(edges, "-E")
        + process_args(nodes, "-N")
    )

    display(SVG(nx.nx_pydot.to_pydot(g).create(format="svg", prog=[prog, *args])))


def enchanced_display(g):
    o = nx.DiGraph() if g.is_directed() else nx.Graph()

    for name, data in g.nodes(data=True):
        label = ""
        if data["table"]:
            label += f"{data['table']}\n"
        if data["attr"] != data["value"]:
            label += f"{data['attr']}->{data['value'].replace(data['attr'] + '_', '')}"
        else:
            label += data["value"]

        new_data = {
            "label": label
        }

        if data.get('marked', False):
            new_data['color'] = 'green'

        o.add_node(name, **new_data)

    for a, b, data in g.edges(data=True):
        h_a = g.nodes[a]["heights"][b]
        if not h_a:
            h_a = ""
        h_b = g.nodes[b]["heights"][a]
        if not h_b:
            h_b = ""

        new_data = {"headlabel": h_b, "taillabel": h_a}
        if data.get("immorality", False):
            new_data["color"] = "red"
        if data.get("immoral", False):
            new_data["color"] = "blue"
        if data.get("triangulated", False):
            new_data["color"] = "green"
        o.add_edge(a, b, **new_data)

    display_graph(o, edges={"labeldistance": 1.5})
