from collections import defaultdict
from typing import TYPE_CHECKING, Sequence
import networkx as nx
from IPython.core.display import display, SVG


# Edge-category palette used by display_induced_graph.  Tuned for legibility
# against a white background and to read distinctly when overlapping.
_EDGE_STYLES = {
    "measured":  {"color": "#1f77b4", "penwidth": "1.0"},                   # structure=True (chosen 2-way)
    "induced":   {"color": "#7f7f7f", "style": "dashed", "penwidth": "0.8"},# evidence=True (added by moralization)
    "chain":     {"color": "#7f7f7f", "penwidth": "0.6"},                   # chain / chain_bridged (height refinement)
    "common":    {"color": "#2ca02c", "penwidth": "0.8"},                   # cmn[0] -> v[h-1] glue (no flag)
    # legacy hugin pipeline tags (preserved for back-compat)
    "immorality":   {"color": "red"},
    "immoral":      {"color": "blue"},
    "triangulated": {"color": "green"},
}


def display_graph(g, prog="dot", graph={}, nodes={}, edges={}):
    display_pydot(nx.nx_pydot.to_pydot(g), prog, graph, nodes, edges)


_ZOOM_CSS = (
    "html,body{margin:0;padding:0;background:#fff}"
    "body{cursor:grab;min-height:100vh}"
    "html.dragging,html.dragging body{cursor:grabbing!important;"
    "user-select:none}"
    "html.dragging *{cursor:grabbing!important}"
    "svg{display:block}"
    "#zoom-ctl{position:fixed;top:8px;right:8px;z-index:1000;"
    "display:flex;gap:4px;background:#fff;border:1px solid #ccc;"
    "border-radius:4px;padding:3px;"
    "font:13px/1 -apple-system,Segoe UI,sans-serif}"
    "#zoom-ctl button{width:26px;height:26px;cursor:pointer;"
    "border:1px solid #ccc;background:#fafafa;border-radius:3px;"
    "padding:0;font:600 14px/1 inherit}"
    "#zoom-ctl button:hover{background:#eee}"
    "#zoom-ctl span{align-self:center;min-width:36px;text-align:center;"
    "color:#666;font-size:11px}"
)

_ZOOM_CTL = (
    '<div id="zoom-ctl">'
    '<button title="Zoom out (or scroll down)" onclick="zoomBy(0.8)">−</button>'
    '<span id="zoom-lvl">100%</span>'
    '<button title="Zoom in (or scroll up)" onclick="zoomBy(1.25)">+</button>'
    '<button title="Reset zoom" onclick="resetZoom()">⤢</button>'
    "</div>"
)

# Cursor-anchored wheel zoom + drag-to-pan, applied to every <svg> in
# the document so it works for both single-SVG and stitched multi-SVG
# pages.
_ZOOM_JS = (
    "<script>(function(){"
    "var svgs=Array.from(document.querySelectorAll('svg'));"
    "if(!svgs.length)return;"
    "var bases=svgs.map(function(s){"
    "var w=s.getAttribute('width')||'';"
    "var h=s.getAttribute('height')||'';"
    "return{w:parseFloat(w),h:parseFloat(h),"
    "unit:(w.match(/[a-z%]+$/i)||['px'])[0]};"
    "});"
    "var scale=1;var lvl=document.getElementById('zoom-lvl');"
    "function apply(){"
    "svgs.forEach(function(s,i){"
    "s.setAttribute('width',(bases[i].w*scale)+bases[i].unit);"
    "s.setAttribute('height',(bases[i].h*scale)+bases[i].unit);"
    "});"
    "lvl.textContent=Math.round(scale*100)+'%';"
    "}"
    "function zoomAt(factor,px,py){"
    "var oldScale=scale;"
    "scale=Math.max(0.05,Math.min(20,scale*factor));"
    "if(scale===oldScale)return;"
    "apply();"
    "var r=scale/oldScale;"
    "window.scrollBy(px*(r-1),py*(r-1));"
    "}"
    "window.zoomBy=function(f){"
    "zoomAt(f,window.scrollX+window.innerWidth/2,"
    "window.scrollY+window.innerHeight/2);"
    "};"
    "window.resetZoom=function(){scale=1;apply();};"
    "document.addEventListener('wheel',function(e){"
    "if(e.target.closest('#zoom-ctl'))return;"
    "e.preventDefault();"
    "var f=Math.exp(-e.deltaY*0.0015);"
    "zoomAt(f,e.pageX,e.pageY);"
    "},{passive:false});"
    "var drag=null;"
    "document.addEventListener('mousedown',function(e){"
    "if(e.button!==0)return;"
    "if(e.target.closest('#zoom-ctl'))return;"
    "drag={x:e.clientX,y:e.clientY,"
    "sx:window.scrollX,sy:window.scrollY};"
    "document.documentElement.classList.add('dragging');"
    "e.preventDefault();"
    "});"
    "document.addEventListener('mousemove',function(e){"
    "if(!drag)return;"
    "window.scrollTo(drag.sx-(e.clientX-drag.x),"
    "drag.sy-(e.clientY-drag.y));"
    "});"
    "function endDrag(){"
    "if(!drag)return;"
    "drag=null;"
    "document.documentElement.classList.remove('dragging');"
    "}"
    "document.addEventListener('mouseup',endDrag);"
    "document.addEventListener('mouseleave',endDrag);"
    "})();</script>"
)


def strip_svg_preamble(svg) -> str:
    """Strip the <?xml ?> preamble and any DOCTYPE declarations from
    an SVG so it can be safely inlined into an HTML body."""
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8")
    svg = svg.lstrip()
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1].lstrip()
    while svg.startswith("<!DOCTYPE") or svg.startswith("<!doctype"):
        svg = svg.split(">", 1)[1].lstrip()
    return svg


def wrap_zoom_html(body: str, title: str, extra_css: str = "") -> str:
    """Wrap ``body`` (which contains one or more inlined <svg> blocks)
    in an HTML page with zoom controls, cursor-anchored wheel zoom,
    and drag-to-pan."""
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>{_ZOOM_CSS}{extra_css}</style></head><body>"
        f"{_ZOOM_CTL}{body}{_ZOOM_JS}</body></html>"
    )


def write_svg_html(svg, out_path: str) -> str:
    """Write an HTML wrapper that inlines ``svg`` (bytes or str), with
    body overflow set so the page scrolls (mlflow's artifact viewer
    doesn't scroll bare SVG).  Output goes to ``out_path``; if it ends
    in ``.svg`` the suffix is replaced with ``.html``."""
    from pathlib import Path

    out = Path(out_path)
    if out.suffix == ".svg":
        out = out.with_suffix(".html")
    html = wrap_zoom_html(strip_svg_preamble(svg), out.stem)
    with open(out, "w") as f:
        f.write(html)
    return str(out)


def display_pydot(g, prog="dot", graph={}, nodes={}, edges={}, out=None):
    process_args = lambda args, pref: [f"{pref}{k}={v}" for k, v in args.items()]
    args = (
        process_args(graph, "-G")
        + process_args(edges, "-E")
        + process_args(nodes, "-N")
    )

    svg = g.create(format="svg", prog=[prog, *args])
    if out is not None:
        if str(out).endswith(".svg"):
            write_svg_html(svg, out)
        else:
            with open(out, "wb") as f:
                f.write(svg)
    else:
        display(SVG(svg))


def _classify_edge(data):
    """Map moral-graph edge data → category key in _EDGE_STYLES."""
    if data.get("structure"):
        return "measured"
    if data.get("evidence"):
        return "induced"
    if data.get("chain") or data.get("chain_bridged"):
        return "chain"
    if data.get("immorality"):
        return "immorality"
    if data.get("immoral"):
        return "immoral"
    if data.get("triangulated"):
        return "triangulated"
    # No flag: in adjuvant's moral graph this is the cmn[0]→v[h-1] glue edge
    # carried over from build_height_chain_graph.
    return "common"


def _node_domain(g, node, attrs):
    """Cardinality of a height-chain graph node's variable, via attrs lookup."""
    from .hugin import get_attrs

    d = g.nodes[node]
    table, order, attr_name = d.get("table"), d.get("order"), d["attr"]
    try:
        attr = get_attrs(attrs, table, order)[attr_name]
    except (KeyError, TypeError):
        return None

    height = d.get("height", 0)
    if d.get("is_common"):
        if attr.common is None:
            return None
        return attr.common.get_domain(height)
    val = attr.vals.get(d["value"]) if hasattr(attr.vals, "get") else attr.vals[d["value"]]
    if val is None:
        return None
    return val.get_domain(height)


def _build_induced_graph_into(
    container,
    g,
    condensed=True,
    attrs=None,
    prefix: str = "",
    skip_nodes: set | None = None,
    node_alias: dict | None = None,
    order_label=None,
):
    """Add an induced/moral graph to a container with id-namespacing.

    ``prefix`` is prepended to every node id and every cluster name so
    multiple induced graphs can coexist in one Dot without collisions.

    ``skip_nodes`` is a set of source node names to drop entirely (no
    rendering, no incident edges).
    ``node_alias`` maps a source node name to an *external* node id
    (already living elsewhere in the Dot).  Aliased nodes are not
    rendered; their incident edges are rerouted to the external id.
    ``order_label`` is an optional callable ``(order:int) -> str``
    used in the (table, order) cluster label; defaults to ``str(order)``.

    Returns a dict with:
      - ``table_subs``: (table, order) -> pydot Subgraph / Cluster
      - ``cluster_ids``: (table, order) -> graphviz cluster id (for
        ltail/lhead anchoring), or None if the (table, order) is a
        plain Subgraph
      - ``anchors``: (table, order) -> any node id inside that group
        (suitable for cross-cluster edges that need a real node target)
      - ``val_names``: (table, order, attr, value) -> rendered node id
    """
    import pydot

    skip_nodes = skip_nodes or set()
    node_alias = node_alias or {}
    order_label = order_label or (lambda o: str(o))

    attr_counts = defaultdict(int)
    attr_vals = defaultdict(set)
    marked_vals = defaultdict(lambda: False)
    for n, d in g.nodes(data=True):
        if n in skip_nodes or n in node_alias:
            continue
        attr_counts[(d["table"], d["order"], d["attr"])] += 1
        attr_vals[(d["table"], d["order"], d["attr"])].add(d["value"])
        marked_vals[(d["table"], d["order"], d["attr"], d["value"])] |= d.get(
            "marked", False
        )

    table_subs: dict[tuple, pydot.Graph] = {}
    cluster_ids: dict[tuple, str | None] = {}
    attr_subs: dict[tuple, pydot.Graph] = {}
    val_names: dict[tuple, str] = {}
    anchors: dict[tuple, str] = {}

    def nid(name: str) -> str:
        if name in node_alias:
            return node_alias[name]
        return f"{prefix}{name}" if prefix else name

    for node_name, d in g.nodes(data=True):
        if node_name in skip_nodes or node_name in node_alias:
            continue

        if (d["table"], d["order"]) not in table_subs:
            if d["table"] and d["order"] is not None:
                label = f"{d['table']}[{order_label(d['order'])}]"
            elif d["table"]:
                label = d["table"]
            else:
                label = ""

            sub_name = f"{prefix}{d['table']}[{d['order']}]"
            if label:
                sub = pydot.Cluster(
                    sub_name, label=label, penwidth="2.0", margin="6"
                )
                cluster_ids[(d["table"], d["order"])] = f"cluster_{sub_name}"
            else:
                sub = pydot.Subgraph(sub_name, label=label)
                cluster_ids[(d["table"], d["order"])] = None
            table_subs[(d["table"], d["order"])] = sub
            container.add_subgraph(sub)

        if (d["table"], d["order"], d["attr"]) not in attr_subs:
            attr_name = f"{prefix}{d['table']}[{d['order']}]/{d['attr']}"

            if attr_counts[(d["table"], d["order"], d["attr"])] <= 1 or (
                condensed and len(attr_vals[d["table"], d["order"], d["attr"]]) == 1
            ):
                sub = pydot.Subgraph(attr_name, label="")
            else:
                sub = pydot.Cluster(
                    attr_name, label=d["attr"], penwidth="1.2", margin="4"
                )
            attr_subs[(d["table"], d["order"], d["attr"])] = sub
            table_subs[(d["table"], d["order"])].add_subgraph(sub)

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
            val_names[(d["table"], d["order"], d["attr"], d["value"])] = nid(node_name)
            attr_subs[(d["table"], d["order"], d["attr"])].add_node(
                pydot.Node(nid(node_name), **new_data)
            )
            anchors.setdefault((d["table"], d["order"]), nid(node_name))

    # In condensed mode multiple height-pairs collapse onto the same
    # rendered (a_id, b_id) — emit only one edge per (endpoints,
    # category) group, keeping the candidate with the largest domain.
    edge_buf: dict[tuple, tuple] = {}
    directed = g.is_directed()

    for a, b, data in g.edges(data=True):
        if a in skip_nodes or b in skip_nodes:
            continue
        # Both endpoints aliased -> the edge is owned (and drawn) by
        # the source submodel; skip to avoid duplicate top-level edges.
        if a in node_alias and b in node_alias:
            continue

        category = _classify_edge(data)
        new_data = dict(_EDGE_STYLES.get(category, {}))

        domain = None
        if attrs is not None:
            da = _node_domain(g, a, attrs)
            db = _node_domain(g, b, attrs)
            if da is not None and db is not None:
                domain = da * db
                new_data["label"] = f"{domain:,d}"
                new_data["fontcolor"] = new_data.get("color", "black")

        if a in node_alias or b in node_alias:
            dst = container
        elif (
            g.nodes[a]["table"] != g.nodes[b]["table"]
            or g.nodes[a]["order"] != g.nodes[b]["order"]
        ):
            dst = container
        elif g.nodes[a]["attr"] != g.nodes[b]["attr"]:
            dst = table_subs[(g.nodes[a]["table"], g.nodes[a]["order"])]
        else:
            dst = attr_subs[
                (g.nodes[a]["table"], g.nodes[a]["order"], g.nodes[a]["attr"])
            ]

        if condensed:
            if ah := g.nodes[a]["height"]:
                new_data["taillabel"] = f'<<FONT POINT-SIZE="9" COLOR="#888888">{ah}</FONT>>'
            if bh := g.nodes[b]["height"]:
                new_data["headlabel"] = f'<<FONT POINT-SIZE="9" COLOR="#888888">{bh}</FONT>>'

            if a in node_alias:
                a_id = node_alias[a]
            else:
                a_id = val_names[
                    (
                        g.nodes[a]["table"],
                        g.nodes[a]["order"],
                        g.nodes[a]["attr"],
                        g.nodes[a]["value"],
                    )
                ]
            if b in node_alias:
                b_id = node_alias[b]
            else:
                b_id = val_names[
                    (
                        g.nodes[b]["table"],
                        g.nodes[b]["order"],
                        g.nodes[b]["attr"],
                        g.nodes[b]["value"],
                    )
                ]
            if a_id == b_id:
                continue
        else:
            a_id, b_id = nid(a), nid(b)

        # Source moral graph is undirected, but we orient every edge so
        # dot can use it as a rank constraint (otherwise the layout
        # sprawls horizontally inside clusters).  Orientation rules:
        #   1. Cross-table aliased edges: aliased side is the source.
        #   2. Inter-order edges (e.g. order=-1 -> order=0/None): the
        #      earlier timestep precedes the later one.  Visible arrow.
        #   3. Otherwise: orient by sorted node id so cliques become
        #      DAGs (no init_rank cycles).  Rendered without arrows.
        a_aliased = a in node_alias
        b_aliased = b in node_alias

        def _swap_endpoints():
            nonlocal a_id, b_id
            a_id, b_id = b_id, a_id
            if "taillabel" in new_data and "headlabel" in new_data:
                new_data["taillabel"], new_data["headlabel"] = (
                    new_data["headlabel"], new_data["taillabel"],
                )
            elif "taillabel" in new_data:
                new_data["headlabel"] = new_data.pop("taillabel")
            elif "headlabel" in new_data:
                new_data["taillabel"] = new_data.pop("headlabel")

        if a_aliased ^ b_aliased:
            if b_aliased:
                _swap_endpoints()
            new_data["dir"] = "forward"
            new_data["color"] = "#d62728"
            new_data["fontcolor"] = "#d62728"
        else:
            a_order = g.nodes[a]["order"]
            b_order = g.nodes[b]["order"]
            a_common = g.nodes[a].get("is_common", False)
            b_common = g.nodes[b].get("is_common", False)
            if a_order != b_order:
                # Inter-order: earlier (more-negative) precedes later
                # (None means current).  Orient regardless of table —
                # the "current main" lives in a table=None bucket but
                # is conceptually the latest timestep.
                def _rank_key(o):
                    return float("inf") if o is None else o
                if _rank_key(a_order) > _rank_key(b_order):
                    _swap_endpoints()
                new_data["dir"] = "forward"
                new_data["color"] = "#d62728"
                new_data["fontcolor"] = "#d62728"
            elif a_common != b_common:
                # Common-glue (cmn[0] -> v[h-1]): orient common above
                # its values so dot ranks the apex above descendants.
                if b_common:
                    _swap_endpoints()
                new_data["dir"] = "none"
            elif category == "measured":
                # Selected structure edges: orient deterministically
                # by sorted endpoint id so dot uses them as DAG-style
                # rank drivers (acyclic by construction — every edge
                # points from the smaller id to the larger, so no
                # init_rank cycles).  Stratifies cross-attribute
                # placement that would otherwise pile up at one rank.
                if a_id > b_id:
                    _swap_endpoints()
                new_data["dir"] = "none"
            else:
                new_data["dir"] = "none"
                new_data["constraint"] = "false"

        ep = (a_id, b_id) if directed else tuple(sorted((a_id, b_id)))
        key = (ep, category)
        cur = edge_buf.get(key)
        cur_domain = cur[0] if cur else None
        # Replace iff: no incumbent, OR incumbent has no domain and we
        # do, OR our domain strictly exceeds incumbent's.
        if (
            cur is None
            or (cur_domain is None and domain is not None)
            or (domain is not None and cur_domain is not None and domain > cur_domain)
        ):
            edge_buf[key] = (domain, dst, a_id, b_id, new_data)

    for _, (_, dst, a_id, b_id, new_data) in edge_buf.items():
        dst.add_edge(pydot.Edge(a_id, b_id, **new_data))

    return {
        "table_subs": table_subs,
        "cluster_ids": cluster_ids,
        "anchors": anchors,
        "val_names": val_names,
    }


def display_induced_graph(g, condensed=True, attrs=None, out=None):
    """Render a height-chain / moral graph.

    If ``attrs`` (a DatasetAttributes mapping) is provided, each edge is
    labeled with the product of its endpoint-variable domains, and the
    junction-tree-style edge categories are colored:
      - measured   (structure=True)         blue, thick
      - induced    (evidence=True)          red, dashed
      - chain      (chain / chain_bridged)  gray
      - common     (cmn glue, no flag)      green
    The legacy immorality/immoral/triangulated tags are preserved.
    """
    import pydot

    if g.is_directed():
        graph_type = "digraph"
    else:
        graph_type = "graph"
    strict = nx.number_of_selfloops(g) == 0 and not g.is_multigraph() and not condensed

    graph_defaults = {
        "nodesep": "0.12",
        "ranksep": "0.28",
        "margin": "0.05",
        "pad": "0.1",
        "splines": "true",
        "overlap": "false",
        **g.graph.get("graph", {}),
    }
    o = pydot.Dot(g.name, graph_type=graph_type, strict=strict, **graph_defaults)
    o.set_node_defaults(fontsize="9", margin="0.03,0.015")
    o.set_edge_defaults(fontsize="9", penwidth="0.8", arrowsize="0.6")

    _build_induced_graph_into(o, g, condensed=condensed, attrs=attrs)

    display_pydot(
        o,
        edges={"labeldistance": 1.0, "labelfontsize": 8, "arrowsize": 0.6},
        out=out,
    )


def _build_junction_tree_into(
    container,
    junction: nx.Graph,
    attrs=None,
    messages: Sequence[Sequence] | None = None,
    prefix: str = "",
):
    """Add a junction tree's nodes and edges to a pydot container.

    ``prefix`` namespaces node ids so multiple junction trees can coexist
    in the same Dot graph without id collisions.  Returns a function
    ``node_id(cl)`` that maps a clique to its prefixed id, useful for
    cross-cluster connections.
    """
    import pydot

    if messages:
        message_order = {}
        for i, generation in enumerate(messages):
            for message in generation:
                message_order[message] = i + 1
    else:
        message_order = None

    if attrs is not None:
        from .hugin import get_clique_domain
    else:
        get_clique_domain = None

    def node_id(cl):
        return f"{prefix}{cl}" if prefix else str(cl)

    for cl in junction.nodes():
        n_vars = sum(
            1 if isinstance(sel, int) else len(sel) for _, _, _, sel in cl
        )
        header_bits = [f"|V|={n_vars}"]
        if get_clique_domain is not None:
            try:
                dom = get_clique_domain(cl, attrs)
                header_bits.append(f"{dom:,d}")
            except Exception:
                pass
        header = " · ".join(header_bits)

        label = '<<TABLE CELLBORDER="1" BORDER="0" CELLPADDING="2" CELLSPACING="0">'
        label += (
            f'<TR><TD COLSPAN="2" BGCOLOR="#eeeeee">'
            f'<I>{header}</I></TD></TR>'
        )
        for table, order, attr, sel in cl:
            if isinstance(sel, int):
                label += f'<TR><TD ALIGN="LEFT"><B>{attr}</B></TD><TD>{sel}</TD></TR>'
            elif len(sel) == 1:
                val, h = next(iter(sel))
                label += f'<TR><TD ALIGN="LEFT"><B>{val}</B></TD><TD>{h}</TD></TR>'
            else:
                label += f'<TR><TD COLSPAN="2"><B>{attr}</B></TD></TR>'
                for val, h in sorted(sel):
                    label += f'<TR><TD ALIGN="LEFT">{val}</TD><TD>{h}</TD></TR>'

        label += "</TABLE>>"

        container.add_node(pydot.Node(node_id(cl), label=label, shape="plaintext"))

    for a, b, d in junction.edges(data=True):
        if not d.get("common"):
            continue  # empty separator — MST bridge between disjoint cliques
        new_data = {"label": f"{d['common']}  ({d['domain']:,d})"}

        if message_order:
            new_data["dir"] = "both"
            new_data["taillabel"] = (
                f'<<FONT COLOR="#1f77b4"><B>{message_order[(b, a)]}</B></FONT>>'
            )
            new_data["headlabel"] = (
                f'<<FONT COLOR="#1f77b4"><B>{message_order[(a, b)]}</B></FONT>>'
            )

        container.add_edge(pydot.Edge(node_id(a), node_id(b), **new_data))

    return node_id


def display_junction_tree(
    junction: nx.Graph,
    g: nx.Graph | nx.DiGraph,
    messages: Sequence[Sequence] | None = None,
    attrs=None,
    out=None,
):
    """Render a junction tree.

    Each clique node gets a header row ``|V|=<vars> · <domain>`` (the
    total domain is computed when ``attrs`` is provided).  Message-pass
    indices are rendered as bold colored head/tail labels so they
    don't collide with the ``common (domain)`` edge label.
    """
    import pydot

    o = pydot.Dot(
        graph_type="graph",
        nodesep="0.08",
        ranksep="0.18",
        margin="0.02",
        pad="0.05",
        splines="true",
        overlap="false",
    )
    o.set_node_defaults(fontsize="9", margin="0")
    o.set_edge_defaults(fontsize="9", penwidth="0.8", arrowsize="0.6")

    _build_junction_tree_into(o, junction, attrs=attrs, messages=messages)

    display_pydot(o, edges={"labeldistance": 1.2, "labelangle": "20"}, out=out)
