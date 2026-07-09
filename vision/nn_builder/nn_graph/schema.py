"""Graph JSON schema helpers: structural validation + topological sort.

Graph shape:
{
  "id": "...", "meta": {"name": "..."},
  "nodes": [{"id": "n1", "type": "conv2d", "params": {...}, "position": {"x":0,"y":0}}],
  "edges": [{"id": "e1", "source": {"node": "n1", "port": "out"},
                          "target": {"node": "n2", "port": "in"}}]
}
"""

from .catalog import NODE_CATALOG, CONFIG_TYPES


class GraphError(Exception):
    def __init__(self, message, node_id=None):
        super().__init__(message)
        self.message = message
        self.node_id = node_id

    def to_dict(self):
        return {"node_id": self.node_id, "message": self.message}


def _arity(node_type):
    spec = NODE_CATALOG[node_type]
    inputs = spec["inputs"]
    if inputs == -1:
        return spec.get("min_inputs", 2), None
    return inputs, inputs


def validate_graph(graph):
    """Structural validation only (no torch). Returns list of GraphError."""
    errors = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_by_id = {}
    for n in nodes:
        if n["id"] in node_by_id:
            errors.append(GraphError(f"Duplicate node id '{n['id']}'", n["id"]))
            continue
        if n["type"] not in NODE_CATALOG:
            errors.append(GraphError(f"Unknown node type '{n['type']}'", n["id"]))
            continue
        node_by_id[n["id"]] = n

    incoming = {nid: [] for nid in node_by_id}
    outgoing = {nid: [] for nid in node_by_id}

    for e in edges:
        src, tgt = e.get("source", {}), e.get("target", {})
        src_id, tgt_id = src.get("node"), tgt.get("node")
        if src_id not in node_by_id:
            errors.append(GraphError(f"Edge {e.get('id')} references missing source node '{src_id}'"))
            continue
        if tgt_id not in node_by_id:
            errors.append(GraphError(f"Edge {e.get('id')} references missing target node '{tgt_id}'"))
            continue
        outgoing[src_id].append(e)
        incoming[tgt_id].append(e)

    input_nodes, output_nodes = [], []
    for nid, n in node_by_id.items():
        ntype = n["type"]
        if ntype in CONFIG_TYPES:
            continue
        if ntype == "input":
            input_nodes.append(nid)
            continue
        if ntype == "output":
            output_nodes.append(nid)
        min_in, exact_in = _arity(ntype)
        n_in = len(incoming[nid])
        if exact_in is not None and n_in != exact_in:
            errors.append(GraphError(
                f"Node '{nid}' ({ntype}) expects {exact_in} input edge(s), got {n_in}", nid))
        elif exact_in is None and n_in < min_in:
            errors.append(GraphError(
                f"Node '{nid}' ({ntype}) expects at least {min_in} input edges, got {n_in}", nid))

    if not input_nodes:
        errors.append(GraphError("Graph must contain at least one 'input' node"))
    if not output_nodes:
        errors.append(GraphError("Graph must contain at least one 'output' node"))

    for kind in ("dataset", "optimizer", "loss"):
        count = sum(1 for n in node_by_id.values() if n["type"] == kind)
        if count != 1:
            errors.append(GraphError(f"Graph must contain exactly one '{kind}' node, found {count}"))

    if errors:
        return errors  # topo sort meaningless if arity/refs are already broken

    try:
        topo_sort(node_by_id, incoming, outgoing)
    except GraphError as e:
        errors.append(e)

    return errors


def topo_sort(node_by_id, incoming, outgoing):
    """Kahn's algorithm over forward-graph nodes (config nodes excluded)."""
    forward_ids = [nid for nid, n in node_by_id.items() if n["type"] not in CONFIG_TYPES]
    remaining_in = {nid: len(incoming[nid]) for nid in forward_ids}
    queue = [nid for nid in forward_ids if remaining_in[nid] == 0]
    order = []

    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for e in outgoing[nid]:
            tgt = e["target"]["node"]
            if tgt not in remaining_in:
                continue
            remaining_in[tgt] -= 1
            if remaining_in[tgt] == 0:
                queue.append(tgt)

    if len(order) != len(forward_ids):
        stuck = set(forward_ids) - set(order)
        raise GraphError(f"Graph contains a cycle involving nodes: {sorted(stuck)}")

    return order


def graph_from_dict(d):
    """Light normalization helper (fills defaults for optional fields)."""
    d.setdefault("meta", {})
    d.setdefault("nodes", [])
    d.setdefault("edges", [])
    for n in d["nodes"]:
        n.setdefault("params", {})
        n.setdefault("position", {"x": 0, "y": 0})
    return d
