"""Structural validation + a torch dry-run that infers per-node output shapes
and surfaces the first shape-related error with its owning node id.
"""

import torch

from .builder import GraphModule, make_dummy_input
from .schema import GraphError, validate_graph


def dry_run(graph):
    """Returns {"ok": bool, "errors": [{"node_id","message"}], "shapes": {node_id: [dims...]}}"""
    struct_errors = validate_graph(graph)
    if struct_errors:
        return {"ok": False, "errors": [e.to_dict() for e in struct_errors], "shapes": {}}

    try:
        module = GraphModule(graph)
    except GraphError as e:
        return {"ok": False, "errors": [e.to_dict()], "shapes": {}}

    input_node = module.node_by_id[module.input_id]
    dummy = make_dummy_input(input_node, batch=2)

    module.eval()
    with torch.no_grad():
        _, shapes, err = module.forward_with_shapes(dummy)

    shapes_out = {nid: list(shape) for nid, shape in shapes.items()}
    if err is not None:
        return {"ok": False, "errors": [err.to_dict()], "shapes": shapes_out}
    return {"ok": True, "errors": [], "shapes": shapes_out}
