"""Turns a validated graph JSON dict into a real, runnable torch.nn.Module.

Constraint (v1): exactly one 'input' node and one 'output' node per graph.
Conv/Linear/BatchNorm in-dimensions are inferred lazily (torch Lazy* modules,
or a custom lazy wrapper for RNN/LSTM/GRU) so the user never has to type
"in_channels" — only output-shape-determining params are exposed in the UI,
matching how playground-style tools work.
"""

import math

import torch
from torch import nn

from .catalog import NODE_CATALOG, CONFIG_TYPES
from .schema import GraphError, topo_sort


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x, need_weights=False)
        return out


class LazyRecurrent(nn.Module):
    """Wraps nn.RNN/LSTM/GRU, deferring input_size until first forward call."""

    _CLASSES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, kind, hidden_size, num_layers, bidirectional, dropout, return_sequence):
        super().__init__()
        self.kind = kind
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if num_layers > 1 else 0.0
        self.return_sequence = return_sequence
        self.rnn = None

    def forward(self, x):
        if self.rnn is None:
            input_size = x.shape[-1]
            cls = self._CLASSES[self.kind]
            self.rnn = cls(
                input_size, self.hidden_size, num_layers=self.num_layers,
                batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout,
            ).to(device=x.device, dtype=x.dtype)
            self.add_module("rnn", self.rnn)
        out, _ = self.rnn(x)
        if self.return_sequence:
            return out
        return out[:, -1, :]


def _instantiate(node_type, params):
    """Returns an nn.Module for node types that participate as real layers,
    or None for structural pass-through types handled directly in GraphModule.forward.
    """
    if node_type == "conv1d":
        return nn.LazyConv1d(params["out_channels"], params["kernel_size"],
                              stride=params["stride"], padding=params["padding"],
                              dilation=params["dilation"], groups=params["groups"], bias=params["bias"])
    if node_type == "conv2d":
        return nn.LazyConv2d(params["out_channels"], params["kernel_size"],
                              stride=params["stride"], padding=params["padding"],
                              dilation=params["dilation"], groups=params["groups"], bias=params["bias"])
    if node_type == "maxpool1d":
        return nn.MaxPool1d(params["kernel_size"], stride=params["stride"], padding=params["padding"])
    if node_type == "maxpool2d":
        return nn.MaxPool2d(params["kernel_size"], stride=params["stride"], padding=params["padding"])
    if node_type == "avgpool1d":
        return nn.AvgPool1d(params["kernel_size"], stride=params["stride"], padding=params["padding"])
    if node_type == "avgpool2d":
        return nn.AvgPool2d(params["kernel_size"], stride=params["stride"], padding=params["padding"])
    if node_type == "adaptive_avgpool2d":
        return nn.AdaptiveAvgPool2d(tuple(params["output_size"]))
    if node_type == "batchnorm1d":
        return nn.LazyBatchNorm1d(eps=params["eps"], momentum=params["momentum"], affine=params["affine"])
    if node_type == "batchnorm2d":
        return nn.LazyBatchNorm2d(eps=params["eps"], momentum=params["momentum"], affine=params["affine"])
    if node_type == "dropout":
        return nn.Dropout(p=params["p"])
    if node_type == "flatten":
        return nn.Flatten(start_dim=params["start_dim"])
    if node_type == "linear":
        return nn.LazyLinear(params["out_features"], bias=params["bias"])
    if node_type in ("rnn", "lstm", "gru"):
        return LazyRecurrent(node_type, params["hidden_size"], params["num_layers"],
                              params["bidirectional"], params["dropout"], params["return_sequence"])
    if node_type == "embedding":
        return nn.Embedding(params["num_embeddings"], params["embedding_dim"])
    if node_type == "positional_encoding":
        return PositionalEncoding(params["max_len"], params["d_model"])
    if node_type == "multihead_attention":
        return SelfAttentionBlock(params["embed_dim"], params["num_heads"], params["dropout"])
    if node_type == "transformer_encoder_layer":
        return nn.TransformerEncoderLayer(
            d_model=params["d_model"], nhead=params["nhead"],
            dim_feedforward=params["dim_feedforward"], dropout=params["dropout"],
            activation=params["activation"], batch_first=True,
        )
    if node_type == "layernorm":
        return nn.LayerNorm(params["normalized_shape"], eps=params["eps"])
    if node_type == "relu":
        return nn.ReLU()
    if node_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=params["negative_slope"])
    if node_type == "sigmoid":
        return nn.Sigmoid()
    if node_type == "tanh":
        return nn.Tanh()
    if node_type == "gelu":
        return nn.GELU()
    if node_type == "softmax":
        return nn.Softmax(dim=params["dim"])
    if node_type == "elu":
        return nn.ELU(alpha=params["alpha"])
    # structural pass-through types: input, output, add, concat, reshape
    return None


def _key(node_id):
    return "node__" + "".join(c if c.isalnum() else "_" for c in node_id)


class GraphModule(nn.Module):
    def __init__(self, graph):
        super().__init__()
        nodes = graph["nodes"]
        edges = graph["edges"]
        self.node_by_id = {n["id"]: n for n in nodes}

        incoming = {nid: [] for nid in self.node_by_id}
        outgoing = {nid: [] for nid in self.node_by_id}
        for e in edges:
            outgoing[e["source"]["node"]].append(e)
            incoming[e["target"]["node"]].append(e)

        self.order = topo_sort(self.node_by_id, incoming, outgoing)
        self.incoming_sources = {
            nid: [e["source"]["node"] for e in incoming[nid]] for nid in self.node_by_id
        }

        input_ids = [nid for nid, n in self.node_by_id.items() if n["type"] == "input"]
        output_ids = [nid for nid, n in self.node_by_id.items() if n["type"] == "output"]
        if len(input_ids) != 1 or len(output_ids) != 1:
            raise GraphError("v1 supports exactly one 'input' and one 'output' node per graph")
        self.input_id = input_ids[0]
        self.output_id = output_ids[0]

        self._mods = nn.ModuleDict()
        for nid, n in self.node_by_id.items():
            if n["type"] in CONFIG_TYPES:
                continue
            mod = _instantiate(n["type"], n.get("params", {}))
            if mod is not None:
                self._mods[_key(nid)] = mod

    def _compute_node(self, nid, values):
        node = self.node_by_id[nid]
        ntype = node["type"]
        params = node.get("params", {})
        srcs = self.incoming_sources[nid]
        inputs = [values[s] for s in srcs]

        if ntype == "output":
            return inputs[0]
        if ntype == "add":
            out = inputs[0]
            for t in inputs[1:]:
                out = out + t
            return out
        if ntype == "concat":
            return torch.cat(inputs, dim=params.get("dim", 1))
        if ntype == "reshape":
            shape = params.get("shape", [-1])
            return inputs[0].reshape(inputs[0].size(0), *shape)
        try:
            return self._mods[_key(nid)](inputs[0])
        except Exception as exc:  # noqa: BLE001 - annotate with node id for the UI
            raise GraphError(f"{ntype}: {exc}", nid) from exc

    def forward(self, x):
        values = {self.input_id: x}
        for nid in self.order:
            if nid == self.input_id:
                continue
            values[nid] = self._compute_node(nid, values)
        return values[self.output_id]

    def forward_with_shapes(self, x):
        """Runs the graph recording each node's output shape; stops at the
        first node that raises, returning the error alongside shapes gathered so far.
        """
        values = {self.input_id: x}
        shapes = {self.input_id: tuple(x.shape)}
        for nid in self.order:
            if nid == self.input_id:
                continue
            try:
                values[nid] = self._compute_node(nid, values)
            except GraphError as e:
                return None, shapes, e
            shapes[nid] = tuple(values[nid].shape)
        return values[self.output_id], shapes, None


def make_dummy_input(input_node, batch=2, device="cpu"):
    params = input_node.get("params", {})
    shape = list(params.get("shape", [2]))
    kind = params.get("input_kind", "tensor2d")
    if kind == "sequence":
        return torch.randint(0, 2, (batch, *shape), dtype=torch.long, device=device)
    return torch.randn(batch, *shape, device=device)


def build_module_from_graph(graph):
    """Build a GraphModule and materialize any Lazy* submodules with a dummy pass."""
    module = GraphModule(graph)
    input_node = module.node_by_id[module.input_id]
    dummy = make_dummy_input(input_node)
    module.eval()
    with torch.no_grad():
        module(dummy)
    module.train()
    return module


def config_node(graph, node_type):
    for n in graph["nodes"]:
        if n["type"] == node_type:
            return n
    return None
