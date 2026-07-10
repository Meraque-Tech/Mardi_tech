"""Single source of truth for every node type the builder supports.

Served verbatim to the frontend via GET /api/catalog, and consumed by
builder.py to know each node's torch module, arity, and constructor params.
Do not duplicate this list anywhere else.
"""

P_INT = "int"
P_FLOAT = "float"
P_BOOL = "bool"
P_ENUM = "enum"
P_LIST_INT = "list_int"


def _p(name, type_, default, **kw):
    d = {"name": name, "type": type_, "default": default}
    d.update(kw)
    return d


NODE_CATALOG = {
    # ---- structural ----
    "input": {
        "family": "struct", "label": "Input", "torch_module": None,
        "inputs": 0, "outputs": 1,
        "params": [
            _p("shape", P_LIST_INT, [2]),
            _p("input_kind", P_ENUM, "tensor2d", options=["tensor2d", "image", "sequence"]),
        ],
    },
    "output": {
        "family": "struct", "label": "Output", "torch_module": None,
        "inputs": 1, "outputs": 0,
        "params": [
            _p("task", P_ENUM, "classification", options=["classification", "regression"]),
        ],
    },
    "add": {
        "family": "struct", "label": "Add (Residual)", "torch_module": None,
        "inputs": -1, "outputs": 1, "min_inputs": 2,
        "params": [],
    },
    "concat": {
        "family": "struct", "label": "Concat", "torch_module": None,
        "inputs": -1, "outputs": 1, "min_inputs": 2,
        "params": [_p("dim", P_INT, 1, min=0, max=4)],
    },
    "reshape": {
        "family": "struct", "label": "Reshape", "torch_module": None,
        "inputs": 1, "outputs": 1,
        "params": [_p("shape", P_LIST_INT, [-1])],
    },
    "flatten": {
        "family": "struct", "label": "Flatten", "torch_module": "nn.Flatten",
        "inputs": 1, "outputs": 1,
        "params": [_p("start_dim", P_INT, 1, min=0, max=4)],
    },

    # ---- cnn ----
    "conv1d": {
        "family": "cnn", "label": "Conv1D", "torch_module": "nn.Conv1d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("out_channels", P_INT, 16, min=1, max=1024),
            _p("kernel_size", P_INT, 3, min=1, max=15),
            _p("stride", P_INT, 1, min=1, max=8),
            _p("padding", P_INT, 1, min=0, max=8),
            _p("dilation", P_INT, 1, min=1, max=8),
            _p("groups", P_INT, 1, min=1, max=64),
            _p("bias", P_BOOL, True),
        ],
    },
    "conv2d": {
        "family": "cnn", "label": "Conv2D", "torch_module": "nn.Conv2d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("out_channels", P_INT, 16, min=1, max=1024),
            _p("kernel_size", P_INT, 3, min=1, max=15),
            _p("stride", P_INT, 1, min=1, max=8),
            _p("padding", P_INT, 1, min=0, max=8),
            _p("dilation", P_INT, 1, min=1, max=8),
            _p("groups", P_INT, 1, min=1, max=64),
            _p("bias", P_BOOL, True),
        ],
    },
    "maxpool1d": {
        "family": "cnn", "label": "MaxPool1D", "torch_module": "nn.MaxPool1d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("kernel_size", P_INT, 2, min=1, max=8),
            _p("stride", P_INT, 2, min=1, max=8),
            _p("padding", P_INT, 0, min=0, max=4),
        ],
    },
    "maxpool2d": {
        "family": "cnn", "label": "MaxPool2D", "torch_module": "nn.MaxPool2d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("kernel_size", P_INT, 2, min=1, max=8),
            _p("stride", P_INT, 2, min=1, max=8),
            _p("padding", P_INT, 0, min=0, max=4),
        ],
    },
    "avgpool1d": {
        "family": "cnn", "label": "AvgPool1D", "torch_module": "nn.AvgPool1d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("kernel_size", P_INT, 2, min=1, max=8),
            _p("stride", P_INT, 2, min=1, max=8),
            _p("padding", P_INT, 0, min=0, max=4),
        ],
    },
    "avgpool2d": {
        "family": "cnn", "label": "AvgPool2D", "torch_module": "nn.AvgPool2d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("kernel_size", P_INT, 2, min=1, max=8),
            _p("stride", P_INT, 2, min=1, max=8),
            _p("padding", P_INT, 0, min=0, max=4),
        ],
    },
    "adaptive_avgpool2d": {
        "family": "cnn", "label": "AdaptiveAvgPool2D", "torch_module": "nn.AdaptiveAvgPool2d",
        "inputs": 1, "outputs": 1,
        "params": [_p("output_size", P_LIST_INT, [1, 1])],
    },
    "batchnorm1d": {
        "family": "cnn", "label": "BatchNorm1D", "torch_module": "nn.BatchNorm1d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("eps", P_FLOAT, 1e-5, min=1e-8, max=1e-2),
            _p("momentum", P_FLOAT, 0.1, min=0.0, max=1.0),
            _p("affine", P_BOOL, True),
        ],
    },
    "batchnorm2d": {
        "family": "cnn", "label": "BatchNorm2D", "torch_module": "nn.BatchNorm2d",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("eps", P_FLOAT, 1e-5, min=1e-8, max=1e-2),
            _p("momentum", P_FLOAT, 0.1, min=0.0, max=1.0),
            _p("affine", P_BOOL, True),
        ],
    },
    "dropout": {
        "family": "cnn", "label": "Dropout", "torch_module": "nn.Dropout",
        "inputs": 1, "outputs": 1,
        "params": [_p("p", P_FLOAT, 0.5, min=0.0, max=0.95)],
    },

    # ---- ann ----
    "linear": {
        "family": "ann", "label": "Linear", "torch_module": "nn.Linear",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("out_features", P_INT, 8, min=1, max=4096),
            _p("bias", P_BOOL, True),
        ],
    },

    # ---- rnn ----
    "rnn": {
        "family": "rnn", "label": "RNN", "torch_module": "nn.RNN",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("hidden_size", P_INT, 32, min=1, max=1024),
            _p("num_layers", P_INT, 1, min=1, max=8),
            _p("bidirectional", P_BOOL, False),
            _p("dropout", P_FLOAT, 0.0, min=0.0, max=0.9),
            _p("return_sequence", P_BOOL, False),
        ],
    },
    "lstm": {
        "family": "rnn", "label": "LSTM", "torch_module": "nn.LSTM",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("hidden_size", P_INT, 32, min=1, max=1024),
            _p("num_layers", P_INT, 1, min=1, max=8),
            _p("bidirectional", P_BOOL, False),
            _p("dropout", P_FLOAT, 0.0, min=0.0, max=0.9),
            _p("return_sequence", P_BOOL, False),
        ],
    },
    "gru": {
        "family": "rnn", "label": "GRU", "torch_module": "nn.GRU",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("hidden_size", P_INT, 32, min=1, max=1024),
            _p("num_layers", P_INT, 1, min=1, max=8),
            _p("bidirectional", P_BOOL, False),
            _p("dropout", P_FLOAT, 0.0, min=0.0, max=0.9),
            _p("return_sequence", P_BOOL, False),
        ],
    },

    # ---- transformer ----
    "embedding": {
        "family": "transformer", "label": "Embedding", "torch_module": "nn.Embedding",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("num_embeddings", P_INT, 256, min=1, max=100000),
            _p("embedding_dim", P_INT, 32, min=1, max=1024),
        ],
    },
    "positional_encoding": {
        "family": "transformer", "label": "Positional Encoding", "torch_module": None,
        "inputs": 1, "outputs": 1,
        "params": [
            _p("max_len", P_INT, 512, min=1, max=8192),
            _p("d_model", P_INT, 32, min=1, max=1024),
        ],
    },
    "multihead_attention": {
        "family": "transformer", "label": "Multi-Head Attention", "torch_module": "nn.MultiheadAttention",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("embed_dim", P_INT, 32, min=1, max=1024),
            _p("num_heads", P_INT, 4, min=1, max=32),
            _p("dropout", P_FLOAT, 0.0, min=0.0, max=0.9),
        ],
    },
    "transformer_encoder_layer": {
        "family": "transformer", "label": "Transformer Encoder Layer", "torch_module": "nn.TransformerEncoderLayer",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("d_model", P_INT, 32, min=1, max=1024),
            _p("nhead", P_INT, 4, min=1, max=32),
            _p("dim_feedforward", P_INT, 128, min=1, max=4096),
            _p("dropout", P_FLOAT, 0.1, min=0.0, max=0.9),
            _p("activation", P_ENUM, "relu", options=["relu", "gelu"]),
        ],
    },
    "layernorm": {
        "family": "transformer", "label": "LayerNorm", "torch_module": "nn.LayerNorm",
        "inputs": 1, "outputs": 1,
        "params": [
            _p("normalized_shape", P_INT, 32, min=1, max=4096),
            _p("eps", P_FLOAT, 1e-5, min=1e-8, max=1e-2),
        ],
    },

    # ---- activations ----
    "relu": {"family": "activation", "label": "ReLU", "torch_module": "nn.ReLU", "inputs": 1, "outputs": 1, "params": []},
    "leaky_relu": {
        "family": "activation", "label": "LeakyReLU", "torch_module": "nn.LeakyReLU",
        "inputs": 1, "outputs": 1,
        "params": [_p("negative_slope", P_FLOAT, 0.01, min=0.0, max=1.0)],
    },
    "sigmoid": {"family": "activation", "label": "Sigmoid", "torch_module": "nn.Sigmoid", "inputs": 1, "outputs": 1, "params": []},
    "tanh": {"family": "activation", "label": "Tanh", "torch_module": "nn.Tanh", "inputs": 1, "outputs": 1, "params": []},
    "gelu": {"family": "activation", "label": "GELU", "torch_module": "nn.GELU", "inputs": 1, "outputs": 1, "params": []},
    "softmax": {
        "family": "activation", "label": "Softmax", "torch_module": "nn.Softmax",
        "inputs": 1, "outputs": 1,
        "params": [_p("dim", P_INT, -1, min=-4, max=4)],
    },
    "elu": {
        "family": "activation", "label": "ELU", "torch_module": "nn.ELU",
        "inputs": 1, "outputs": 1,
        "params": [_p("alpha", P_FLOAT, 1.0, min=0.0, max=10.0)],
    },

    # ---- config pseudo-nodes (edge-less, no forward participation) ----
    "optimizer": {
        "family": "config", "label": "Optimizer", "torch_module": None,
        "inputs": 0, "outputs": 0,
        "params": [
            _p("kind", P_ENUM, "adam", options=["sgd", "adam", "adamw", "rmsprop"]),
            _p("lr", P_FLOAT, 0.001, min=1e-6, max=1.0),
            _p("weight_decay", P_FLOAT, 0.0, min=0.0, max=1.0),
            _p("momentum", P_FLOAT, 0.9, min=0.0, max=1.0),
        ],
    },
    "loss": {
        "family": "config", "label": "Loss", "torch_module": None,
        "inputs": 0, "outputs": 0,
        "params": [
            _p("kind", P_ENUM, "cross_entropy", options=["cross_entropy", "mse", "bce_with_logits", "nll"]),
        ],
    },
    "dataset": {
        "family": "config", "label": "Dataset", "torch_module": None,
        "inputs": 0, "outputs": 0,
        "params": [
            _p("kind", P_ENUM, "circle", options=[
                "circle", "xor", "gaussian", "spiral", "mnist",
                "sequence_copy", "sequence_classify",
            ]),
            _p("noise", P_FLOAT, 0.0, min=0.0, max=0.5),
            _p("num_samples", P_INT, 500, min=10, max=60000),
            _p("batch_size", P_INT, 32, min=1, max=1024),
            _p("train_ratio", P_FLOAT, 0.5, min=0.1, max=0.9),
            _p("seq_len", P_INT, 16, min=2, max=256),
            _p("vocab_size", P_INT, 10, min=2, max=1000),
        ],
    },
    "train_config": {
        "family": "config", "label": "Training Config", "torch_module": None,
        "inputs": 0, "outputs": 0,
        "params": [
            _p("epochs", P_INT, 20, min=1, max=1000),
            _p("seed", P_INT, 0, min=0, max=2**31 - 1),
            _p("progress_every", P_INT, 1, min=1, max=200),
            _p("early_stopping", P_BOOL, False),
            _p("early_stopping_patience", P_INT, 5, min=1, max=200),
            _p("early_stopping_min_delta", P_FLOAT, 0.0001, min=0.0, max=1.0),
            _p("grad_clip_norm", P_FLOAT, 0.0, min=0.0, max=100.0),
        ],
    },
}

STRUCTURAL_ONLY = {"input", "output", "add", "concat", "reshape"}
CONFIG_TYPES = {"optimizer", "loss", "dataset", "train_config"}
FORWARD_TYPES = set(NODE_CATALOG.keys()) - CONFIG_TYPES


def families():
    out = {}
    for type_, spec in NODE_CATALOG.items():
        out.setdefault(spec["family"], []).append(type_)
    return out


def catalog_payload():
    return {"nodes": NODE_CATALOG, "families": families()}
