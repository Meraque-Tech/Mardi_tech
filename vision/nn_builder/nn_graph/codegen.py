"""Generates a standalone, dependency-free-of-this-package train.py from a
validated graph JSON + training config. The generated script only imports
torch/torchvision/numpy and can run with no server/nn_builder package present.
"""

from .builder import GraphModule, build_module_from_graph, config_node, _key
from .catalog import CONFIG_TYPES

_ACTIVATION_CTORS = {
    "relu": lambda p: "nn.ReLU()",
    "sigmoid": lambda p: "nn.Sigmoid()",
    "tanh": lambda p: "nn.Tanh()",
    "gelu": lambda p: "nn.GELU()",
    "leaky_relu": lambda p: f"nn.LeakyReLU(negative_slope={p['negative_slope']})",
    "softmax": lambda p: f"nn.Softmax(dim={p['dim']})",
    "elu": lambda p: f"nn.ELU(alpha={p['alpha']})",
}


def _var(nid):
    return "x_" + "".join(c if c.isalnum() else "_" for c in nid)


def _attr(nid):
    return "self." + "".join(c if c.isalnum() else "_" for c in nid)


def _ctor_line(nid, node, materialized_mod):
    ntype = node["type"]
    p = node.get("params", {})

    if ntype in _ACTIVATION_CTORS:
        return f"{_attr(nid)} = {_ACTIVATION_CTORS[ntype](p)}"
    if ntype == "flatten":
        return f"{_attr(nid)} = nn.Flatten(start_dim={p['start_dim']})"
    if ntype == "dropout":
        return f"{_attr(nid)} = nn.Dropout(p={p['p']})"
    if ntype == "maxpool1d":
        return f"{_attr(nid)} = nn.MaxPool1d(kernel_size={p['kernel_size']}, stride={p['stride']}, padding={p['padding']})"
    if ntype == "maxpool2d":
        return f"{_attr(nid)} = nn.MaxPool2d(kernel_size={p['kernel_size']}, stride={p['stride']}, padding={p['padding']})"
    if ntype == "avgpool1d":
        return f"{_attr(nid)} = nn.AvgPool1d(kernel_size={p['kernel_size']}, stride={p['stride']}, padding={p['padding']})"
    if ntype == "avgpool2d":
        return f"{_attr(nid)} = nn.AvgPool2d(kernel_size={p['kernel_size']}, stride={p['stride']}, padding={p['padding']})"
    if ntype == "adaptive_avgpool2d":
        return f"{_attr(nid)} = nn.AdaptiveAvgPool2d(output_size={tuple(p['output_size'])})"
    if ntype == "conv1d":
        in_ch = materialized_mod.in_channels
        return (f"{_attr(nid)} = nn.Conv1d({in_ch}, {p['out_channels']}, kernel_size={p['kernel_size']}, "
                f"stride={p['stride']}, padding={p['padding']}, dilation={p['dilation']}, "
                f"groups={p['groups']}, bias={p['bias']})")
    if ntype == "conv2d":
        in_ch = materialized_mod.in_channels
        return (f"{_attr(nid)} = nn.Conv2d({in_ch}, {p['out_channels']}, kernel_size={p['kernel_size']}, "
                f"stride={p['stride']}, padding={p['padding']}, dilation={p['dilation']}, "
                f"groups={p['groups']}, bias={p['bias']})")
    if ntype == "batchnorm1d":
        return f"{_attr(nid)} = nn.BatchNorm1d({materialized_mod.num_features}, eps={p['eps']}, momentum={p['momentum']}, affine={p['affine']})"
    if ntype == "batchnorm2d":
        return f"{_attr(nid)} = nn.BatchNorm2d({materialized_mod.num_features}, eps={p['eps']}, momentum={p['momentum']}, affine={p['affine']})"
    if ntype == "linear":
        return f"{_attr(nid)} = nn.Linear({materialized_mod.in_features}, {p['out_features']}, bias={p['bias']})"
    if ntype in ("rnn", "lstm", "gru"):
        cls = {"rnn": "nn.RNN", "lstm": "nn.LSTM", "gru": "nn.GRU"}[ntype]
        input_size = materialized_mod.rnn.input_size
        dropout = p["dropout"] if p["num_layers"] > 1 else 0.0
        return (f"{_attr(nid)} = {cls}({input_size}, {p['hidden_size']}, num_layers={p['num_layers']}, "
                f"batch_first=True, bidirectional={p['bidirectional']}, dropout={dropout})")
    if ntype == "embedding":
        return f"{_attr(nid)} = nn.Embedding({p['num_embeddings']}, {p['embedding_dim']})"
    if ntype == "positional_encoding":
        return f"{_attr(nid)}_pe = sinusoidal_pe({p['max_len']}, {p['d_model']})"
    if ntype == "multihead_attention":
        return f"{_attr(nid)} = nn.MultiheadAttention({p['embed_dim']}, {p['num_heads']}, dropout={p['dropout']}, batch_first=True)"
    if ntype == "transformer_encoder_layer":
        return (f"{_attr(nid)} = nn.TransformerEncoderLayer(d_model={p['d_model']}, nhead={p['nhead']}, "
                f"dim_feedforward={p['dim_feedforward']}, dropout={p['dropout']}, "
                f"activation='{p['activation']}', batch_first=True)")
    if ntype == "layernorm":
        return f"{_attr(nid)} = nn.LayerNorm({p['normalized_shape']}, eps={p['eps']})"
    return None  # input / output / add / concat / reshape: no submodule


def _forward_line(nid, node, srcs):
    ntype = node["type"]
    p = node.get("params", {})
    out = _var(nid)
    ins = [_var(s) for s in srcs]

    if ntype == "output":
        return f"{out} = {ins[0]}"
    if ntype == "add":
        return f"{out} = " + " + ".join(ins)
    if ntype == "concat":
        return f"{out} = torch.cat([{', '.join(ins)}], dim={p.get('dim', 1)})"
    if ntype == "reshape":
        return f"{out} = {ins[0]}.reshape({ins[0]}.size(0), {', '.join(str(s) for s in p.get('shape', [-1]))})"
    if ntype in ("rnn", "lstm", "gru"):
        seq_var = f"_seq_{out}"
        line1 = f"{seq_var}, _ = {_attr(nid)}({ins[0]})"
        if p.get("return_sequence"):
            line2 = f"{out} = {seq_var}"
        else:
            line2 = f"{out} = {seq_var}[:, -1, :]"
        return line1 + "\n        " + line2
    if ntype == "multihead_attention":
        return f"{out}, _ = {_attr(nid)}({ins[0]}, {ins[0]}, {ins[0]})"
    if ntype == "positional_encoding":
        return f"{out} = {ins[0]} + {_attr(nid)}_pe[:, :{ins[0]}.size(1), :]"
    return f"{out} = {_attr(nid)}({ins[0]})"


_STANDALONE_DATA_AND_TRAIN_HELPERS = '''
def _circle(n, noise, rng):
    radius = 5.0
    half = n // 2
    X, y = [], []
    def lbl(x, y_):
        return 1 if (x * x + y_ * y_) < (radius * 0.5) ** 2 else 0
    for lo, hi in ((0, radius * 0.5), (radius * 0.7, radius)):
        for _ in range(half if lo == 0 else n - half):
            r = rng.uniform(lo, hi)
            a = rng.uniform(0, 2 * np.pi)
            x_, y_ = r * np.sin(a), r * np.cos(a)
            nx, ny = rng.uniform(-radius, radius) * noise / 3, rng.uniform(-radius, radius) * noise / 3
            X.append([x_, y_]); y.append(lbl(x_ + nx, y_ + ny))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def _xor(n, noise, rng):
    X, y = [], []
    pad = 0.3
    for _ in range(n):
        x_ = rng.uniform(-5, 5); x_ += pad if x_ > 0 else -pad
        y_ = rng.uniform(-5, 5); y_ += pad if y_ > 0 else -pad
        nx, ny = rng.uniform(-5, 5) * noise, rng.uniform(-5, 5) * noise
        X.append([x_, y_]); y.append(1 if (x_ + nx) * (y_ + ny) >= 0 else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def _gaussian(n, noise, rng):
    variance = np.interp(noise, [0, 0.5], [0.5, 4.0])
    half = n // 2
    X, y = [], []
    for cx, cy, label in ((2, 2, 1), (-2, -2, 0)):
        pts = rng.normal(loc=[cx, cy], scale=variance, size=(half, 2))
        X.append(pts); y.append(np.full(half, label, dtype=np.int64))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)

def _spiral(n, noise, rng):
    half = n // 2
    X, y = [], []
    for delta_t, label in ((0.0, 1), (np.pi, 0)):
        for i in range(half):
            r = i / half * 5
            t = 1.75 * i / half * 2 * np.pi + delta_t
            x_ = r * np.sin(t) + rng.uniform(-1, 1) * noise
            y_ = r * np.cos(t) + rng.uniform(-1, 1) * noise
            X.append([x_, y_]); y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

_TOY_2D_FNS = {"circle": _circle, "xor": _xor, "gaussian": _gaussian, "spiral": _spiral}

def _make_sequence_dataset(kind, n, seq_len, vocab_size, rng):
    vocab_size = max(2, vocab_size)
    X = rng.randint(0, vocab_size, size=(n, seq_len)).astype(np.int64)
    if kind == "sequence_copy":
        y = X.copy()
    else:
        odd = (X % 2).sum(axis=1)
        y = (odd > (seq_len / 2)).astype(np.int64)
    return X, y

def _build_dataset(cfg):
    from torch.utils.data import DataLoader, TensorDataset, Subset
    kind = cfg.get("kind", "circle")
    batch_size = int(cfg.get("batch_size", 32))
    train_ratio = float(cfg.get("train_ratio", 0.5))
    seed = int(cfg.get("seed", 0))
    rng = np.random.RandomState(seed)

    if kind in _TOY_2D_FNS:
        X, y = _TOY_2D_FNS[kind](int(cfg.get("num_samples", 500)), float(cfg.get("noise", 0.0)), rng)
        # X/y are generated class-contiguous (e.g. spiral: all class 1, then all
        # class 0) -- shuffle before splitting or the held-out set can end up
        # single-class, which breaks accuracy/confusion-matrix/ROC-AUC on it.
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]
        n_train = int(len(X) * train_ratio)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        train_ds = TensorDataset(Xt[:n_train], yt[:n_train])
        val_ds = TensorDataset(Xt[n_train:], yt[n_train:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None
        return train_loader, val_loader, "2d", (-6, 6)

    if kind == "mnist":
        from torchvision import datasets, transforms
        full = datasets.MNIST(root=".cache", train=True, download=True, transform=transforms.ToTensor())
        num_samples = min(int(cfg.get("num_samples", 2000)), len(full))
        idx = rng.choice(len(full), size=num_samples, replace=False)
        n_train = int(num_samples * train_ratio)
        train_loader = DataLoader(Subset(full, idx[:n_train].tolist()), batch_size=batch_size, shuffle=True)
        val_ds = Subset(full, idx[n_train:].tolist())
        val_loader = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) else None
        return train_loader, val_loader, "image", None

    seq_len, vocab_size = int(cfg.get("seq_len", 16)), int(cfg.get("vocab_size", 10))
    X, y = _make_sequence_dataset(kind, int(cfg.get("num_samples", 500)), seq_len, vocab_size, rng)
    Xt, yt = torch.from_numpy(X).long(), torch.from_numpy(y).long()
    n_train = int(len(X) * train_ratio)
    train_loader = DataLoader(TensorDataset(Xt[:n_train], yt[:n_train]), batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(Xt[n_train:], yt[n_train:])
    val_loader = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) else None
    return train_loader, val_loader, kind, None

def _make_optimizer(params, cfg):
    kind = cfg.get("kind", "adam")
    lr, wd, mom = float(cfg.get("lr", 0.001)), float(cfg.get("weight_decay", 0.0)), float(cfg.get("momentum", 0.9))
    if kind == "sgd":
        return optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    if kind == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=mom, weight_decay=wd)
    return optim.Adam(params, lr=lr, weight_decay=wd)

def _make_loss(cfg):
    kind = cfg.get("kind", "cross_entropy")
    if kind == "mse":
        return nn.MSELoss()
    if kind == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    if kind == "nll":
        return nn.NLLLoss()
    return nn.CrossEntropyLoss()

def _compute_loss(loss_fn, pred, target, task):
    if task == "sequence_copy":
        return loss_fn(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    if isinstance(loss_fn, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        p = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
        return loss_fn(p, target.float())
    return loss_fn(pred, target)

def _compute_accuracy(pred, target, task):
    with torch.no_grad():
        if task == "sequence_copy":
            return (pred.argmax(dim=-1) == target).float().mean().item()
        if pred.dim() > 1 and pred.size(-1) > 1:
            labels = pred.argmax(dim=-1)
        else:
            labels = (torch.sigmoid(pred.squeeze(-1)) > 0.5).long()
        return (labels == target).float().mean().item()

def _evaluate(model, loss_fn, loader, task):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total_loss += _compute_loss(loss_fn, pred, yb, task).item()
            total_acc += _compute_accuracy(pred, yb, task)
            n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)

_CLASSIFICATION_TASKS = {"2d", "image", "sequence_classify"}

def _print_classification_report(model, loader, task):
    """Confusion matrix + precision/recall/F1 (+ ROC-AUC if scikit-learn is
    installed) over the held-out split. Not computed for sequence_copy (a
    per-timestep task, not single-label classification). mAP is an object-
    detection/retrieval metric and doesn't apply to these datasets."""
    if task not in _CLASSIFICATION_TASKS:
        return
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            if pred.dim() > 1 and pred.size(-1) > 1:
                prob = torch.softmax(pred, dim=-1)
                labels = prob.argmax(dim=-1)
            else:
                p1 = torch.sigmoid(pred.squeeze(-1))
                prob = torch.stack([1 - p1, p1], dim=-1)
                labels = (p1 > 0.5).long()
            y_true.append(yb.numpy()); y_pred.append(labels.numpy()); y_prob.append(prob.numpy())
    model.train()
    if not y_true:
        return
    y_true, y_pred, y_prob = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)
    num_classes = y_prob.shape[-1]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print("\\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    print(f"precision(macro)={np.mean(precisions):.4f}  recall(macro)={np.mean(recalls):.4f}  f1(macro)={np.mean(f1s):.4f}")

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes)))
        print(f"roc_auc(macro)={auc:.4f}")

        aps = []
        for c in range(num_classes):
            y_true_c = (y_true == c).astype(int)
            if 0 < y_true_c.sum() < len(y_true_c):
                aps.append(average_precision_score(y_true_c, y_prob[:, c]))
        if aps:
            print(f"mAP (mean per-class average precision)={np.mean(aps):.4f}")
    except (ImportError, ValueError):
        print("roc_auc/mAP: unavailable (install scikit-learn, or too few classes present in this split)")
'''

_HELPERS = '''
def sinusoidal_pe(max_len, d_model):
    import math
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)
'''


def graph_to_train_py(graph, train_config=None):
    """Returns the full standalone train.py source as a string."""
    module = build_module_from_graph(graph)  # materializes Lazy* modules so we can read concrete dims

    ctor_lines = []
    forward_lines = []
    needs_pe_helper = False

    for nid in module.order:
        node = module.node_by_id[nid]
        if node["type"] in CONFIG_TYPES:
            continue
        mkey = _key(nid)
        mod = module._mods[mkey] if mkey in module._mods else None  # noqa: SLF001 - internal but same-package
        ctor = _ctor_line(nid, node, mod)
        if ctor:
            ctor_lines.append("        " + ctor)
        if node["type"] == "positional_encoding":
            needs_pe_helper = True
        if nid == module.input_id:
            continue
        srcs = module.incoming_sources[nid]
        forward_lines.append("        " + _forward_line(nid, node, srcs))

    dataset_node = config_node(graph, "dataset")
    optimizer_node = config_node(graph, "optimizer")
    loss_node = config_node(graph, "loss")
    tc_node = config_node(graph, "train_config")
    input_node = module.node_by_id[module.input_id]

    src = f'''"""Auto-generated by nn_builder — standalone, runnable with `python3 train.py`.
Requires: torch, torchvision, numpy.
"""
import math
import numpy as np
import torch
from torch import nn, optim
{_HELPERS if needs_pe_helper else ""}

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
{chr(10).join(ctor_lines) if ctor_lines else "        pass"}

    def forward(self, {_var(module.input_id)}):
{chr(10).join(forward_lines)}
        return {_var(module.output_id)}


INPUT_SHAPE = {input_node.get("params", {}).get("shape", [2])}
INPUT_KIND = "{input_node.get("params", {}).get("input_kind", "tensor2d")}"
DATASET_CFG = {dataset_node["params"] if dataset_node else {}}
OPTIMIZER_CFG = {optimizer_node["params"] if optimizer_node else {}}
LOSS_CFG = {loss_node["params"] if loss_node else {}}
TRAIN_CFG = {tc_node["params"] if tc_node else {"epochs": 20}}

{_STANDALONE_DATA_AND_TRAIN_HELPERS}

def main():
    model = GeneratedModel()
    print(model)
    print(f"Trainable parameters: {{sum(p.numel() for p in model.parameters() if p.requires_grad)}}")
    train_loader, val_loader, task, x_range = _build_dataset(DATASET_CFG)
    optimizer = _make_optimizer(model.parameters(), OPTIMIZER_CFG)
    loss_fn = _make_loss(LOSS_CFG)
    epochs = int(TRAIN_CFG.get("epochs", 20))
    grad_clip_norm = float(TRAIN_CFG.get("grad_clip_norm", 0.0))
    early_stopping = bool(TRAIN_CFG.get("early_stopping", False))
    patience = int(TRAIN_CFG.get("early_stopping_patience", 5))
    min_delta = float(TRAIN_CFG.get("early_stopping_min_delta", 0.0001))

    best_val_loss, epochs_no_improve = float("inf"), 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = _compute_loss(loss_fn, pred, yb, task)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            total_acc += _compute_accuracy(pred, yb, task)
            n += 1

        msg = f"epoch {{epoch+1}}/{{epochs}}  loss={{total_loss/max(n,1):.4f}}  acc={{total_acc/max(n,1):.4f}}"
        val_loss = None
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, loss_fn, val_loader, task)
            msg += f"  val_loss={{val_loss:.4f}}  val_acc={{val_acc:.4f}}"
        print(msg)

        if early_stopping and val_loss is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss, epochs_no_improve = val_loss, 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {{epoch+1}} (no val_loss improvement for {{patience}} epochs)")
                    break

    print("\\n--- Test-set evaluation (held-out split) ---")
    if val_loader is not None:
        _print_classification_report(model, val_loader, task)
    else:
        print("No held-out split available (train_ratio was 1.0)")


if __name__ == "__main__":
    main()
'''
    return src
