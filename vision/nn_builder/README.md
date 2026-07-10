# NN Builder

A drag-and-drop visual neural-network builder: nodes are PyTorch layers/ops
(Conv, Pool, Linear, RNN/LSTM/GRU, Transformer blocks, activations), edges are
tensor connections. The backend turns the graph into a real `torch.nn.Module`,
can train it live (2D toy datasets à la playground.tensorflow.org, MNIST, or a
synthetic sequence task), and can export a standalone `train.py`.

## Run locally (no Docker)

```bash
cd vision/nn_builder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 api_server.py --port 5050
```

Open http://localhost:5050/

## Run with Docker

```bash
./run_nn_builder.sh    # from repo root
./kill_nn_builder.sh
```

or directly:

```bash
cd vision/nn_builder
docker compose up --build
```

or, using the shortcut script in this directory:

```bash
cd vision/nn_builder
./build_run.sh
```

(`build_run.sh` is just `docker compose up --build` in the foreground — it
rebuilds the image if `requirements.txt`/`Dockerfile.x86` changed and streams
logs to the terminal; press Ctrl-C to stop. Edit it to add `-d` if you want it
to run detached instead.)

## Example: building LeNet-5 in the UI

The classic PyTorch tutorial network:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        return self.fc3(f6)
```

maps to this node graph — drag each item from the palette in order, connecting
each node's output dot to the next node's input dot:

| # | Palette item (family) | Params to set |
|---|---|---|
| 1 | **Input** (struct) | `shape = 1, 28, 28`, `input_kind = image` |
| 2 | **Conv2D** (cnn) | `out_channels = 6`, `kernel_size = 5`, `stride = 1`, `padding = 0` |
| 3 | **ReLU** (activation) | — |
| 4 | **MaxPool2D** (cnn) | `kernel_size = 2`, `stride = 2` |
| 5 | **Conv2D** (cnn) | `out_channels = 16`, `kernel_size = 5`, `stride = 1`, `padding = 0` |
| 6 | **ReLU** | — |
| 7 | **MaxPool2D** | `kernel_size = 2`, `stride = 2` |
| 8 | **Flatten** (struct) | `start_dim = 1` |
| 9 | **Linear** (ann) | `out_features = 120` |
| 10 | **ReLU** | — |
| 11 | **Linear** | `out_features = 84` |
| 12 | **ReLU** | — |
| 13 | **Linear** | `out_features = 10` |
| 14 | **Output** (struct) | `task = classification` |

Then add the config nodes anywhere on the canvas — they don't connect via
edges, just need to exist once each: **Dataset** (`kind = mnist`),
**Optimizer** (`kind = adam`, `lr = 0.001`), **Loss** (`kind = cross_entropy`),
**Training Config** (`epochs = ...`).

Two things to know if you're translating from raw PyTorch code like this:

1. **No manual `16 * 5 * 5`.** `Linear`/`Conv2d` nodes build on `nn.LazyLinear`
   / `nn.LazyConv2d` ([builder.py](nn_graph/builder.py)), so you only ever set
   `out_features` / `out_channels` — the input size is inferred automatically
   the first time data flows through the graph.
2. **Image size**: the tutorial assumes a 32×32 input (5×5 feature maps before
   flatten); the built-in `mnist` dataset serves real 28×28 images. With
   `padding=0` as above you'll get 4×4 feature maps instead — harmless, since
   `fc1`'s input size is inferred either way. For the exact 32×32 tutorial
   shapes, set `padding=2` on the first `Conv2D` instead.

Once wired up, click **Validate** to see each node's inferred output shape,
**Build** to confirm the parameter count, or switch to the **Training** tab,
pick `mnist`, and press **Play** to train it live.

## Testing your model: evaluation output

Once a training run finishes (its held-out split — the `1 - train_ratio` slice
of the **Dataset** node — is what gets evaluated), the **Training** tab shows a
**Test-set Evaluation** card for any classification task (2D toy datasets,
MNIST, `sequence_classify`) with:

- **Confusion matrix** — rows = true label, cols = predicted, shaded by count.
- **Precision / Recall / F1** — macro-averaged across classes.
- **ROC-AUC** — binary one-vs-the-other for 2-class problems, macro one-vs-rest
  for multi-class (MNIST); requires `scikit-learn` (already in
  `requirements.txt`) and is reported as `n/a` if a class is entirely absent
  from that particular held-out split.

This is computed server-side in [nn_graph/metrics.py](nn_graph/metrics.py) and
streamed once over `/ws` as `train/eval` right before `train/done`. The
exported standalone `train.py` ([codegen.py](nn_graph/codegen.py)) prints the
same report to stdout after its training loop.

**Not covered:** `sequence_copy` is a per-timestep prediction task, not
single-label classification, so it has no confusion matrix/ROC-AUC. **mAP**
(mean average precision) is an object-detection/retrieval metric — it needs
bounding boxes or ranked retrieval results, neither of which the current
classification-only datasets produce, so it isn't computed. If an object-
detection dataset/head gets added later, mAP belongs in `metrics.py` alongside
these.

For CNN/image graphs specifically: the confusion matrix is the main tool for
spotting *which* classes get confused (e.g. digit 4 vs 9), which a single
accuracy number hides.

## Early stopping, regularization & other training controls

The **Training Config** node (in the Training tab's hyperparameter panel) has:

- `early_stopping` (bool) — when on, training stops once `val_loss` hasn't
  improved by at least `early_stopping_min_delta` for `early_stopping_patience`
  epochs in a row. The `train/done` message reports `early_stopped: true` and
  the epoch it stopped at.
- `grad_clip_norm` — if > 0, clips gradients to this max norm
  (`torch.nn.utils.clip_grad_norm_`) before each optimizer step. Useful for
  RNN/Transformer graphs prone to exploding gradients.

Regularization already available as ordinary nodes/params — nothing extra to
configure beyond dropping them into the graph:

- **Dropout** node — set `p` (drop probability), anywhere in the graph.
- **BatchNorm1D/2D** nodes — normalizes activations, often reduces the need
  for aggressive dropout.
- **Optimizer** node's `weight_decay` — L2 regularization, supported by every
  optimizer kind (`sgd`/`adam`/`adamw`/`rmsprop`).
- **`grad_clip_norm`** above — not regularization in the statistical sense,
  but the usual companion control for stable training on RNN/Transformer
  graphs, so it lives on the same node.

### Other things worth knowing about as you go further

- **Learning-rate schedules** (step decay, cosine, warmup) aren't exposed yet —
  currently the `Optimizer` node's `lr` is fixed for the whole run. Would live
  as a new field on `train_config` plus a `torch.optim.lr_scheduler` call in
  `trainer.py`/`codegen.py` if you need it.
- **Data augmentation** (random crop/flip for images) isn't in `datasets.py`
  yet — the MNIST loader uses a plain `ToTensor()` transform.
- **Class-imbalance handling** (weighted loss, oversampling) isn't automatic —
  if a dataset's classes are imbalanced, pass `weight=` into the loss node's
  underlying `nn.CrossEntropyLoss` manually today (a `class_weights` param on
  the `loss` node would be the natural place to add it).

## Layout

- `nn_graph/catalog.py` — single source of truth for every node type + params (served via `GET /api/catalog`).
- `nn_graph/schema.py` — structural graph validation (DAG, arity, required config nodes).
- `nn_graph/builder.py` — builds a real `nn.Module` from the graph (lazy in-dim inference).
- `nn_graph/validation.py` — dry-run shape inference, per-node error reporting.
- `nn_graph/datasets.py` — 2D toy datasets (circle/xor/gaussian/spiral, matching TF Playground), MNIST, synthetic sequence tasks.
- `nn_graph/metrics.py` — post-training evaluation: confusion matrix, precision/recall/F1, ROC-AUC.
- `nn_graph/trainer.py` — background-thread training session (play/pause/step/stop, early stopping, grad clipping) streaming metrics over `/ws`.
- `nn_graph/codegen.py` — exports a standalone, dependency-free `train.py`.
- `api_server.py` — Flask + flask-sock REST/WS backend.
- `ui/` — vanilla JS (no build step) node-graph editor + training dashboard, styled to match `race_nav/server/agv_dashboard`.
- `saved_graphs/` — server-side saved architectures (gitignored).
