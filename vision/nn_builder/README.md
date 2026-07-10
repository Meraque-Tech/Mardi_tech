# NN Builder

A drag-and-drop visual neural-network builder: nodes are PyTorch layers/ops
(Conv, Pool, Linear, RNN/LSTM/GRU, Transformer blocks, activations), edges are
tensor connections. The backend turns the graph into a real `torch.nn.Module`,
can train it live (2D toy datasets à la playground.tensorflow.org, MNIST, or a
synthetic sequence task), and can export a standalone `train.py`. A separate
**Detect** tab covers pretrained object detection (YOLOv8/YOLO11 via
`ultralytics`) — see "Object detection" below.

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
- **ROC curve + AUC** — the actual FPR-vs-TPR curve (not just the scalar), with
  a dashed random-chance baseline. Binary problems get one curve; multi-class
  (e.g. MNIST) gets one curve per class if there are ≤6 classes, or a single
  macro-averaged curve above that (10 overlapping ROC curves for MNIST's
  digits would just be spaghetti — see `references/anti-patterns.md` in the
  dataviz skill on why identity-encoded series don't scale past a handful).
- **mAP** — mean Average Precision the *classification* way: per-class area
  under the one-vs-rest precision-recall curve, averaged over classes (the
  same computation multi-label/classification benchmarks like PASCAL VOC's
  classification task report as "mAP"). This is *not* detection mAP (which
  averages AP over IoU-matched bounding boxes) — this tool has no detection
  dataset/head, so only the classification variant is offered. Both need
  `scikit-learn` (already in `requirements.txt`) and report `n/a`/are skipped
  if a class is entirely absent from that particular held-out split.
- **Sample Predictions** (image tasks only, e.g. MNIST) — an actual gallery of
  held-out digit images with their predicted label, bordered green if correct
  and red if wrong (with the true label shown alongside when it's wrong). This
  is the "let me actually see what the model classified" view — the numbers
  above tell you *how much* it got wrong, this shows *what*.

There's also a standalone **Test** button next to Play/Pause/Step/Stop that
re-runs this same evaluation against whatever model is currently built,
without retraining. Each click draws a **brand-new random batch** from the
same dataset config (new noise/points for the 2D toy sets, a fresh random
MNIST subset, new random sequences) — it deliberately does *not* reuse the
fixed held-out split from training, so repeated clicks give you a real sense
of how the model generalizes rather than re-showing the same numbers. (The
one automatic evaluation that fires right after training finishes is the
exception — that one *does* use the training run's actual held-out split,
matching normal "evaluate on the val set" convention.) It calls
`POST /api/train/evaluate`, which reuses the last-built model kept on the
`TrainingSession` (`nn_graph/trainer.py`'s `evaluate_now(fresh=True)`); it
errors cleanly (no crash) if nothing has been trained yet this session.

This is computed server-side in [nn_graph/metrics.py](nn_graph/metrics.py) and
streamed once over `/ws` as `train/eval` right before `train/done`. The
exported standalone `train.py` ([codegen.py](nn_graph/codegen.py)) prints the
confusion matrix / precision / recall / F1 / ROC-AUC / mAP to stdout after its
training loop (no image gallery there — it's a console script).

**Not covered:** `sequence_copy` is a per-timestep prediction task, not
single-label classification, so it has no confusion matrix/ROC/mAP. Detection
mAP (IoU-matched bounding boxes) isn't offered, since there's no detection
dataset/head in this tool yet — if one gets added, it belongs in `metrics.py`
alongside the classification mAP above, as a distinctly-named metric (the two
are not interchangeable).

## Downloading trained weights

A **Download .pt** button sits next to Test in the Training tab. It calls
`GET /api/train/download`, which serializes the current model's weights
(`torch.save(state_dict, ...)`) and streams them back as an attachment named
`<graph_name>_state_dict.pt`. Like Test, it needs a model to already exist
(press Play at least once first) or it errors cleanly instead of crashing.

The checkpoint's keys are deliberately remapped (`nn_graph/builder.py`'s
`export_state_dict()`) to match the flat `self.<node_id> = ...` naming used by
the exported `train.py`'s `GeneratedModel` class from **Export .py** — not the
nested `self._mods["node__<id>"]` naming this tool uses internally. That means
the two downloads are meant to be used together:

```bash
# 1. Export .py  -> train.py       (the architecture, as plain PyTorch code)
# 2. Download .pt -> model_state_dict.pt   (the weights you just trained in the UI)

python3 - <<'EOF'
import torch
from train import GeneratedModel

model = GeneratedModel()
model.load_state_dict(torch.load("model_state_dict.pt"))
model.eval()
EOF
```

This was verified end-to-end: `load_state_dict(..., strict=True)` succeeds
with zero missing/unexpected keys for CNN, MLP, and residual graphs alike.
(Positional-encoding nodes are the one deliberate exception — they're
non-trainable and regenerated deterministically in the exported code, so
they're excluded from the checkpoint rather than mismatched.)

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

## Object detection: YOLOv8 / YOLO11 (pretrained, fine-tunable)

The **Detect** tab is a deliberately separate workflow from the node-graph
editor above. YOLO models don't decompose into the atomic Conv/Pool/Linear
nodes the graph builder uses for training from scratch — they're normally
loaded pretrained and fine-tuned, and they need bounding-box datasets,
detection losses + NMS, and detection mAP (IoU-matched boxes), none of which
the classification pipeline (`nn_graph/builder.py`/`trainer.py`/`metrics.py`)
has. So `nn_graph/detection.py` wraps the `ultralytics` package directly
instead of forcing YOLO through the generic graph engine.

**Workflow:**
1. Pick a model — **YOLOv8** or **YOLO11**, sizes n/s/m/l/x (n = fastest/smallest,
   x = most accurate/slowest) — and click **Load Model**. COCO-pretrained
   weights download automatically on first use (cached under
   `.cache_ultralytics/`, gitignored).
2. **Test on an image**: drag any photo into the drop zone. Real inference
   runs immediately — bounding boxes, class labels, and confidences drawn
   right on the image — no training required. This is the fastest way to
   confirm the whole pipeline actually works.
3. **Fine-tune**: pick a dataset — built-in `coco8` (8 images, a few-second
   sanity check) or `coco128` (128 images, a real if tiny fine-tuning pass),
   or drop your own `data.yaml` (standard Ultralytics/YOLO format: an
   `images/`+`labels/` directory pair with YOLO-format `.txt` label files)
   into `detect_datasets/` — set epochs/imgsz/batch, and press **Play**. Loss
   and real detection mAP@0.5 / mAP@0.5:0.95 stream live via the same `/ws`
   envelope as the classification trainer, on topics `detect/progress`,
   `detect/done`, `detect/error`.
4. **Download .pt** grabs the fine-tuned weights (`best.pt` from that run).

**This is where real object-detection mAP lives** — IoU-matched bounding
boxes at the 0.5 and 0.5:0.95 thresholds, exactly what YOLO/COCO benchmarks
report. It's computed by Ultralytics' own validator, not by this tool, and is
a fundamentally different calculation from the classification mAP in
`nn_graph/metrics.py` (mean per-class average precision over class
probabilities, no boxes involved) — see that module's docstring for why the
two aren't interchangeable, and don't compare the numbers across tabs.

**Known limitations:**
- **Stop is best-effort**: Ultralytics has no public mid-run cancel API, so
  Stop works by shrinking `trainer.epochs` inside the `on_fit_epoch_end`
  callback — it finishes the epoch already in progress, then exits, rather
  than cancelling instantly.
- **No pause/step/resume** for fine-tuning (unlike the classification
  trainer) — Ultralytics' training loop is a single blocking call per run,
  not something this tool drives batch-by-batch.
- Custom dataset validation is minimal — malformed `data.yaml`/label files
  surface as whatever error Ultralytics itself raises, broadcast via
  `detect/error`, rather than a friendly pre-check.

## Layout

- `nn_graph/catalog.py` — single source of truth for every node type + params (served via `GET /api/catalog`).
- `nn_graph/schema.py` — structural graph validation (DAG, arity, required config nodes).
- `nn_graph/builder.py` — builds a real `nn.Module` from the graph (lazy in-dim inference).
- `nn_graph/validation.py` — dry-run shape inference, per-node error reporting.
- `nn_graph/datasets.py` — 2D toy datasets (circle/xor/gaussian/spiral, matching TF Playground), MNIST, synthetic sequence tasks.
- `nn_graph/metrics.py` — post-training evaluation: confusion matrix, precision/recall/F1, ROC-AUC.
- `nn_graph/trainer.py` — background-thread training session (play/pause/step/stop, early stopping, grad clipping) streaming metrics over `/ws`.
- `nn_graph/codegen.py` — exports a standalone, dependency-free `train.py`.
- `nn_graph/detection.py` — pretrained YOLOv8/YOLO11 object-detection workflow via `ultralytics` (separate from the classification pipeline above).
- `api_server.py` — Flask + flask-sock REST/WS backend.
- `ui/` — vanilla JS (no build step) node-graph editor + training dashboard + detect dashboard, styled to match `race_nav/server/agv_dashboard`.
- `saved_graphs/` — server-side saved architectures (gitignored).
- `detect_datasets/` — custom YOLO-format `data.yaml` datasets you provide for fine-tuning (gitignored).
- `.cache_ultralytics/` — pretrained weights, fine-tuning runs, and auto-downloaded sample datasets for the Detect tab (gitignored, kept separate from `.cache/`'s MNIST cache).
