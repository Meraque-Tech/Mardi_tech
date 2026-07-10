"""Pretrained object-detection workflow (YOLOv8 / YOLO11) via the `ultralytics`
package. Deliberately separate from nn_graph/builder.py's from-scratch
classification graph engine: YOLO is used as a pretrained detector you load
and fine-tune, not assembled node-by-node from Conv/Pool/Linear primitives --
it needs bounding-box datasets, detection losses + NMS, and detection mAP
(IoU-matched boxes), none of which the classification pipeline has.

Real object-detection mAP lives here, distinct from nn_graph/metrics.py's
classification mAP (mean per-class average precision over probabilities) --
see that module's docstring for why the two aren't interchangeable.
"""

import json
import os
import threading
import time
import uuid

BASE = os.path.dirname(os.path.abspath(__file__))
# Deliberately a sibling of the classification pipeline's .cache/ dir, not
# nested inside it: keeps ultralytics' (fairly heavy) weights/runs/datasets
# fully separate from nn_graph/datasets.py's MNIST cache.
CACHE_DIR = os.path.join(BASE, "..", ".cache_ultralytics")
_CONFIG_DIR = os.path.join(CACHE_DIR, "ultralytics_config")

# Keep ultralytics fully self-contained under this module's cache dir instead
# of leaking weights/runs/datasets into the user's home directory or repo root
# (its default behavior is to create them relative to wherever the process
# first ran from).
os.environ.setdefault("YOLO_CONFIG_DIR", _CONFIG_DIR)


def _configure_ultralytics_dirs():
    """Pre-writes ultralytics' settings.json with our cache paths *before*
    ultralytics is ever imported in this process.

    ultralytics bakes `DATASETS_DIR = Path(SETTINGS["datasets_dir"])` into a
    module-level constant at import time (ultralytics/utils/__init__.py) --
    calling `settings.update(...)` *after* that first import mutates the
    live settings dict/file but does not retroactively fix that already-
    computed constant, so dataset auto-download silently keeps resolving
    against the old path. Writing the file first, before the package's
    first import anywhere in this process, is the only way that sticks.
    """
    settings_path = os.path.join(_CONFIG_DIR, "Ultralytics", "settings.json")
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    payload = {
        "settings_version": "0.0.6",
        "datasets_dir": os.path.join(CACHE_DIR, "ultralytics_datasets"),
        "weights_dir": os.path.join(CACHE_DIR, "ultralytics_weights"),
        "runs_dir": os.path.join(CACHE_DIR, "ultralytics_runs"),
        "uuid": uuid.uuid4().hex + uuid.uuid4().hex,
        "sync": False, "api_key": "", "openai_api_key": "",
        "clearml": True, "comet": True, "dvc": True, "hub": False,
        "mlflow": True, "neptune": True, "raytune": True,
        "tensorboard": False, "wandb": False,
        "vscode_msg": True, "openvino_msg": True,
    }
    if os.path.isfile(settings_path):
        with open(settings_path) as f:
            existing = json.load(f)
        if existing.get("datasets_dir") == payload["datasets_dir"]:
            return  # already correct (e.g. a previous run of this server already wrote it)
    with open(settings_path, "w") as f:
        json.dump(payload, f, indent=2)

    from ultralytics import settings  # noqa: F401 - first import of ultralytics in this
    # process now happens with the correct settings.json already on disk.


MODEL_FAMILIES = {
    "yolov8": {"label": "YOLOv8", "prefix": "yolov8", "sizes": ["n", "s", "m", "l", "x"]},
    "yolo11": {"label": "YOLO11", "prefix": "yolo11", "sizes": ["n", "s", "m", "l", "x"]},
}

BUILTIN_DATASETS = {
    "coco8": "8 images -- fastest possible sanity check that training runs end-to-end.",
    "coco128": "128 images -- a real (if tiny) fine-tuning pass, still fast on CPU.",
}


def list_models():
    out = []
    for family, spec in MODEL_FAMILIES.items():
        for size in spec["sizes"]:
            out.append({"id": f"{spec['prefix']}{size}", "family": family, "label": f"{spec['label']}-{size}"})
    return out


def list_datasets():
    """Built-ins (auto-downloaded by ultralytics on first use) plus any
    custom data.yaml files placed under detect_datasets/ (mirrors the
    saved_graphs/ convention: user-provided, gitignored)."""
    out = [{"id": k, "label": k, "builtin": True, "note": v} for k, v in BUILTIN_DATASETS.items()]
    custom_dir = os.path.join(BASE, "..", "detect_datasets")
    if os.path.isdir(custom_dir):
        for fn in sorted(os.listdir(custom_dir)):
            if fn.endswith((".yaml", ".yml")):
                out.append({"id": os.path.join(custom_dir, fn), "label": fn, "builtin": False, "note": "custom dataset"})
    return out


class DetectionSession:
    """Wraps one loaded ultralytics YOLO model: on-demand inference, and a
    background-thread fine-tuning run with progress broadcast over the same
    {topic, data} websocket envelope the classification trainer uses."""

    def __init__(self, on_message):
        self.on_message = on_message
        self.model = None
        self.model_id = None
        self.lock = threading.Lock()
        self.status = {"loaded": False, "model_id": None, "training": False, "epoch": 0, "epochs": 0}
        self._stop = threading.Event()
        self.thread = None
        self.last_weights_path = None

    def load(self, model_id):
        valid_ids = {m["id"] for m in list_models()}
        if model_id not in valid_ids:
            return {"ok": False, "error": f"Unknown model '{model_id}'"}
        _configure_ultralytics_dirs()
        from ultralytics import YOLO

        weights_dir = os.path.join(CACHE_DIR, "ultralytics_weights")
        os.makedirs(weights_dir, exist_ok=True)
        weight_path = os.path.join(weights_dir, f"{model_id}.pt")
        try:
            model = YOLO(weight_path if os.path.isfile(weight_path) else f"{model_id}.pt")
        except Exception as exc:  # noqa: BLE001 - surface download/load errors to the UI
            return {"ok": False, "error": str(exc)}

        with self.lock:
            self.model = model
            self.model_id = model_id
            self.status["loaded"] = True
            self.status["model_id"] = model_id
        num_params = sum(p.numel() for p in model.model.parameters())
        return {"ok": True, "model_id": model_id, "num_params": num_params, "class_names": list(model.names.values())}

    def predict(self, image_path, conf=0.25):
        if self.model is None:
            return {"ok": False, "error": "No model loaded -- click Load Model first."}
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        r = results[0]
        h, w = r.orig_shape
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cls": cls_id, "label": self.model.names[cls_id],
                "conf": float(box.conf[0].item()),
            })
        return {"ok": True, "width": w, "height": h, "boxes": boxes}

    def start_training(self, dataset_id, epochs, imgsz, batch):
        if self.model is None:
            return {"ok": False, "error": "No model loaded -- click Load Model first."}
        if self.is_training():
            return {"ok": False, "error": "A fine-tuning run is already in progress. Stop it first."}

        data = dataset_id
        if dataset_id not in BUILTIN_DATASETS and not os.path.isfile(dataset_id):
            return {"ok": False, "error": f"Dataset '{dataset_id}' not found."}
        if dataset_id in BUILTIN_DATASETS:
            data = f"{dataset_id}.yaml"

        self._stop.clear()
        run_id = uuid.uuid4().hex[:8]
        with self.lock:
            self.status["training"] = True
            self.status["epoch"] = 0
            self.status["epochs"] = epochs
        self.thread = threading.Thread(target=self._train_run, args=(data, epochs, imgsz, batch, run_id), daemon=True)
        self.thread.start()
        return {"ok": True, "run_id": run_id}

    def is_training(self):
        with self.lock:
            return self.status["training"]

    def stop_training(self):
        self._stop.set()

    def _train_run(self, data, epochs, imgsz, batch, run_id):
        def on_fit_epoch_end(trainer):
            metrics = dict(trainer.metrics)
            with self.lock:
                self.status["epoch"] = trainer.epoch + 1
            self.on_message("detect/progress", {
                "run_id": run_id,
                "epoch": trainer.epoch + 1,
                "epochs": trainer.epochs,
                "box_loss": metrics.get("val/box_loss"),
                "cls_loss": metrics.get("val/cls_loss"),
                "dfl_loss": metrics.get("val/dfl_loss"),
                "precision": metrics.get("metrics/precision(B)"),
                "recall": metrics.get("metrics/recall(B)"),
                "map50": metrics.get("metrics/mAP50(B)"),
                "map50_95": metrics.get("metrics/mAP50-95(B)"),
            })
            if self._stop.is_set():
                # Best-effort stop: ultralytics has no public mid-run cancel API,
                # so force the epoch loop to exit after this (already-completed)
                # epoch rather than hard-killing the thread mid-checkpoint-write.
                trainer.epochs = trainer.epoch + 1

        try:
            self.model.reset_callbacks()  # drop any callback added by a previous training run on this model
            self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            self.model.train(
                data=data, epochs=epochs, imgsz=imgsz, batch=batch,
                project=os.path.join(CACHE_DIR, "ultralytics_runs"), name=run_id,
                exist_ok=True, verbose=False,
            )
            best = getattr(self.model.trainer, "best", None)
            self.last_weights_path = str(best) if best else None
            self.on_message("detect/done", {"run_id": run_id, "weights_path": self.last_weights_path})
        except Exception as exc:  # noqa: BLE001 - surface to UI without killing the server
            self.on_message("detect/error", {"run_id": run_id, "message": str(exc)})
        finally:
            with self.lock:
                self.status["training"] = False
