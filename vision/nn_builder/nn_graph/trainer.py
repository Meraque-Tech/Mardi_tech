"""Background-thread training loop driving a GraphModule, with play/pause/
step/stop control and websocket-friendly progress callbacks.
"""

import random
import threading
import time
import uuid

import numpy as np
import torch
from torch import nn, optim

from .builder import build_module_from_graph, config_node
from .datasets import build_dataset
from .metrics import collect_image_samples, evaluate_classification, is_classification_task


def make_optimizer(params, cfg):
    kind = cfg.get("kind", "adam")
    lr = float(cfg.get("lr", 0.001))
    wd = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    if kind == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    if kind == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd)
    return optim.Adam(params, lr=lr, weight_decay=wd)


def make_loss(cfg):
    kind = cfg.get("kind", "cross_entropy")
    if kind == "mse":
        return nn.MSELoss()
    if kind == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    if kind == "nll":
        return nn.NLLLoss()
    return nn.CrossEntropyLoss()


def compute_loss(loss_fn, pred, target, task):
    if task == "sequence_copy":
        return loss_fn(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    if isinstance(loss_fn, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        return loss_fn(pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred,
                        target.float())
    return loss_fn(pred, target)


def compute_accuracy(pred, target, task):
    with torch.no_grad():
        if task == "sequence_copy":
            pred_labels = pred.argmax(dim=-1)
            return (pred_labels == target).float().mean().item()
        if pred.dim() > 1 and pred.size(-1) > 1:
            pred_labels = pred.argmax(dim=-1)
        else:
            pred_labels = (torch.sigmoid(pred.squeeze(-1)) > 0.5).long()
        return (pred_labels == target).float().mean().item()


def compute_decision_boundary(module, x_range, grid_size=30):
    lo, hi = x_range
    xs = np.linspace(lo, hi, grid_size)
    ys = np.linspace(lo, hi, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    module.eval()
    with torch.no_grad():
        pred = module(torch.from_numpy(pts))
        if pred.dim() > 1 and pred.size(-1) > 1:
            prob = torch.softmax(pred, dim=-1)[:, 1]
        else:
            prob = torch.sigmoid(pred.squeeze(-1))
    module.train()
    return prob.reshape(grid_size, grid_size).tolist()


class TrainingSession:
    def __init__(self, graph, on_message):
        self.session_id = uuid.uuid4().hex[:12]
        self.graph = graph
        self.on_message = on_message
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._step_once = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.status = {"running": False, "epoch": 0, "step": 0, "paused": False}
        self.module = None  # last-built model, kept around so "Test" can re-evaluate without retraining
        self.bundle = None  # the exact held-out split from the training run (used for the automatic post-training eval)
        self.dataset_cfg = None  # dataset node params, used to sample a *fresh* random batch for on-demand "Test" clicks

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        with self.lock:
            self.status["running"] = True
        self.thread.start()

    def stop(self):
        self._stop.set()
        self._pause.clear()

    def pause(self):
        self._pause.set()
        with self.lock:
            self.status["paused"] = True

    def resume(self):
        self._pause.clear()
        with self.lock:
            self.status["paused"] = False

    def step(self):
        self._step_once.set()

    def is_running(self):
        with self.lock:
            return self.status["running"]

    def _wait_if_paused(self):
        while self._pause.is_set() and not self._stop.is_set():
            if self._step_once.is_set():
                self._step_once.clear()
                return
            time.sleep(0.05)

    def _run(self):
        try:
            module = build_module_from_graph(self.graph)
            optimizer_cfg = config_node(self.graph, "optimizer")["params"]
            loss_cfg = config_node(self.graph, "loss")["params"]
            dataset_cfg = config_node(self.graph, "dataset")["params"]
            tc_node = config_node(self.graph, "train_config")
            tc = tc_node["params"] if tc_node else {}

            opt = make_optimizer(module.parameters(), optimizer_cfg)
            loss_fn = make_loss(loss_cfg)
            bundle = build_dataset(dataset_cfg)
            self.module = module
            self.bundle = bundle
            self.dataset_cfg = dataset_cfg
            epochs = int(tc.get("epochs", 20))
            progress_every = int(tc.get("progress_every", 1))
            grad_clip_norm = float(tc.get("grad_clip_norm", 0.0))
            early_stopping = bool(tc.get("early_stopping", False))
            patience = int(tc.get("early_stopping_patience", 5))
            min_delta = float(tc.get("early_stopping_min_delta", 0.0001))

            best_val_loss = float("inf")
            epochs_no_improve = 0
            early_stopped = False

            step = 0
            epoch = -1
            for epoch in range(epochs):
                if self._stop.is_set():
                    break
                epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
                for xb, yb in bundle.train_loader:
                    self._wait_if_paused()
                    if self._stop.is_set():
                        break

                    opt.zero_grad()
                    pred = module(xb)
                    loss_val = compute_loss(loss_fn, pred, yb, bundle.task)
                    loss_val.backward()
                    if grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(module.parameters(), grad_clip_norm)
                    opt.step()

                    step += 1
                    n_batches += 1
                    loss_scalar = loss_val.item()
                    acc_scalar = compute_accuracy(pred, yb, bundle.task)
                    epoch_loss += loss_scalar
                    epoch_acc += acc_scalar

                    with self.lock:
                        self.status["epoch"] = epoch
                        self.status["step"] = step

                    if step % progress_every == 0:
                        self.on_message("train/progress", {
                            "session_id": self.session_id, "epoch": epoch, "step": step,
                            "loss": loss_scalar, "accuracy": acc_scalar,
                        })

                if self._stop.is_set():
                    break

                val_loss = None
                if bundle.val_loader is not None:
                    val_loss, val_acc = self._evaluate(module, loss_fn, bundle)
                    self.on_message("train/progress", {
                        "session_id": self.session_id, "epoch": epoch, "step": step,
                        "loss": epoch_loss / max(n_batches, 1), "accuracy": epoch_acc / max(n_batches, 1),
                        "val_loss": val_loss, "val_accuracy": val_acc,
                    })

                if bundle.task == "2d" and bundle.x_range is not None:
                    grid = compute_decision_boundary(module, bundle.x_range)
                    self.on_message("train/boundary", {
                        "session_id": self.session_id, "epoch": epoch,
                        "grid_size": len(grid), "values": grid,
                    })

                if early_stopping and val_loss is not None:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            early_stopped = True
                            break

            if not self._stop.is_set():
                self.evaluate_now(fresh=False)  # the training run's actual held-out split

            self.on_message("train/done", {
                "session_id": self.session_id, "final_epoch": epoch, "early_stopped": early_stopped,
            })
        except Exception as exc:  # noqa: BLE001 - surface to UI without killing the server
            self.on_message("train/error", {"session_id": self.session_id, "message": str(exc)})
        finally:
            with self.lock:
                self.status["running"] = False

    def evaluate_now(self, fresh=True):
        """Re-runs test-set evaluation against whatever model is currently
        built (the last completed/in-progress training run), without
        retraining.

        fresh=True (the UI's on-demand "Test" button) draws a brand-new random
        batch from the same dataset config -- different noise/points/MNIST
        subset/sequence each click, never the model's original training-time
        held-out split. fresh=False (the automatic post-training call) uses
        that exact held-out split, matching normal "evaluate on the val set"
        convention right after training finishes.
        """
        if self.module is None or self.bundle is None:
            return {"ok": False, "error": "No trained model yet — press Play at least once first."}

        bundle = self.bundle
        if fresh:
            cfg = dict(self.dataset_cfg)
            cfg["seed"] = random.randint(0, 2**31 - 1)
            bundle = build_dataset(cfg)

        if bundle.val_loader is None:
            return {"ok": False, "error": "No held-out split to test against (train_ratio was 1.0)."}
        if not is_classification_task(bundle.task):
            return {"ok": False, "error": f"Test-set evaluation isn't computed for task '{bundle.task}'."}

        eval_metrics = evaluate_classification(self.module, bundle.val_loader, bundle.num_classes)
        if eval_metrics is None:
            return {"ok": False, "error": "Held-out split is empty."}
        if bundle.task == "image":
            eval_metrics["samples"] = collect_image_samples(self.module, bundle.val_loader)
        self.on_message("train/eval", {"session_id": self.session_id, **eval_metrics})
        return {"ok": True}

    def _evaluate(self, module, loss_fn, bundle):
        module.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in bundle.val_loader:
                pred = module(xb)
                total_loss += compute_loss(loss_fn, pred, yb, bundle.task).item()
                total_acc += compute_accuracy(pred, yb, bundle.task)
                n += 1
        module.train()
        return total_loss / max(n, 1), total_acc / max(n, 1)
