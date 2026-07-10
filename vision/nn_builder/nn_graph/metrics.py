"""Post-training evaluation metrics: confusion matrix, precision/recall/F1,
ROC-AUC, mAP, computed once against the held-out split after training finishes.

Applicable to single-label classification tasks — the 2D toy datasets, MNIST,
and sequence_classify. Not computed for sequence_copy (a per-timestep
prediction task, not a single-label classification target).

mAP here means mean Average Precision the *classification* way: per class c,
the area under that class's one-vs-rest precision-recall curve (sklearn's
`average_precision_score`), averaged over classes. This is the same "mAP"
reported by multi-label/classification benchmarks (e.g. PASCAL VOC's
classification task) and only needs the probabilities this tool already
produces — no bounding boxes required. It is a *different* computation from
detection mAP (which averages AP over IoU-matched boxes per class); this tool
has no detection dataset/head, so that variant isn't offered.
"""

import numpy as np
import torch

CLASSIFICATION_TASKS = {"2d", "image", "sequence_classify"}


def is_classification_task(task):
    return task in CLASSIFICATION_TASKS


def evaluate_classification(module, loader, num_classes):
    module.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = module(xb)
            if pred.dim() > 1 and pred.size(-1) > 1:
                prob = torch.softmax(pred, dim=-1)
                labels = prob.argmax(dim=-1)
            else:
                prob1 = torch.sigmoid(pred.squeeze(-1))
                prob = torch.stack([1 - prob1, prob1], dim=-1)
                labels = (prob1 > 0.5).long()
            all_true.append(yb.cpu().numpy())
            all_pred.append(labels.cpu().numpy())
            all_prob.append(prob.cpu().numpy())
    module.train()

    if not all_true:
        return None

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    cm = _confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1 = _prf_macro(cm)
    accuracy = float((y_true == y_pred).mean())
    roc_auc = _roc_auc(y_true, y_prob, num_classes)
    mean_ap = _mean_average_precision(y_true, y_prob, num_classes)
    roc_curve_data = _roc_curve_data(y_true, y_prob, num_classes)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "map": mean_ap,
        "roc_curve": roc_curve_data,
        "confusion_matrix": cm.tolist(),
        "num_classes": num_classes,
        "num_samples": int(len(y_true)),
    }


def _confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _prf_macro(cm):
    num_classes = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def _roc_auc(y_true, y_prob, num_classes):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        ))
    except ValueError:
        return None  # e.g. a class absent from this particular held-out split


def _roc_curve_data(y_true, y_prob, num_classes, max_points=60):
    """Returns {"fpr": [...], "curves": [{"label": ..., "tpr": [...]}]} for the
    ROC-curve plot (not just the scalar AUC). All per-class curves are
    interpolated onto a shared FPR grid so multiple classes can share one
    x-axis. Beyond 6 classes (e.g. MNIST's 10) we show the macro-average curve
    only -- one line per class would be unreadable spaghetti."""
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return None

    grid = np.linspace(0, 1, max_points)
    curves = []
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        curves.append({"label": "class 1", "tpr": np.interp(grid, fpr, tpr).tolist()})
    else:
        per_class_tprs, eligible = [], 0
        for c in range(num_classes):
            y_true_c = (y_true == c).astype(int)
            if 0 < y_true_c.sum() < len(y_true_c):
                fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
                tpr_i = np.interp(grid, fpr, tpr)
                per_class_tprs.append(tpr_i)
                eligible += 1
                if num_classes <= 6:
                    curves.append({"label": f"class {c}", "tpr": tpr_i.tolist()})
        if num_classes > 6 and per_class_tprs:
            curves = [{"label": f"macro-avg ({eligible} classes)", "tpr": np.mean(per_class_tprs, axis=0).tolist()}]

    if not curves:
        return None
    return {"fpr": grid.tolist(), "curves": curves}


def _mean_average_precision(y_true, y_prob, num_classes):
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return None
    aps = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue  # class absent (or the only class) in this split -- AP undefined
        aps.append(average_precision_score(y_true_c, y_prob[:, c]))
    return float(np.mean(aps)) if aps else None


def collect_image_samples(module, loader, max_samples=16):
    """Grabs a handful of held-out images with true/predicted labels, for a
    'what did the model actually get right/wrong' preview in the UI. Pixels
    are returned as 0-255 grayscale (channel-averaged) so the frontend can
    draw them directly onto a canvas."""
    module.eval()
    samples = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = module(xb)
            if pred.dim() > 1 and pred.size(-1) > 1:
                labels = torch.softmax(pred, dim=-1).argmax(dim=-1)
            else:
                labels = (torch.sigmoid(pred.squeeze(-1)) > 0.5).long()
            for i in range(xb.size(0)):
                if len(samples) >= max_samples:
                    break
                img = xb[i]  # (C, H, W), values in [0, 1]
                gray = img.mean(dim=0) if img.size(0) > 1 else img[0]
                samples.append({
                    "pixels": gray.clamp(0, 1).mul(255).byte().cpu().numpy().tolist(),
                    "true": int(yb[i].item()),
                    "pred": int(labels[i].item()),
                })
            if len(samples) >= max_samples:
                break
    module.train()
    return samples
