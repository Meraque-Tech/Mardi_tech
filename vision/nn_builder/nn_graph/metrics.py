"""Post-training evaluation metrics: confusion matrix, precision/recall/F1,
ROC-AUC, computed once against the held-out split after training finishes.

Applicable to single-label classification tasks — the 2D toy datasets, MNIST,
and sequence_classify. Not computed for sequence_copy (a per-timestep
prediction task, not a single-label classification target).

mAP (mean average precision) is an object-detection / retrieval metric — it
requires bounding boxes or ranked retrieval results, neither of which this
tool's classification-only datasets produce. It isn't computed here; if you
add a detection dataset/head later, mAP would live alongside these.
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
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
