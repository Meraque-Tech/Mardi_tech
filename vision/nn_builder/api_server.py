#!/usr/bin/env python3
"""Flask + flask-sock backend for the visual PyTorch neural-network builder.

Endpoints:
  GET  /                        UI (ui/index.html)
  GET  /api/catalog             node type catalog (single source of truth)
  POST /api/graph/validate      structural + shape-inference dry run
  POST /api/graph/build         build the graph, return param count + summary
  POST /api/graph/export        generate a standalone train.py
  GET  /api/graphs              list saved graphs
  POST /api/graphs/<name>       save/overwrite a graph
  GET  /api/graphs/<name>       load a graph
  DELETE /api/graphs/<name>     delete a graph
  POST /api/train/start         start a background training session
  POST /api/train/stop|pause|resume|step|evaluate
  GET  /api/train/status
  GET  /api/train/download     download the current model's weights as a .pt file
  GET  /api/detect/models       available pretrained YOLOv8/YOLO11 variants
  GET  /api/detect/datasets     built-in + custom fine-tuning datasets
  POST /api/detect/load         load/download a pretrained YOLO checkpoint
  POST /api/detect/predict      run detection on an uploaded image
  POST /api/detect/train        fine-tune the loaded model in the background
  POST /api/detect/stop
  GET  /api/detect/status
  GET  /api/detect/download     download the fine-tuned weights
  WS   /ws                      {topic, data} envelope: train/*, detect/*
"""

import argparse
import io
import json
import os
import re
import tempfile
import threading

import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_sock import Sock

from nn_graph.builder import build_module_from_graph, export_state_dict
from nn_graph.catalog import catalog_payload
from nn_graph.codegen import graph_to_train_py
from nn_graph.detection import DetectionSession, list_datasets, list_models
from nn_graph.schema import GraphError, graph_from_dict, validate_graph
from nn_graph.trainer import TrainingSession
from nn_graph.validation import dry_run

BASE = os.path.dirname(os.path.abspath(__file__))
SAVED_GRAPHS_DIR = os.path.join(BASE, "saved_graphs")
os.makedirs(SAVED_GRAPHS_DIR, exist_ok=True)

app = Flask(__name__, static_folder="ui", static_url_path="")
sock = Sock(app)

ws_clients = []
ws_lock = threading.Lock()

_session: TrainingSession = None
_session_lock = threading.Lock()

_detect_session: DetectionSession = None
_detect_lock = threading.Lock()


def broadcast(topic, data):
    payload = json.dumps({"topic": topic, "data": data})
    with ws_lock:
        dead = []
        for ws in ws_clients:
            try:
                ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)


@sock.route("/ws")
def ws_route(ws):
    with ws_lock:
        ws_clients.append(ws)
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
    finally:
        with ws_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)


@app.get("/")
def index():
    return send_from_directory("ui", "index.html")


@app.get("/api/catalog")
def catalog():
    return jsonify(catalog_payload())


def _graph_from_request():
    body = request.get_json(silent=True) or {}
    return graph_from_dict(body.get("graph", body))


@app.post("/api/graph/validate")
def graph_validate():
    graph = _graph_from_request()
    return jsonify(dry_run(graph))


@app.post("/api/graph/build")
def graph_build():
    graph = _graph_from_request()
    result = dry_run(graph)
    if not result["ok"]:
        return jsonify(result), 400
    try:
        module = build_module_from_graph(graph)
    except GraphError as e:
        return jsonify({"ok": False, "errors": [e.to_dict()], "shapes": result["shapes"]}), 400
    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    layer_summary = [
        {"id": nid, "type": module.node_by_id[nid]["type"], "output_shape": result["shapes"].get(nid)}
        for nid in module.order
    ]
    return jsonify({"ok": True, "param_count": param_count, "layer_summary": layer_summary})


@app.post("/api/graph/export")
def graph_export():
    body = request.get_json(silent=True) or {}
    graph = graph_from_dict(body.get("graph", body))
    struct_errors = validate_graph(graph)
    if struct_errors:
        return jsonify({"ok": False, "errors": [e.to_dict() for e in struct_errors]}), 400
    try:
        code = graph_to_train_py(graph, body.get("train_config"))
    except GraphError as e:
        return jsonify({"ok": False, "errors": [e.to_dict()]}), 400
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", graph.get("meta", {}).get("name") or "model")
    return jsonify({"ok": True, "filename": f"train_{name}.py", "code": code})


def _safe_name(name):
    base = os.path.basename(name)
    if not re.match(r"^[a-zA-Z0-9_-]+$", base):
        return None
    return base


@app.get("/api/graphs")
def graphs_list():
    items = []
    for fn in sorted(os.listdir(SAVED_GRAPHS_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(SAVED_GRAPHS_DIR, fn)
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        items.append({
            "name": fn[:-5],
            "id": d.get("id"),
            "saved_at": d.get("meta", {}).get("saved_at"),
            "family_hint": d.get("meta", {}).get("family_hint"),
        })
    return jsonify(items)


@app.post("/api/graphs/<name>")
def graphs_save(name):
    safe = _safe_name(name)
    if not safe:
        return jsonify({"error": "invalid name"}), 400
    import datetime

    body = request.get_json(silent=True) or {}
    graph = graph_from_dict(body.get("graph", body))
    graph["meta"]["saved_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(os.path.join(SAVED_GRAPHS_DIR, f"{safe}.json"), "w") as f:
        json.dump(graph, f, indent=2)
    return jsonify({"ok": True, "name": safe})


@app.get("/api/graphs/<name>")
def graphs_load(name):
    safe = _safe_name(name)
    path = os.path.join(SAVED_GRAPHS_DIR, f"{safe}.json") if safe else None
    if not safe or not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.delete("/api/graphs/<name>")
def graphs_delete(name):
    safe = _safe_name(name)
    path = os.path.join(SAVED_GRAPHS_DIR, f"{safe}.json") if safe else None
    if not safe or not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    os.remove(path)
    return jsonify({"ok": True})


@app.post("/api/train/start")
def train_start():
    global _session
    with _session_lock:
        if _session is not None and _session.is_running():
            return jsonify({"error": "A training session is already running. POST /api/train/stop first."}), 409
        body = request.get_json(silent=True) or {}
        graph = graph_from_dict(body.get("graph", body))
        struct_errors = validate_graph(graph)
        if struct_errors:
            return jsonify({"ok": False, "errors": [e.to_dict() for e in struct_errors]}), 400
        _session = TrainingSession(graph, broadcast)
        _session.start()
        return jsonify({"ok": True, "session_id": _session.session_id})


def _with_session(fn):
    with _session_lock:
        if _session is None:
            return jsonify({"error": "no session"}), 409
        fn(_session)
        return jsonify({"ok": True, "status": _session.status})


@app.post("/api/train/stop")
def train_stop():
    return _with_session(lambda s: s.stop())


@app.post("/api/train/pause")
def train_pause():
    return _with_session(lambda s: s.pause())


@app.post("/api/train/resume")
def train_resume():
    return _with_session(lambda s: s.resume())


@app.post("/api/train/step")
def train_step():
    return _with_session(lambda s: s.step())


@app.post("/api/train/evaluate")
def train_evaluate():
    with _session_lock:
        if _session is None:
            return jsonify({"ok": False, "error": "No trained model yet — press Play at least once first."}), 409
        result = _session.evaluate_now()
        return jsonify(result), (200 if result.get("ok") else 400)


@app.get("/api/train/status")
def train_status():
    if _session is None:
        return jsonify({"running": False})
    return jsonify({"session_id": _session.session_id, **_session.status})


@app.get("/api/train/download")
def train_download():
    if _session is None or _session.module is None:
        return jsonify({"error": "No trained model yet — press Play at least once first."}), 409
    state_dict = export_state_dict(_session.module)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", (_session.graph.get("meta") or {}).get("name") or "model")
    return send_file(
        buffer, as_attachment=True, download_name=f"{name}_state_dict.pt",
        mimetype="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Object detection (pretrained YOLOv8/YOLO11 via ultralytics) -- a separate
# workflow from the classification graph builder above; see nn_graph/detection.py.
# ---------------------------------------------------------------------------

def _get_detect_session():
    global _detect_session
    if _detect_session is None:
        _detect_session = DetectionSession(broadcast)
    return _detect_session


@app.get("/api/detect/models")
def detect_models():
    return jsonify(list_models())


@app.get("/api/detect/datasets")
def detect_datasets():
    return jsonify(list_datasets())


@app.post("/api/detect/load")
def detect_load():
    body = request.get_json(silent=True) or {}
    model_id = body.get("model_id")
    with _detect_lock:
        session = _get_detect_session()
        result = session.load(model_id)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.post("/api/detect/predict")
def detect_predict():
    if _detect_session is None or _detect_session.model is None:
        return jsonify({"ok": False, "error": "No model loaded — click Load Model first."}), 409
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400
    conf = float(request.form.get("conf", 0.25))
    file = request.files["image"]
    suffix = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = _detect_session.predict(tmp_path, conf=conf)
    finally:
        os.remove(tmp_path)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.post("/api/detect/train")
def detect_train():
    body = request.get_json(silent=True) or {}
    with _detect_lock:
        session = _get_detect_session()
        result = session.start_training(
            dataset_id=body.get("dataset_id", "coco8"),
            epochs=int(body.get("epochs", 10)),
            imgsz=int(body.get("imgsz", 640)),
            batch=int(body.get("batch", 8)),
        )
    return jsonify(result), (200 if result.get("ok") else 400)


@app.post("/api/detect/stop")
def detect_stop():
    if _detect_session is None:
        return jsonify({"error": "no session"}), 409
    _detect_session.stop_training()
    return jsonify({"ok": True})


@app.get("/api/detect/status")
def detect_status():
    if _detect_session is None:
        return jsonify({"loaded": False, "training": False})
    return jsonify(_detect_session.status)


@app.get("/api/detect/download")
def detect_download():
    if _detect_session is None or not _detect_session.last_weights_path:
        return jsonify({"error": "No fine-tuned weights yet — run a training pass first."}), 409
    path = _detect_session.last_weights_path
    if not os.path.isfile(path):
        return jsonify({"error": "Weights file no longer exists on disk"}), 404
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", _detect_session.model_id or "model")
    return send_file(path, as_attachment=True, download_name=f"{name}_finetuned.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)
