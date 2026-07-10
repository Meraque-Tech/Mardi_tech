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

## Layout

- `nn_graph/catalog.py` — single source of truth for every node type + params (served via `GET /api/catalog`).
- `nn_graph/schema.py` — structural graph validation (DAG, arity, required config nodes).
- `nn_graph/builder.py` — builds a real `nn.Module` from the graph (lazy in-dim inference).
- `nn_graph/validation.py` — dry-run shape inference, per-node error reporting.
- `nn_graph/datasets.py` — 2D toy datasets (circle/xor/gaussian/spiral, matching TF Playground), MNIST, synthetic sequence tasks.
- `nn_graph/trainer.py` — background-thread training session (play/pause/step/stop) streaming metrics over `/ws`.
- `nn_graph/codegen.py` — exports a standalone, dependency-free `train.py`.
- `api_server.py` — Flask + flask-sock REST/WS backend.
- `ui/` — vanilla JS (no build step) node-graph editor + training dashboard, styled to match `race_nav/server/agv_dashboard`.
- `saved_graphs/` — server-side saved architectures (gitignored).
