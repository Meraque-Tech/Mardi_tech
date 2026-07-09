import { getSpec } from "../nodes/node_defs.js";
import * as api from "../graph/graphIO.js";
import wsClient from "../ws/ws_client.js";
import { LossChart } from "./lossChart.js";
import { DecisionBoundary } from "./decisionBoundary.js";

const TOY_2D = new Set(["circle", "xor", "gaussian", "spiral"]);
const ALL_DATASETS = ["circle", "xor", "gaussian", "spiral", "mnist", "sequence_copy", "sequence_classify"];

export class TrainingPanel {
  constructor(root, graphModel) {
    this.root = root;
    this.graph = graphModel;
    this.root.innerHTML = this._html();

    this.lossChart = new LossChart(this.root.querySelector("#loss-canvas"));
    this.boundary = new DecisionBoundary(this.root.querySelector("#boundary-canvas"));

    this._wireControls();
    this._wireWs();
    graphModel.onChange(() => this._syncFromGraph());
    this._syncFromGraph();
  }

  _html() {
    return `
      <div id="training-controls">
        <h3>Dataset</h3>
        <div class="dataset-grid" id="dataset-grid"></div>
        <h3>Hyperparameters</h3>
        <div id="config-form"></div>
        <h3>Controls</h3>
        <div class="controls-row">
          <button class="btn primary" id="btn-play">Play</button>
          <button class="btn" id="btn-pause">Pause</button>
          <button class="btn" id="btn-step">Step</button>
          <button class="btn danger" id="btn-stop">Stop</button>
        </div>
      </div>
      <div id="training-viz">
        <div class="viz-card">
          <h4>Loss / Accuracy</h4>
          <div class="metrics-row">
            <span>epoch <span class="val" id="metric-epoch">-</span></span>
            <span>loss <span class="val" id="metric-loss">-</span></span>
            <span>accuracy <span class="val" id="metric-acc">-</span></span>
          </div>
          <canvas class="chart-canvas" id="loss-canvas"></canvas>
        </div>
        <div class="viz-card" id="boundary-card">
          <h4>Decision Boundary</h4>
          <canvas id="boundary-canvas" width="320" height="320"></canvas>
        </div>
      </div>
    `;
  }

  _configNode(type) {
    return [...this.graph.nodes.values()].find((n) => n.type === type);
  }

  _syncFromGraph() {
    const dsNode = this._configNode("dataset");
    this._renderDatasetGrid(dsNode);
    this._renderConfigForm();
    const boundaryCard = this.root.querySelector("#boundary-card");
    boundaryCard.style.display = dsNode && TOY_2D.has(dsNode.params.kind) ? "" : "none";
    if (dsNode && TOY_2D.has(dsNode.params.kind)) this._refreshPreviewPoints(dsNode.params);
  }

  _renderDatasetGrid(dsNode) {
    const grid = this.root.querySelector("#dataset-grid");
    grid.innerHTML = ALL_DATASETS.map((k) => `<button class="dataset-btn ${dsNode && dsNode.params.kind === k ? "active" : ""}" data-kind="${k}">${k}</button>`).join("");
    grid.querySelectorAll(".dataset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!dsNode) return;
        this.graph.updateParams(dsNode.id, { kind: btn.dataset.kind });
      });
    });
  }

  _renderConfigForm() {
    const container = this.root.querySelector("#config-form");
    container.innerHTML = "";
    for (const type of ["dataset", "optimizer", "loss", "train_config"]) {
      const node = this._configNode(type);
      if (!node) continue;
      const spec = getSpec(type);
      const wrap = document.createElement("div");
      wrap.style.marginBottom = "0.6rem";
      wrap.innerHTML = `<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;margin-bottom:0.2rem;">${spec.label}</div>`;
      for (const p of spec.params) {
        if (p.name === "kind" && type === "dataset") continue; // handled by dataset grid
        const row = document.createElement("div");
        row.className = "param-row";
        const value = node.params[p.name];
        if (p.type === "bool") {
          row.innerHTML = `<label><input type="checkbox" ${value ? "checked" : ""}/> ${p.name}</label>`;
          row.querySelector("input").addEventListener("change", (e) => this.graph.updateParams(node.id, { [p.name]: e.target.checked }));
        } else if (p.type === "enum") {
          const opts = (p.options || []).map((o) => `<option value="${o}" ${o === value ? "selected" : ""}>${o}</option>`).join("");
          row.innerHTML = `<label>${p.name}</label><select>${opts}</select>`;
          row.querySelector("select").addEventListener("change", (e) => this.graph.updateParams(node.id, { [p.name]: e.target.value }));
        } else {
          row.innerHTML = `<label>${p.name}</label><input type="number" step="${p.type === "float" ? "any" : "1"}" value="${value}"/>`;
          row.querySelector("input").addEventListener("change", (e) => {
            const v = p.type === "float" ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
            this.graph.updateParams(node.id, { [p.name]: v });
          });
        }
        container.appendChild(row);
      }
    }
  }

  _wireControls() {
    this.root.querySelector("#btn-play").addEventListener("click", async () => {
      this.lossChart.reset();
      const res = await api.trainStart(this.graph.toJSON());
      if (!res.ok) this._error(res.errors);
    });
    this.root.querySelector("#btn-pause").addEventListener("click", () => api.trainPause());
    this.root.querySelector("#btn-step").addEventListener("click", () => api.trainStep());
    this.root.querySelector("#btn-stop").addEventListener("click", () => api.trainStop());
  }

  _error(errors) {
    console.error("Training error:", errors);
    const el = document.getElementById("toast");
    if (el) {
      el.textContent = (errors || []).map((e) => e.message).join("; ") || "Training failed";
      el.style.display = "block";
      setTimeout(() => (el.style.display = "none"), 4000);
    }
  }

  _wireWs() {
    wsClient.subscribe("train/progress", (data) => {
      this.root.querySelector("#metric-epoch").textContent = data.epoch;
      this.root.querySelector("#metric-loss").textContent = data.loss?.toFixed(4);
      this.root.querySelector("#metric-acc").textContent = data.accuracy !== undefined ? data.accuracy.toFixed(3) : "-";
      this.lossChart.push({ step: data.step, loss: data.loss, accuracy: data.accuracy });
    });
    wsClient.subscribe("train/boundary", (data) => {
      this.boundary.draw(data.values);
    });
    wsClient.subscribe("train/error", (data) => this._error([{ message: data.message }]));
  }

  _refreshPreviewPoints(params) {
    const pts = makePreviewPoints(params.kind, params.noise || 0, 160);
    this.boundary.setSamplePoints(pts, [-6, 6]);
    this.boundary.draw(null);
  }
}

// Lightweight client-side mirrors of nn_graph/datasets.py's 2D generators, used
// only to render an illustrative sample-point overlay before/while training.
function makePreviewPoints(kind, noise, n) {
  const pts = [];
  const rnd = (a, b) => a + Math.random() * (b - a);
  if (kind === "circle") {
    const radius = 5;
    for (let i = 0; i < n; i++) {
      const inner = i < n / 2;
      const r = inner ? rnd(0, radius * 0.5) : rnd(radius * 0.7, radius);
      const a = rnd(0, 2 * Math.PI);
      const x = r * Math.sin(a), y = r * Math.cos(a);
      pts.push({ x, y, label: x * x + y * y < (radius * 0.5) ** 2 ? 1 : 0 });
    }
  } else if (kind === "xor") {
    for (let i = 0; i < n; i++) {
      let x = rnd(-5, 5); x += x > 0 ? 0.3 : -0.3;
      let y = rnd(-5, 5); y += y > 0 ? 0.3 : -0.3;
      pts.push({ x, y, label: x * y >= 0 ? 1 : 0 });
    }
  } else if (kind === "gaussian") {
    const variance = 0.5 + noise * 7;
    for (let i = 0; i < n; i++) {
      const label = i < n / 2 ? 1 : 0;
      const [cx, cy] = label === 1 ? [2, 2] : [-2, -2];
      pts.push({ x: cx + (Math.random() - 0.5) * variance * 2, y: cy + (Math.random() - 0.5) * variance * 2, label });
    }
  } else if (kind === "spiral") {
    const half = n / 2;
    for (let label of [1, 0]) {
      const deltaT = label === 1 ? 0 : Math.PI;
      for (let i = 0; i < half; i++) {
        const r = (i / half) * 5;
        const t = 1.75 * (i / half) * 2 * Math.PI + deltaT;
        pts.push({ x: r * Math.sin(t) + rnd(-1, 1) * noise, y: r * Math.cos(t) + rnd(-1, 1) * noise, label });
      }
    }
  }
  return pts;
}
