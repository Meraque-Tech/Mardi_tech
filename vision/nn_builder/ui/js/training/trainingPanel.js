import { getSpec } from "../nodes/node_defs.js";
import * as api from "../graph/graphIO.js";
import wsClient from "../ws/ws_client.js";
import { MiniLineChart } from "./lossChart.js";
import { DecisionBoundary } from "./decisionBoundary.js";
import { ConfusionMatrix } from "./confusionMatrix.js";
import { RocCurve } from "./rocCurve.js";
import { SamplePredictions } from "./samplePredictions.js";

const TOY_2D = new Set(["circle", "xor", "gaussian", "spiral"]);
const ALL_DATASETS = ["circle", "xor", "gaussian", "spiral", "mnist", "sequence_copy", "sequence_classify"];

export class TrainingPanel {
  constructor(root, graphModel) {
    this.root = root;
    this.graph = graphModel;
    this.root.innerHTML = this._html();

    this.lossChart = new MiniLineChart(this.root.querySelector("#loss-canvas"), {
      color: "#3987e5", label: "loss", format: (v) => v.toFixed(4),
    });
    this.accChart = new MiniLineChart(this.root.querySelector("#acc-canvas"), {
      color: "#0ca30c", label: "accuracy", format: (v) => v.toFixed(3),
    });
    this.boundary = new DecisionBoundary(this.root.querySelector("#boundary-canvas"));
    this.confusion = new ConfusionMatrix(this.root.querySelector("#confusion-canvas"));
    this.roc = new RocCurve(this.root.querySelector("#roc-canvas"));
    this.samples = new SamplePredictions(this.root.querySelector("#sample-grid"));

    this._wireControls();
    this._wireWs();
    graphModel.onChange(() => this._syncFromGraph());
    this._syncFromGraph();
  }

  // Canvases are sized from their parent's clientWidth, which is 0 while
  // their card is display:none (Training tab hidden, or eval-card not shown
  // yet) — call this right after the tab/card becomes visible, or the
  // canvas locks in at 0x0 until something else (e.g. a page reload)
  // happens to trigger a resize.
  resize() {
    this.lossChart._resize();
    this.accChart._resize();
    this.boundary._resize();
    this.confusion._resize();
    this.roc._resize();
  }

  _html() {
    return `
      <div id="training-controls">
        <div class="section-title">Dataset</div>
        <div class="dataset-grid" id="dataset-grid"></div>
        <div class="section-title">Hyperparameters</div>
        <div id="config-form"></div>
        <div class="section-title">Controls</div>
        <div class="controls-row">
          <button class="btn primary" id="btn-play">${icon("play")} Play</button>
          <button class="btn" id="btn-pause">${icon("pause")} Pause</button>
        </div>
        <div class="controls-row">
          <button class="btn" id="btn-step">${icon("step")} Step</button>
          <button class="btn danger" id="btn-stop">${icon("stop")} Stop</button>
        </div>
        <div class="controls-row">
          <button class="btn" id="btn-test" title="Re-evaluate the current model on a fresh random batch, without retraining">${icon("test")} Test</button>
        </div>
        <div class="controls-row">
          <button class="btn" id="btn-download-pt" title="Download the current model's weights as a .pt file">${icon("download")} Download .pt</button>
        </div>
      </div>
      <div id="training-viz">
        <div class="stat-tiles">
          <div class="stat-tile">
            <div class="stat-label">Epoch</div>
            <div class="stat-value tabular" id="metric-epoch">–</div>
          </div>
          <div class="stat-tile accent">
            <div class="stat-label">Loss</div>
            <div class="stat-value tabular" id="metric-loss">–</div>
          </div>
          <div class="stat-tile good">
            <div class="stat-label">Accuracy</div>
            <div class="stat-value tabular" id="metric-acc">–</div>
          </div>
        </div>
        <div class="viz-row">
          <div class="viz-card">
            <h4>Loss &amp; Accuracy</h4>
            <div class="chart-stack">
              <div class="chart-block">
                <canvas class="chart-canvas" id="loss-canvas"></canvas>
              </div>
              <div class="chart-block">
                <canvas class="chart-canvas" id="acc-canvas"></canvas>
              </div>
            </div>
          </div>
          <div class="viz-card" id="boundary-card">
            <h4>Decision Boundary</h4>
            <div class="boundary-wrap">
              <canvas id="boundary-canvas" width="300" height="300"></canvas>
              <div class="boundary-legend">
                <span>class 0</span><span class="ramp"></span><span>class 1</span>
              </div>
            </div>
          </div>
        </div>
        <div class="viz-card" id="eval-card" style="display:none;">
          <h4>Test-set Evaluation <span id="eval-n" style="font-weight:400; text-transform:none; color:var(--ink-muted);"></span></h4>
          <div class="stat-tiles" style="grid-template-columns: repeat(5, 1fr); margin-bottom: 1rem;">
            <div class="stat-tile"><div class="stat-label">Precision</div><div class="stat-value tabular" id="metric-precision">–</div></div>
            <div class="stat-tile"><div class="stat-label">Recall</div><div class="stat-value tabular" id="metric-recall">–</div></div>
            <div class="stat-tile"><div class="stat-label">F1</div><div class="stat-value tabular" id="metric-f1">–</div></div>
            <div class="stat-tile accent"><div class="stat-label">ROC-AUC</div><div class="stat-value tabular" id="metric-auc">–</div></div>
            <div class="stat-tile accent"><div class="stat-label">mAP</div><div class="stat-value tabular" id="metric-map">–</div></div>
          </div>
          <div class="viz-row" style="grid-template-columns: 1fr 1fr;">
            <div class="boundary-wrap">
              <div class="section-title" style="align-self:flex-start;">Confusion Matrix</div>
              <canvas id="confusion-canvas" width="280" height="280"></canvas>
              <div class="boundary-legend"><span>predicted →, true ↓ · shade = count</span></div>
            </div>
            <div class="boundary-wrap">
              <div class="section-title" style="align-self:flex-start;">ROC Curve</div>
              <canvas id="roc-canvas" width="280" height="280"></canvas>
              <div class="boundary-legend"><span>dashed = random-chance baseline</span></div>
            </div>
          </div>
          <div id="sample-card" style="margin-top:1rem; display:none;">
            <div class="section-title" style="margin-bottom:0.5rem;">Sample Predictions <span style="font-weight:400; text-transform:none; color:var(--ink-muted);">(green border = correct)</span></div>
            <div class="sample-grid" id="sample-grid"></div>
          </div>
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
      this.accChart.reset();
      this.root.querySelector("#eval-card").style.display = "none";
      const res = await api.trainStart(this.graph.toJSON());
      if (!res.ok) this._error(res.errors);
    });
    this.root.querySelector("#btn-pause").addEventListener("click", () => api.trainPause());
    this.root.querySelector("#btn-step").addEventListener("click", () => api.trainStep());
    this.root.querySelector("#btn-stop").addEventListener("click", () => api.trainStop());
    this.root.querySelector("#btn-test").addEventListener("click", async () => {
      const btn = this.root.querySelector("#btn-test");
      btn.disabled = true;
      try {
        const res = await api.trainEvaluate();
        if (!res.ok) this._error([{ message: res.error || "Test failed" }]);
      } finally {
        btn.disabled = false;
      }
    });
    this.root.querySelector("#btn-download-pt").addEventListener("click", async () => {
      const btn = this.root.querySelector("#btn-download-pt");
      btn.disabled = true;
      try {
        const res = await api.downloadCheckpoint();
        if (!res.ok) this._error([{ message: res.error }]);
      } finally {
        btn.disabled = false;
      }
    });
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
      this.root.querySelector("#metric-loss").textContent = data.loss?.toFixed(4) ?? "–";
      this.root.querySelector("#metric-acc").textContent = data.accuracy !== undefined ? data.accuracy.toFixed(3) : "–";
      this.lossChart.push(data.step, data.loss);
      this.accChart.push(data.step, data.accuracy);
    });
    wsClient.subscribe("train/boundary", (data) => {
      this.boundary.draw(data.values);
    });
    wsClient.subscribe("train/eval", (data) => {
      const fmt = (v) => (v === null || v === undefined ? "n/a" : v.toFixed(3));
      const card = this.root.querySelector("#eval-card");
      card.style.display = "";
      this.root.querySelector("#eval-n").textContent = `(n=${data.num_samples})`;
      this.root.querySelector("#metric-precision").textContent = fmt(data.precision);
      this.root.querySelector("#metric-recall").textContent = fmt(data.recall);
      this.root.querySelector("#metric-f1").textContent = fmt(data.f1);
      this.root.querySelector("#metric-auc").textContent = fmt(data.roc_auc);
      this.root.querySelector("#metric-map").textContent = fmt(data.map);

      // These canvases live inside a card that was display:none until the
      // line above -- resize them now that they have real dimensions, or
      // they draw onto a stale 0x0 canvas and only "fix themselves" on the
      // next full page reload (when everything resizes from scratch).
      this.confusion._resize();
      this.roc._resize();
      this.confusion.draw(data.confusion_matrix);
      this.roc.draw(data.roc_curve);

      const sampleCard = this.root.querySelector("#sample-card");
      if (data.samples && data.samples.length) {
        sampleCard.style.display = "";
        this.samples.draw(data.samples);
      } else {
        sampleCard.style.display = "none";
      }
    });
    wsClient.subscribe("train/error", (data) => this._error([{ message: data.message }]));
  }

  _refreshPreviewPoints(params) {
    const pts = makePreviewPoints(params.kind, params.noise || 0, 160);
    this.boundary.setSamplePoints(pts, [-6, 6]);
    this.boundary.draw(null);
  }
}

const ICONS = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>',
  step: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5l8 7-8 7zM16 5h2v14h-2z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>',
  test: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3h6M10 3v5.5L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 8.5V3"/><path d="M7.5 14h9"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>',
};
function icon(name) {
  return ICONS[name] || "";
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
