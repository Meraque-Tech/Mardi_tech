import * as api from "./detectIO.js";
import wsClient from "../ws/ws_client.js";
import { MiniLineChart } from "../training/lossChart.js";

// Fixed categorical order (dataviz skill: identity encoding, never cycled
// arbitrarily) -- reused as a cycling palette since detection classes can
// number in the dozens (COCO has 80); the label drawn on each box is what
// actually carries identity once we run out of distinct hues.
const BOX_COLORS = ["#3987e5", "#199e70", "#c98500", "#4caf3f", "#9085e9", "#d55181", "#e66767", "#d95926"];

export class DetectPanel {
  constructor(root) {
    this.root = root;
    this.root.innerHTML = this._html();
    this.currentImage = null; // HTMLImageElement currently shown
    this.modelLoaded = false;

    this.lossChart = new MiniLineChart(this.root.querySelector("#detect-loss-canvas"), {
      color: "#3987e5", label: "loss", format: (v) => v.toFixed(3),
    });
    this.map50Chart = new MiniLineChart(this.root.querySelector("#detect-map50-canvas"), {
      color: "#0ca30c", label: "mAP50", format: (v) => v.toFixed(3),
    });
    this.map5095Chart = new MiniLineChart(this.root.querySelector("#detect-map5095-canvas"), {
      color: "#9085e9", label: "mAP50-95", format: (v) => v.toFixed(3),
    });

    this._loadOptions();
    this._wireControls();
    this._wireWs();
  }

  resize() {
    this.lossChart._resize();
    this.map50Chart._resize();
    this.map5095Chart._resize();
  }

  _html() {
    return `
      <div id="detect-controls">
        <div class="section-title">Model</div>
        <select id="detect-model-select"></select>
        <div class="controls-row"><button class="btn primary" id="btn-load-model">Load Model</button></div>
        <div class="model-status" id="detect-model-status">No model loaded — pretrained COCO weights download on first use.</div>

        <div class="section-title">Fine-tune dataset</div>
        <select id="detect-dataset-select"></select>
        <div class="dataset-note" id="detect-dataset-note"></div>

        <div class="section-title">Hyperparameters</div>
        <div class="param-row"><label>epochs</label><input type="number" id="detect-epochs" value="10" min="1" max="1000"/></div>
        <div class="param-row"><label>imgsz</label><input type="number" id="detect-imgsz" value="640" min="32" max="1280" step="32"/></div>
        <div class="param-row"><label>batch</label><input type="number" id="detect-batch" value="8" min="1" max="256"/></div>

        <div class="section-title">Fine-tune Controls</div>
        <div class="controls-row">
          <button class="btn primary" id="btn-detect-play">${icon("play")} Play</button>
          <button class="btn danger" id="btn-detect-stop">${icon("stop")} Stop</button>
        </div>
        <div class="controls-row">
          <button class="btn" id="btn-detect-download">${icon("download")} Download .pt</button>
        </div>
      </div>
      <div id="detect-viz">
        <div class="viz-card">
          <h4>Test on an image</h4>
          <div class="dropzone" id="detect-dropzone">Drag &amp; drop an image here, or click to choose one</div>
          <input type="file" id="detect-file-input" accept="image/*" style="display:none;" />
          <div class="image-canvas-wrap" id="image-canvas-wrap" style="display:none; margin-top:0.8rem;">
            <canvas id="detect-image-canvas"></canvas>
          </div>
          <div class="detection-list" id="detection-list"></div>
        </div>
        <div class="stat-tiles" id="detect-stat-tiles">
          <div class="stat-tile"><div class="stat-label">Epoch</div><div class="stat-value tabular" id="detect-metric-epoch">–</div></div>
          <div class="stat-tile accent"><div class="stat-label">mAP50</div><div class="stat-value tabular" id="detect-metric-map50">–</div></div>
          <div class="stat-tile accent"><div class="stat-label">mAP50-95</div><div class="stat-value tabular" id="detect-metric-map5095">–</div></div>
        </div>
        <div class="viz-card" id="detect-charts-card">
          <h4>Fine-tuning Progress</h4>
          <div class="chart-stack">
            <div class="chart-block"><canvas class="chart-canvas" id="detect-loss-canvas"></canvas></div>
            <div class="chart-block"><canvas class="chart-canvas" id="detect-map50-canvas"></canvas></div>
            <div class="chart-block"><canvas class="chart-canvas" id="detect-map5095-canvas"></canvas></div>
          </div>
        </div>
      </div>
    `;
  }

  async _loadOptions() {
    const models = await api.listModels();
    const select = this.root.querySelector("#detect-model-select");
    const groups = {};
    for (const m of models) {
      groups[m.family] = groups[m.family] || [];
      groups[m.family].push(m);
    }
    select.innerHTML = Object.entries(groups).map(([family, items]) => `
      <optgroup label="${items[0].label.split("-")[0]}">
        ${items.map((m) => `<option value="${m.id}">${m.label}</option>`).join("")}
      </optgroup>
    `).join("");

    const datasets = await api.listDatasets();
    const dsSelect = this.root.querySelector("#detect-dataset-select");
    dsSelect.innerHTML = datasets.map((d) => `<option value="${d.id}">${d.label}${d.builtin ? "" : " (custom)"}</option>`).join("");
    const noteEl = this.root.querySelector("#detect-dataset-note");
    const updateNote = () => {
      const d = datasets.find((x) => x.id === dsSelect.value);
      noteEl.textContent = d ? d.note : "";
    };
    dsSelect.addEventListener("change", updateNote);
    updateNote();
  }

  _wireControls() {
    this.root.querySelector("#btn-load-model").addEventListener("click", async () => {
      const btn = this.root.querySelector("#btn-load-model");
      const status = this.root.querySelector("#detect-model-status");
      const modelId = this.root.querySelector("#detect-model-select").value;
      btn.disabled = true;
      status.className = "model-status";
      status.textContent = "Loading (downloading pretrained weights on first use — can take a minute)…";
      try {
        const res = await api.loadModel(modelId);
        if (res.ok) {
          this.modelLoaded = true;
          status.className = "model-status ok";
          status.textContent = `Loaded ${res.model_id} — ${res.num_params.toLocaleString()} parameters, ${res.class_names.length} classes.`;
        } else {
          status.textContent = res.error || "Failed to load model.";
        }
      } finally {
        btn.disabled = false;
      }
    });

    const dropzone = this.root.querySelector("#detect-dropzone");
    const fileInput = this.root.querySelector("#detect-file-input");
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files[0]) this._runDetection(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) this._runDetection(fileInput.files[0]);
    });

    this.root.querySelector("#btn-detect-play").addEventListener("click", async () => {
      this.lossChart.reset();
      this.map50Chart.reset();
      this.map5095Chart.reset();
      const res = await api.trainDetect({
        dataset_id: this.root.querySelector("#detect-dataset-select").value,
        epochs: parseInt(this.root.querySelector("#detect-epochs").value, 10),
        imgsz: parseInt(this.root.querySelector("#detect-imgsz").value, 10),
        batch: parseInt(this.root.querySelector("#detect-batch").value, 10),
      });
      if (!res.ok) this._error(res.error);
    });
    this.root.querySelector("#btn-detect-stop").addEventListener("click", () => api.stopDetect());
    this.root.querySelector("#btn-detect-download").addEventListener("click", async () => {
      const res = await api.downloadWeights();
      if (!res.ok) this._error(res.error);
    });
  }

  async _runDetection(file) {
    if (!this.modelLoaded) return this._error("Load a model first.");
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      this.currentImage = img;
      const conf = 0.25;
      const res = await api.predict(file, conf);
      URL.revokeObjectURL(url);
      if (!res.ok) return this._error(res.error);
      this._drawDetections(img, res.boxes);
    };
    img.src = url;
  }

  _drawDetections(img, boxes) {
    const wrap = this.root.querySelector("#image-canvas-wrap");
    wrap.style.display = "";
    const canvas = this.root.querySelector("#detect-image-canvas");
    const maxW = this.root.querySelector("#detect-viz").clientWidth - 32;
    const scale = Math.min(1, maxW / img.naturalWidth);
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const classColor = {};
    let colorIdx = 0;
    boxes.forEach((b) => {
      if (!(b.cls in classColor)) classColor[b.cls] = BOX_COLORS[colorIdx++ % BOX_COLORS.length];
      const color = classColor[b.cls];
      const x = b.x1 * scale, y = b.y1 * scale, w = (b.x2 - b.x1) * scale, h = (b.y2 - b.y1) * scale;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const label = `${b.label} ${(b.conf * 100).toFixed(0)}%`;
      ctx.font = "600 11px -apple-system, sans-serif";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
      ctx.fillStyle = "#0d0d0d";
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    });

    const list = this.root.querySelector("#detection-list");
    if (!boxes.length) {
      list.innerHTML = '<div class="empty-hint">No objects detected above the confidence threshold.</div>';
    } else {
      list.innerHTML = boxes
        .slice()
        .sort((a, b) => b.conf - a.conf)
        .map((b) => `
          <div class="detection-item">
            <span class="swatch" style="background:${classColor[b.cls]}"></span>
            <span>${b.label}</span>
            <span class="conf">${(b.conf * 100).toFixed(1)}%</span>
          </div>
        `).join("");
    }
  }

  _error(message) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message || "Something went wrong";
    el.style.display = "block";
    setTimeout(() => (el.style.display = "none"), 4000);
  }

  _wireWs() {
    wsClient.subscribe("detect/progress", (data) => {
      // ultralytics fires one extra on_fit_epoch_end for the final best.pt
      // re-validation pass after the last real training epoch -- clamp the
      // display so it doesn't show a confusing "3/2".
      this.root.querySelector("#detect-metric-epoch").textContent = `${Math.min(data.epoch, data.epochs)}/${data.epochs}`;
      this.root.querySelector("#detect-metric-map50").textContent = data.map50 !== undefined ? data.map50.toFixed(3) : "–";
      this.root.querySelector("#detect-metric-map5095").textContent = data.map50_95 !== undefined ? data.map50_95.toFixed(3) : "–";
      const totalLoss = (data.box_loss || 0) + (data.cls_loss || 0) + (data.dfl_loss || 0);
      this.lossChart.push(data.epoch, totalLoss);
      this.map50Chart.push(data.epoch, data.map50 || 0);
      this.map5095Chart.push(data.epoch, data.map50_95 || 0);
    });
    wsClient.subscribe("detect/done", () => this._error("Fine-tuning complete — Download .pt is ready."));
    wsClient.subscribe("detect/error", (data) => this._error(data.message));
  }
}

const ICONS = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 19h16"/></svg>',
};
function icon(name) {
  return ICONS[name] || "";
}
