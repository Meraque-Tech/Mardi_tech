const state = {
  source: "upload",
  datasetYaml: "",
  datasetSummary: null,
  pollTimer: null,
  metricsHistory: [],
  metricsAvailable: false,
  logMode: "recent",
};

const $ = (id) => document.getElementById(id);

const CONTROL_DEFAULTS = {
  "model-size": "nano",
  epochs: 100,
  imgsz: 640,
  batch: 16,
  patience: 20,
  "save-period": -1,
  "run-name": "train",
  project: "runs/detect",
  workers: "2",
  optimizer: "auto",
  seed: 0,
  lr0: 0.01,
  lrf: 0.01,
  "weight-decay": 0.0005,
  "warmup-epochs": 3.0,
  freeze: "",
  "cos-lr": false,
  activation: "silu",
  pretrained: true,
  "exist-ok": false,
  resume: false,
};

const TRAINING_PRESETS = {
  stable: {
    epochs: 100,
    imgsz: 640,
    batch: 16,
    patience: 20,
    workers: "2",
    optimizer: "auto",
    lr0: 0.01,
    lrf: 0.01,
    "weight-decay": 0.0005,
    "warmup-epochs": 3.0,
    "cos-lr": false,
    pretrained: true,
    activation: "silu",
  },
  low_vram: {
    "model-size": "nano",
    epochs: 100,
    imgsz: 512,
    batch: 4,
    patience: 20,
    workers: "0",
    optimizer: "auto",
    lr0: 0.005,
    lrf: 0.01,
    "weight-decay": 0.0005,
    "warmup-epochs": 3.0,
    "cos-lr": true,
    pretrained: true,
    activation: "silu",
  },
  quick: {
    "model-size": "nano",
    epochs: 10,
    imgsz: 640,
    batch: 8,
    patience: 5,
    workers: "2",
    optimizer: "auto",
    lr0: 0.01,
    lrf: 0.01,
    "weight-decay": 0.0005,
    "warmup-epochs": 1.0,
    "cos-lr": false,
    pretrained: true,
    activation: "silu",
  },
  accuracy: {
    "model-size": "small",
    epochs: 150,
    imgsz: 768,
    batch: 8,
    patience: 40,
    workers: "4",
    optimizer: "auto",
    lr0: 0.006,
    lrf: 0.01,
    "weight-decay": 0.0005,
    "warmup-epochs": 3.0,
    "cos-lr": true,
    pretrained: true,
    activation: "silu",
  },
};

function setMessage(text, isError = false) {
  const message = $("message");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function numberValue(id) {
  return Number($(id).value);
}

function optionalNumberValue(id) {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
}

function classNames() {
  return $("classes").value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function setClassNames(classes) {
  if (Array.isArray(classes) && classes.length) {
    $("classes").value = classes.join("\n");
  }
}

function setControlValue(id, value) {
  const element = $(id);
  if (!element) {
    return;
  }
  if (element.type === "checkbox") {
    element.checked = Boolean(value);
    return;
  }
  element.value = value;
}

function applyControlValues(values) {
  Object.entries(values).forEach(([id, value]) => setControlValue(id, value));
  refreshWeightsStatus();
  refreshMetrics();
}

function applyPreset(name) {
  const preset = TRAINING_PRESETS[name];
  if (!preset) {
    return;
  }
  applyControlValues(preset);
  setMessage(`Applied ${name.replace("_", " ")} preset.`);
}

function cleanYamlValue(value) {
  return value
    .trim()
    .replace(/,$/, "")
    .replace(/^["']|["']$/g, "")
    .trim();
}

function parseInlineYamlNames(value) {
  const trimmed = value.trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map(cleanYamlValue)
      .filter(Boolean);
  }

  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map((item) => cleanYamlValue(item.split(":").slice(1).join(":")))
      .filter(Boolean);
  }

  return [];
}

function parseYamlClassNames(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const namesLineIndex = lines.findIndex((line) => /^\s*names\s*:/.test(line));
  if (namesLineIndex === -1) {
    return [];
  }

  const namesLine = lines[namesLineIndex];
  const inlineValue = namesLine.split(":").slice(1).join(":").trim();
  if (inlineValue) {
    return parseInlineYamlNames(inlineValue);
  }

  const namesIndent = namesLine.match(/^\s*/)[0].length;
  const detected = [];
  for (const line of lines.slice(namesLineIndex + 1)) {
    if (!line.trim()) {
      continue;
    }

    const indent = line.match(/^\s*/)[0].length;
    if (indent <= namesIndent) {
      break;
    }

    const trimmed = line.trim();
    if (trimmed.startsWith("-")) {
      detected.push(cleanYamlValue(trimmed.slice(1)));
      continue;
    }

    if (/^["']?\d+["']?\s*:/.test(trimmed)) {
      detected.push(cleanYamlValue(trimmed.split(":").slice(1).join(":")));
    }
  }

  return detected.filter(Boolean);
}

async function detectClassesFromFolderSelection() {
  const files = Array.from($("folder-files").files);
  const yamlFile = files.find((file) => {
    const name = (file.webkitRelativePath || file.name).split("/").pop();
    return name === "data.yaml" || name === "dataset.yaml";
  });

  if (!yamlFile) {
    throw new Error("No data.yaml or dataset.yaml file found in the selected folder.");
  }

  const classes = parseYamlClassNames(await yamlFile.text());
  if (!classes.length) {
    throw new Error(`No class names found in ${yamlFile.webkitRelativePath || yamlFile.name}.`);
  }

  return classes;
}

function splitConfig() {
  return {
    train: numberValue("split-train"),
    val: numberValue("split-val"),
    test: numberValue("split-test"),
  };
}

function validateSplitTotal() {
  const split = splitConfig();
  const total = split.train + split.val + split.test;
  if (total !== 100) {
    throw new Error(`Train, val, and test split values must total 100%. Current total: ${total}%.`);
  }
}

function weightTarget() {
  return {
    project: $("project").value || "runs/detect",
    name: $("run-name").value || "train",
  };
}

function weightDownloadUrl(weight) {
  const params = new URLSearchParams(weightTarget());
  return `/api/train/weights/${weight}?${params.toString()}`;
}

function formatBytes(bytes) {
  if (!bytes) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function metricText(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return `${value}${suffix}`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderClassMetrics(classes) {
  const container = $("class-metrics");
  if (!Array.isArray(classes) || !classes.length) {
    container.innerHTML = "";
    return;
  }

  const rows = classes.map((item) => `
    <tr>
      <td>${escapeHtml(item.class_name)}</td>
      <td>${item.instances}</td>
      <td>${metricText(item.f1)}</td>
      <td>${metricText(item.precision)}</td>
      <td>${metricText(item.recall)}</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <h4>Per-Class F1</h4>
    <table>
      <thead>
        <tr>
          <th>Class</th>
          <th>Instances</th>
          <th>F1</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderDatasetSummary(summary) {
  const container = $("dataset-summary");
  state.datasetSummary = summary || null;
  if (!summary || !summary.splits) {
    container.innerHTML = "";
    return;
  }

  const splitRows = ["train", "val", "test"].map((split) => {
    const row = summary.splits[split] || {};
    return `
      <div>
        <span>${split.toUpperCase()}</span>
        <strong>${row.images || 0}</strong>
        <small>${row.missing_labels || 0} missing labels</small>
      </div>
    `;
  }).join("");
  const classPreview = Array.isArray(summary.classes) ? summary.classes.slice(0, 8).join(", ") : "";
  const classSuffix = Array.isArray(summary.classes) && summary.classes.length > 8 ? "..." : "";
  const warnings = Array.isArray(summary.warnings) && summary.warnings.length
    ? `<ul>${summary.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "<p>No dataset warnings found.</p>";

  container.innerHTML = `
    <div class="summary-header">
      <h3>Dataset Summary</h3>
      <p>${summary.total_images || 0} images, ${summary.class_count || 0} classes</p>
    </div>
    <div class="summary-grid">${splitRows}</div>
    <p>${escapeHtml(classPreview)}${classSuffix}</p>
    ${warnings}
  `;
}

function formatBestMetric(row, key, label) {
  if (!row || row[key] === null || row[key] === undefined) {
    return "";
  }
  return `<div><span>${label}</span><strong>${metricText(row[key])}</strong><small>Epoch ${row.epoch}</small></div>`;
}

function renderBestMetrics(best) {
  const container = $("best-metrics");
  if (!best) {
    container.innerHTML = "";
    return;
  }
  const rows = [
    formatBestMetric(best.best_map50_95, "map50_95", "Best mAP50-95"),
    formatBestMetric(best.best_map50, "map50", "Best mAP50"),
    formatBestMetric(best.best_f1, "overall_f1", "Best F1"),
    formatBestMetric(best.lowest_validation_loss, "testing_loss", "Lowest validation loss"),
  ].filter(Boolean);
  container.innerHTML = rows.length ? `<h4>Best Epochs</h4><div class="best-grid">${rows.join("")}</div>` : "";
}

function setArtifactButtons(artifacts) {
  const isEnabled = (key) => {
    if (typeof artifacts === "boolean") {
      return artifacts;
    }
    return Boolean(artifacts && artifacts[key] && artifacts[key].available);
  };
  $("download-results-csv").disabled = !isEnabled("results_csv");
  $("download-accuracy-graph").disabled = !isEnabled("accuracy_graph");
  $("download-loss-graph").disabled = !isEnabled("loss_graph");
}

function resetCharts() {
  drawLineChart("accuracy-chart", [], []);
  drawLineChart("loss-chart", [], []);
}

function chartPoint(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? null : Number(value);
}

function drawLineChart(canvasId, history, series) {
  const canvas = $(canvasId);
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width || canvas.width));
  const height = Math.max(200, Math.floor(rect.height || canvas.height));
  canvas.width = Math.floor(width * scale);
  canvas.height = Math.floor(height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, width, height);

  context.fillStyle = "#f8fafb";
  context.fillRect(0, 0, width, height);

  if (!history.length) {
    context.fillStyle = "#607080";
    context.font = "13px sans-serif";
    context.fillText("No epoch data yet", 16, 28);
    return;
  }

  const padding = { top: 18, right: 18, bottom: 34, left: 46 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const epochs = history.map((item) => item.epoch);
  const values = [];
  series.forEach((line) => {
    history.forEach((item) => {
      const value = chartPoint(item[line.key]);
      if (value !== null) {
        values.push(value);
      }
    });
  });

  if (!values.length) {
    context.fillStyle = "#607080";
    context.font = "13px sans-serif";
    context.fillText("No values available", 16, 28);
    return;
  }

  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (minValue === maxValue) {
    minValue -= 0.1;
    maxValue += 0.1;
  }
  const valuePadding = (maxValue - minValue) * 0.08;
  minValue = Math.max(0, minValue - valuePadding);
  maxValue += valuePadding;

  const xFor = (epoch) => padding.left + ((epoch - minEpoch) / Math.max(1, maxEpoch - minEpoch)) * plotWidth;
  const yFor = (value) => padding.top + (1 - ((value - minValue) / (maxValue - minValue))) * plotHeight;

  context.strokeStyle = "#d9dee7";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();

  context.fillStyle = "#607080";
  context.font = "11px sans-serif";
  context.fillText(maxValue.toFixed(2), 6, padding.top + 4);
  context.fillText(minValue.toFixed(2), 6, padding.top + plotHeight);
  context.fillText(`E${minEpoch}`, padding.left, height - 10);
  context.fillText(`E${maxEpoch}`, padding.left + plotWidth - 24, height - 10);

  series.forEach((line, index) => {
    const points = history
      .map((item) => ({ epoch: item.epoch, value: chartPoint(item[line.key]) }))
      .filter((item) => item.value !== null);
    if (!points.length) {
      return;
    }

    context.strokeStyle = line.color;
    context.lineWidth = 2;
    context.beginPath();
    points.forEach((point, pointIndex) => {
      const x = xFor(point.epoch);
      const y = yFor(point.value);
      if (pointIndex === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();

    points.forEach((point) => {
      context.fillStyle = line.color;
      context.beginPath();
      context.arc(xFor(point.epoch), yFor(point.value), 2.5, 0, Math.PI * 2);
      context.fill();
    });

    const legendX = padding.left + index * 112;
    context.fillStyle = line.color;
    context.fillRect(legendX, 6, 10, 3);
    context.fillStyle = "#17202a";
    context.font = "11px sans-serif";
    context.fillText(line.label, legendX + 14, 10);
  });
}

function renderMetricCharts(history) {
  const rows = Array.isArray(history) ? history : [];
  state.metricsHistory = rows;
  drawLineChart("accuracy-chart", rows, [
    { key: "map50", label: "mAP50", color: "#16745f" },
    { key: "map50_95", label: "mAP50-95", color: "#5b6ee1" },
    { key: "overall_f1", label: "F1", color: "#b45f06" },
  ]);
  drawLineChart("loss-chart", rows, [
    { key: "training_loss", label: "Train loss", color: "#a43d3d" },
    { key: "testing_loss", label: "Val loss", color: "#16745f" },
  ]);
}

function redrawCharts() {
  renderMetricCharts(state.metricsHistory);
}

function saveBlobWithBrowserDownload(blob, filename) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `Request failed: ${response.status}`);
  }
  return payload;
}

async function loadConfig() {
  const config = await apiJson("/api/config");
  $("data-root").textContent = `Dataset workspace: ${config.data_root}`;
  $("device").value = config.default_device || "";
  $("rf-workspace").value = config.roboflow.workspace || "";
  $("rf-project").value = config.roboflow.project || "";
  $("rf-version").value = config.roboflow.version || "";
  $("rf-format").value = config.roboflow.format || "yolov8";
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
  document.querySelectorAll(".source-view").forEach((view) => {
    view.classList.toggle("active", view.id === `source-${source}`);
  });
}

async function prepareUploadedDataset() {
  const file = $("upload-file").files[0];
  if (!file) {
    throw new Error("Choose a ZIP file first.");
  }

  const form = new FormData();
  form.append("file", file);
  form.append("classes", JSON.stringify(classNames()));
  form.append("train", $("split-train").value);
  form.append("val", $("split-val").value);
  form.append("test", $("split-test").value);
  form.append("name", $("dataset-name").value);
  form.append("force_split", $("upload-force-split").checked ? "true" : "false");

  const response = await fetch("/api/dataset/upload", {
    method: "POST",
    body: form,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Upload failed.");
  }
  return payload;
}

async function prepareFolderDataset() {
  const files = Array.from($("folder-files").files);
  if (!files.length) {
    throw new Error("Choose a dataset folder first.");
  }

  const form = new FormData();
  files.forEach((file) => {
    form.append("files", file, file.webkitRelativePath || file.name);
  });
  form.append("classes", JSON.stringify(classNames()));
  form.append("train", $("split-train").value);
  form.append("val", $("split-val").value);
  form.append("test", $("split-test").value);
  form.append("name", $("dataset-name").value);
  form.append("force_split", $("folder-force-split").checked ? "true" : "false");

  const response = await fetch("/api/dataset/folder", {
    method: "POST",
    body: form,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Folder upload failed.");
  }
  return payload;
}

async function prepareRoboflowDataset() {
  return apiJson("/api/dataset/roboflow", {
    method: "POST",
    body: JSON.stringify({
      workspace: $("rf-workspace").value,
      project: $("rf-project").value,
      version: $("rf-version").value,
      format: $("rf-format").value || "yolov8",
      classes: classNames(),
      split: splitConfig(),
      name: $("dataset-name").value,
    }),
  });
}

async function prepareDataset() {
  setMessage("Preparing dataset...");
  try {
    validateSplitTotal();

    let result;
    if (state.source === "upload") {
      result = await prepareUploadedDataset();
    } else if (state.source === "folder") {
      result = await prepareFolderDataset();
    } else {
      result = await prepareRoboflowDataset();
    }

    state.datasetYaml = result.dataset_yaml;
    $("dataset-yaml").value = result.dataset_yaml;
    setClassNames(result.classes);
    renderDatasetSummary(result.summary);
    setMessage(result.message);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function detectClasses() {
  setMessage("Detecting classes...");
  try {
    let classes = [];
    if (state.source === "folder") {
      classes = await detectClassesFromFolderSelection();
    } else {
      throw new Error("For ZIP and Roboflow, leave Class IDs empty and click Prepare Dataset to auto-fill from data.yaml.");
    }

    setClassNames(classes);
    setMessage(`Detected ${classes.length} class names.`);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function startTraining() {
  const datasetYaml = $("dataset-yaml").value || state.datasetYaml;
  if (!datasetYaml) {
    setMessage("Prepare a dataset first.", true);
    return;
  }

  setMessage("Starting training...");
  try {
    const result = await apiJson("/api/train/start", {
      method: "POST",
      body: JSON.stringify({
        dataset_yaml: datasetYaml,
        model_size: $("model-size").value,
        epochs: numberValue("epochs"),
        imgsz: numberValue("imgsz"),
        batch: numberValue("batch"),
        patience: numberValue("patience"),
        save_period: numberValue("save-period"),
        device: $("device").value || null,
        workers: numberValue("workers"),
        optimizer: $("optimizer").value,
        lr0: numberValue("lr0"),
        lrf: numberValue("lrf"),
        weight_decay: numberValue("weight-decay"),
        cos_lr: $("cos-lr").checked,
        warmup_epochs: numberValue("warmup-epochs"),
        freeze: optionalNumberValue("freeze"),
        pretrained: $("pretrained").checked,
        activation: $("activation").value,
        exist_ok: $("exist-ok").checked,
        seed: numberValue("seed"),
        project: $("project").value,
        name: $("run-name").value,
        resume: $("resume").checked,
      }),
    });
    setMessage(`${result.message}\nPID: ${result.pid}`);
    pollStatus();
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function stopTraining() {
  try {
    const result = await apiJson("/api/train/stop", { method: "POST", body: "{}" });
    setMessage(result.message);
    pollStatus();
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function refreshWeightsStatus() {
  try {
    const status = await apiJson("/api/train/weights/status", {
      method: "POST",
      body: JSON.stringify(weightTarget()),
    });
    $("download-best").disabled = !status.best.available;
    $("download-last").disabled = !status.last.available;

    const available = ["best", "last"].filter((weight) => status[weight].available);
    if (available.length) {
      const sizes = available.map((weight) => `${weight}.pt ${formatBytes(status[weight].size)}`);
      const runText = status.run_dir ? ` from ${status.run_dir}` : "";
      $("weights-status").textContent = `Available: ${sizes.join(", ")}${runText}`;
    } else {
      $("weights-status").textContent = "No trained weights found for this run yet.";
    }
  } catch (error) {
    $("download-best").disabled = true;
    $("download-last").disabled = true;
    $("weights-status").textContent = error.message;
  }
}

async function refreshMetrics() {
  try {
    const metrics = await apiJson("/api/train/metrics", {
      method: "POST",
      body: JSON.stringify(weightTarget()),
    });

    if (!metrics.available) {
      state.metricsAvailable = false;
      $("metric-f1").textContent = "-";
      $("metric-weighted-f1").textContent = "-";
      $("metric-train-loss").textContent = "-";
      $("metric-test-loss").textContent = "-";
      $("metric-map50").textContent = "-";
      $("metric-map").textContent = "-";
      renderBestMetrics(null);
      renderClassMetrics([]);
      resetCharts();
      setArtifactButtons(metrics.artifacts || false);
      $("metrics-status").textContent = "No results.csv found for this run yet.";
      return;
    }

    state.metricsAvailable = true;
    $("metric-f1").textContent = metricText(metrics.overall_f1);
    $("metric-weighted-f1").textContent = metricText(metrics.weighted_f1);
    $("metric-train-loss").textContent = metricText(metrics.training_loss);
    $("metric-test-loss").textContent = metricText(metrics.testing_loss);
    $("metric-map50").textContent = metricText(metrics.map50);
    $("metric-map").textContent = metricText(metrics.map50_95);
    renderBestMetrics(metrics.best);
    renderClassMetrics(metrics.per_class);
    renderMetricCharts(metrics.history);
    setArtifactButtons(metrics.artifacts || true);
    $("metrics-status").textContent = `Epoch ${metrics.epoch}. ${metrics.note}`;
  } catch (error) {
    resetCharts();
    setArtifactButtons(false);
    $("metrics-status").textContent = error.message;
  }
}

async function downloadWeight(weight) {
  try {
    const response = await fetch(weightDownloadUrl(weight));
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Download failed: ${response.status}`);
    }

    const blob = await response.blob();
    const filename = `${weight}.pt`;

    if ("showDirectoryPicker" in window) {
      try {
        const directory = await window.showDirectoryPicker();
        const file = await directory.getFileHandle(filename, { create: true });
        const writable = await file.createWritable();
        await writable.write(blob);
        await writable.close();
        setMessage(`Saved ${filename}.`);
        return;
      } catch (error) {
        if (error.name === "AbortError") {
          return;
        }
        if (error.name !== "SecurityError") {
          throw error;
        }
      }
    }

    saveBlobWithBrowserDownload(blob, filename);
    setMessage(`Downloading ${filename}.`);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    setMessage(error.message, true);
  }
}

async function downloadArtifact(artifact, filename) {
  try {
    const response = await fetch("/api/train/artifacts/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...weightTarget(), artifact }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Download failed: ${response.status}`);
    }
    saveBlobWithBrowserDownload(await response.blob(), filename);
    setMessage(`Downloading ${filename}.`);
  } catch (error) {
    setMessage(error.message, true);
  }
}

function logEndpoint() {
  if (state.logMode === "full") {
    return "/api/train/logs/full";
  }
  if (state.logMode === "errors") {
    return "/api/train/logs/errors";
  }
  return "/api/train/logs";
}

function setLogMode(mode) {
  state.logMode = mode;
  document.querySelectorAll("[data-log-mode]").forEach((button) => {
    button.classList.toggle("active-control", button.dataset.logMode === mode);
  });
  refreshLogs();
}

function downloadLog(url) {
  window.location.href = url;
}

async function pollStatus() {
  try {
    const status = await apiJson("/api/train/status");
    $("status-pill").textContent = status.running ? "Training" : "Idle";
    if (state.logMode === "recent") {
      $("logs").textContent = status.log_tail || "";
    }
    $("log-status").textContent = status.history_log_file
      ? `Current log: ${status.log_file} | Run log: ${status.history_log_file}`
      : `Current log: ${status.log_file}`;
    refreshWeightsStatus();
    refreshMetrics();
  } catch (error) {
    $("status-pill").textContent = "Error";
    setMessage(error.message, true);
  }
}

async function refreshLogs() {
  const response = await fetch(logEndpoint());
  const text = await response.text();
  $("logs").textContent = text || (state.logMode === "errors" ? "No warnings or errors found." : "");
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});

$("prepare-dataset").addEventListener("click", prepareDataset);
$("detect-classes").addEventListener("click", detectClasses);
$("start-training").addEventListener("click", startTraining);
$("stop-training").addEventListener("click", stopTraining);
$("refresh-logs").addEventListener("click", refreshLogs);
$("refresh-metrics").addEventListener("click", refreshMetrics);
$("download-best").addEventListener("click", () => downloadWeight("best"));
$("download-last").addEventListener("click", () => downloadWeight("last"));
$("download-results-csv").addEventListener("click", () => downloadArtifact("results_csv", "results.csv"));
$("download-accuracy-graph").addEventListener("click", () => downloadArtifact("accuracy_graph", "accuracy_by_epoch.png"));
$("download-loss-graph").addEventListener("click", () => downloadArtifact("loss_graph", "loss_by_epoch.png"));
$("download-current-log").addEventListener("click", () => downloadLog("/api/train/logs/download"));
$("download-history-log").addEventListener("click", () => downloadLog("/api/train/logs/history/download"));
$("project").addEventListener("input", refreshWeightsStatus);
$("run-name").addEventListener("input", refreshWeightsStatus);
$("project").addEventListener("input", refreshMetrics);
$("run-name").addEventListener("input", refreshMetrics);
document.querySelectorAll("[data-log-mode]").forEach((button) => {
  button.addEventListener("click", () => setLogMode(button.dataset.logMode));
});
document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});
$("reset-advanced").addEventListener("click", () => {
  applyControlValues(CONTROL_DEFAULTS);
  setMessage("Reset training controls to defaults.");
});
window.addEventListener("resize", redrawCharts);

loadConfig().catch((error) => setMessage(error.message, true));
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
