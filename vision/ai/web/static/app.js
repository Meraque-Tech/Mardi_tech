const state = {
  source: "upload",
  datasetYaml: "",
  datasetSummary: null,
  pollTimer: null,
  metricsHistory: [],
  metricsAvailable: false,
  logMode: "recent",
  activePreset: null,
  isPreparing: false,
  isDetecting: false,
  isStarting: false,
  isStopping: false,
  stopRequested: false,
  running: false,
  pollInFlight: false,
  targetRevision: 0,
  targetTimer: null,
  lastDataRefresh: 0,
  resizeFrame: null,
  downloads: new Set(),
  folderTooLarge: false,
  resolvedRunPath: "",
  runResolutionType: "not_found",
  trainingStarted: false,
  trainingCompleted: false,
  trainingOutcome: "",
  datasetNameEdited: false,
  resumeAvailable: false,
  resumeCheckpoint: "",
};

const $ = (id) => document.getElementById(id);

const STATUS_LABELS = {
  idle: "Idle",
  preparing: "Preparing",
  starting: "Starting",
  training: "Training",
  stopping: "Stopping",
  stopped: "Stopped",
  completed: "Completed",
  failed: "Failed",
  error: "Error",
};

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
  optimizer: "Adam",
  seed: 42,
  lr0: 0.001,
  lrf: 0.01,
  "weight-decay": 0.0005,
  "warmup-epochs": 3.0,
  freeze: "",
  "cos-lr": false,
  activation: "silu",
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
    activation: "silu",
  },
};

function setMessage(text, isError = false) {
  const message = $("message");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setStatusPhase(phase) {
  const status = $("status-pill");
  status.textContent = STATUS_LABELS[phase] || phase;
  status.className = `status-pill status-${phase}`;
}

function hasSelectedDatasetSource() {
  if (state.source === "upload") {
    return Boolean($("upload-file").files[0]);
  }
  if (state.source === "folder") {
    return Boolean($("folder-files").files.length) && !state.folderTooLarge;
  }
  return Boolean($("rf-project").value.trim() && $("rf-version").value.trim());
}

function syncTrainingGuide() {
  const hasDataset = Boolean(state.datasetYaml);
  let currentStep = 1;
  if (hasSelectedDatasetSource()) {
    currentStep = 2;
  }
  if (hasDataset) {
    currentStep = 3;
  }
  if (state.trainingStarted || state.running) {
    currentStep = 4;
  }
  if (state.trainingCompleted) {
    currentStep = 5;
  }

  document.querySelectorAll("[data-guide-step]").forEach((step) => {
    const number = Number(step.dataset.guideStep);
    const completed = number < currentStep;
    const active = number === currentStep;
    step.classList.toggle("completed", completed);
    step.classList.toggle("active", active);
    if (active) {
      step.setAttribute("aria-current", "step");
    } else {
      step.removeAttribute("aria-current");
    }
    step.querySelector(".guide-step-marker").textContent = completed ? "✓" : String(number);
  });
  $("guide-progress").textContent = `Step ${currentStep} of 5`;
}

function syncActionStates() {
  const locked = state.running || state.isStarting || state.isStopping;
  const preparing = state.isPreparing || state.isDetecting;
  const hasDataset = Boolean(state.datasetYaml);
  const canResume = state.resumeAvailable && $("resume").checked;

  const invalidFolderSelection = state.source === "folder" && state.folderTooLarge;
  $("prepare-dataset").disabled = locked || preparing || invalidFolderSelection;
  $("prepare-dataset").textContent = state.isPreparing ? "Preparing..." : "Prepare Dataset";
  $("prepare-dataset").setAttribute("aria-busy", String(state.isPreparing));
  $("detect-classes").disabled = locked || preparing;
  $("detect-classes").textContent = state.isDetecting ? "Detecting..." : "Auto Fetch";
  $("detect-classes").setAttribute("aria-busy", String(state.isDetecting));
  $("start-training").disabled = locked || preparing || (!hasDataset && !canResume);
  $("start-training").textContent = state.isStarting ? "Starting..." : "Start";
  $("start-training").setAttribute("aria-busy", String(state.isStarting));
  $("stop-training").disabled = !state.running || state.isStopping;
  $("stop-training").textContent = state.isStopping ? "Stopping..." : "Stop";
  $("stop-training").setAttribute("aria-busy", String(state.isStopping));

  document.querySelectorAll(".training-panel input, .training-panel select, .advanced-panel input, .advanced-panel select").forEach((control) => {
    control.disabled = locked;
  });
  $("resume").disabled = locked || !state.resumeAvailable;
  document.querySelectorAll("[data-preset], #reset-advanced").forEach((button) => {
    button.disabled = locked;
  });

  if (state.isPreparing) {
    setStatusPhase("preparing");
  } else if (state.isStarting) {
    setStatusPhase("starting");
  } else if (state.isStopping) {
    setStatusPhase("stopping");
  } else if (state.running) {
    setStatusPhase("training");
  }
  syncTrainingGuide();
}

function usesCustomSplit() {
  if (state.source === "upload") {
    return $("upload-force-split").checked;
  }
  if (state.source === "folder") {
    return $("folder-force-split").checked;
  }
  return false;
}

function cleanDatasetName(value) {
  return String(value || "")
    .replace(/\.zip$/i, "")
    .trim()
    .replace(/[^A-Za-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function suggestedDatasetName() {
  if (state.source === "upload") {
    return cleanDatasetName($("upload-file").files[0]?.name);
  }
  if (state.source === "folder") {
    const file = $("folder-files").files[0];
    const path = file?.webkitRelativePath || file?.name || "";
    return cleanDatasetName(path.split("/")[0]);
  }
  return cleanDatasetName($("rf-project").value);
}

function updateDatasetNameSuggestion() {
  if (state.datasetNameEdited) {
    return;
  }
  $("dataset-name").value = suggestedDatasetName() || "dataset";
}

function syncDatasetSourceControls() {
  const isFolder = state.source === "folder";
  const splitEnabled = usesCustomSplit();
  $("detect-classes").hidden = !isFolder;
  $("split-controls").hidden = !splitEnabled;

  const classNotes = {
    upload: "For ZIP uploads, leave this empty to read class names from data.yaml.",
    folder: "Use Auto Fetch to read class names from data.yaml, or enter one class per line.",
    roboflow: "Leave this empty to use the class names supplied by the Roboflow dataset version.",
  };
  $("classes-note").textContent = classNotes[state.source];
  updateDatasetNameSuggestion();
  if (splitEnabled) {
    updateSplitTotal();
  }
}

function setResumeAvailability(available, checkpoint = "") {
  state.resumeAvailable = Boolean(available);
  state.resumeCheckpoint = state.resumeAvailable ? checkpoint : "";
  if (!state.resumeAvailable) {
    $("resume").checked = false;
  }
  $("resume-status").textContent = state.resumeAvailable
    ? `Available checkpoint: ${state.resumeCheckpoint}`
    : "No resumable checkpoint found for this run.";
  syncActionStates();
}

function setActivePreset(name) {
  state.activePreset = name || null;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    const selected = Boolean(name && button.dataset.preset === name);
    button.setAttribute("aria-pressed", String(selected));
  });
  $("preset-status").textContent = name
    ? `${name.replace("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase())} preset selected`
    : "Custom settings";
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
  updateCurrentRunDisplay();
  syncActionStates();
  if (Object.hasOwn(values, "project") || Object.hasOwn(values, "run-name")) {
    scheduleTargetRefresh();
  }
}

function applyPreset(name) {
  const preset = TRAINING_PRESETS[name];
  if (!preset) {
    return;
  }
  applyControlValues(preset);
  setActivePreset(name);
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

function updateSplitTotal() {
  const split = splitConfig();
  const values = [split.train, split.val, split.test];
  const total = values.reduce((sum, value) => sum + value, 0);
  const validRanges = values.every((value) => Number.isFinite(value) && value >= 0 && value <= 100) && split.train > 0;
  const isValid = validRanges && total === 100;
  const indicator = $("split-total");
  indicator.textContent = Number.isFinite(total) ? `Total: ${total}%` : "Total: invalid";
  indicator.classList.toggle("valid", isValid);
  indicator.classList.toggle("invalid", !isValid);
  return isValid;
}

function validateSplitTotal() {
  const split = splitConfig();
  const total = split.train + split.val + split.test;
  if (!updateSplitTotal() || total !== 100) {
    throw new Error("Train, validation, and test splits must total 100%.");
  }
}

function weightTarget() {
  return {
    project: $("project").value || "runs/detect",
    name: $("run-name").value || "train",
  };
}

function normalizedPath(value) {
  return String(value || "").replace(/\\/g, "/").replace(/\/+$/, "");
}

function resolvedPathDiffers(target, runDir, resolutionType) {
  if (!runDir || resolutionType === "exact") {
    return false;
  }
  if (resolutionType === "legacy") {
    return true;
  }

  const selected = normalizedPath(`${target.project}/${target.name}`).replace(/^\.\//, "");
  const resolved = normalizedPath(runDir);
  return resolved !== selected && !resolved.endsWith(`/${selected}`);
}

function updateCurrentRunDisplay(details = null) {
  const target = weightTarget();
  if (details) {
    const runDir = details.run_dir || details.runDir || "";
    const resolutionType = details.resolution_type || details.resolutionType || (runDir ? "legacy" : "not_found");
    state.resolvedRunPath = runDir;
    state.runResolutionType = resolutionType;
  }

  $("current-run-name").textContent = target.name;
  $("current-run-path").textContent = `Selected: ${target.project}/${target.name}`;

  const resolved = $("resolved-run-path");
  const showResolved = resolvedPathDiffers(
    target,
    state.resolvedRunPath,
    state.runResolutionType,
  );
  resolved.hidden = !showResolved;
  if (showResolved) {
    const label = state.runResolutionType === "legacy" ? "Legacy run" : "Actual output";
    resolved.textContent = `${label}: ${state.resolvedRunPath}`;
  } else {
    resolved.textContent = "";
  }
}

function renderEpochProgress(progress = {}, phase = "idle") {
  const current = Math.max(0, Number(progress.current) || 0);
  const completed = Math.max(0, Number(progress.completed) || 0);
  const total = Math.max(0, Number(progress.total) || 0);
  const percent = total
    ? Math.min(100, Math.max(0, Number(progress.percent) || (current / total) * 100))
    : 0;

  let label = "Waiting to start";
  let detail = "The current epoch will appear here when training starts.";
  if (phase === "starting") {
    label = total ? `Starting a ${total}-epoch run` : "Starting training";
    detail = "Loading the model and preparing the dataloaders.";
  } else if (phase === "training") {
    label = current && total ? `Epoch ${current} of ${total}` : "Starting first epoch";
    detail = completed
      ? `${completed} ${completed === 1 ? "epoch" : "epochs"} completed.`
      : "The first epoch is in progress.";
  } else if (phase === "completed") {
    label = total ? `Completed ${completed || current} of ${total}` : "Training completed";
    detail = completed < total
      ? "Training finished early using the configured stopping criteria."
      : "All configured epochs completed.";
  } else if (phase === "stopped" || phase === "failed") {
    const verb = phase === "stopped" ? "Stopped" : "Failed";
    label = current && total ? `${verb} during epoch ${current} of ${total}` : `${verb} before the first epoch`;
    detail = `${completed} ${completed === 1 ? "epoch was" : "epochs were"} fully completed.`;
  } else if (current && total) {
    label = `Last recorded epoch ${current} of ${total}`;
    detail = `${completed} ${completed === 1 ? "epoch was" : "epochs were"} fully completed.`;
  }

  const roundedPercent = Math.round(percent);
  $("epoch-progress-label").textContent = label;
  $("epoch-progress-percent").textContent = `${roundedPercent}%`;
  $("epoch-progress-fill").style.width = `${percent}%`;
  $("epoch-progress-detail").textContent = detail;
  const track = $("epoch-progress-track");
  track.setAttribute("aria-valuenow", String(roundedPercent));
  track.setAttribute("aria-valuetext", label);
}

function refreshTargetData() {
  const revision = state.targetRevision;
  const target = weightTarget();
  updateCurrentRunDisplay();
  return Promise.all([
    refreshWeightsStatus(target, revision),
    refreshMetrics(target, revision),
  ]);
}

function scheduleTargetRefresh() {
  window.clearTimeout(state.targetTimer);
  state.targetRevision += 1;
  state.resolvedRunPath = "";
  state.runResolutionType = "not_found";
  setResumeAvailability(false);
  updateCurrentRunDisplay();
  state.targetTimer = window.setTimeout(refreshTargetData, 400);
}

function weightDownloadUrl(weight) {
  const params = new URLSearchParams(weightTarget());
  return `/api/train/weights/${weight}?${params.toString()}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes)) || Number(bytes) < 0) {
    return "Unknown size";
  }
  if (Number(bytes) === 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = Number(bytes);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function updateFileSelection() {
  const zip = $("upload-file").files[0];
  $("upload-selection").textContent = zip
    ? `${zip.name} (${formatBytes(zip.size)})`
    : "No ZIP selected.";

  const folderFiles = Array.from($("folder-files").files);
  if (!folderFiles.length) {
    state.folderTooLarge = false;
    $("folder-selection").classList.remove("error");
    $("folder-selection").textContent = "No folder selected. Folder upload supports up to 1,000 files; use ZIP for larger datasets.";
    updateDatasetNameSuggestion();
    syncActionStates();
    return;
  }
  const totalSize = folderFiles.reduce((sum, file) => sum + file.size, 0);
  const firstPath = folderFiles[0].webkitRelativePath || folderFiles[0].name;
  const folderName = firstPath.split("/")[0];
  state.folderTooLarge = folderFiles.length > 1000;
  $("folder-selection").classList.toggle("error", state.folderTooLarge);
  const limitNote = folderFiles.length > 1000
    ? " Too many files for folder upload; use Upload ZIP."
    : " Folder upload supports up to 1,000 files.";
  $("folder-selection").textContent = `${folderName}: ${folderFiles.length} files (${formatBytes(totalSize)}).${limitNote}`;
  updateDatasetNameSuggestion();
  syncActionStates();
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
  const distribution = Array.isArray(summary.class_distribution) ? summary.class_distribution : [];
  const totalInstances = distribution.reduce((sum, item) => sum + (Number(item.instances) || 0), 0);
  const distributionRows = distribution.map((item) => {
    const instances = Number(item.instances) || 0;
    const images = Number(item.images) || 0;
    const share = totalInstances ? (instances / totalInstances) * 100 : 0;
    return `
      <tr>
        <td>${escapeHtml(item.class_name)}</td>
        <td>${images}</td>
        <td>${instances}</td>
        <td>
          <div class="distribution-share">
            <span><i style="width: ${share.toFixed(2)}%"></i></span>
            <small>${share.toFixed(1)}%</small>
          </div>
        </td>
      </tr>
    `;
  }).join("");
  const distributionTable = distributionRows
    ? `
      <div class="dataset-distribution">
        <h4>Dataset Distribution</h4>
        <table>
          <thead><tr><th>Class</th><th>Images</th><th>Instances</th><th>Share</th></tr></thead>
          <tbody>${distributionRows}</tbody>
          <tfoot><tr><th>Total</th><td></td><th>${totalInstances}</th><td></td></tr></tfoot>
        </table>
      </div>
    `
    : "<p>No class distribution is available.</p>";
  const warnings = Array.isArray(summary.warnings) && summary.warnings.length
    ? `<ul>${summary.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "<p>No dataset warnings found.</p>";

  container.innerHTML = `
    <details class="dataset-summary-details" open>
      <summary>
        <span>Dataset Summary</span>
        <small>${summary.total_images || 0} images, ${summary.class_count || 0} classes</small>
      </summary>
      <div class="dataset-summary-content">
        <div class="summary-grid">${splitRows}</div>
        ${distributionTable}
        ${warnings}
      </div>
    </details>
  `;
}

function formatBestMetric(row, key, label) {
  if (!row || row[key] === null || row[key] === undefined) {
    return "";
  }
  return `<div><span>${label}</span><strong>${metricText(row[key])}</strong><small>Epoch ${row.epoch}</small></div>`;
}

function renderBestMetrics(best, history = []) {
  const container = $("best-metrics");
  if (!best) {
    container.innerHTML = "";
    return;
  }
  const summary = { ...best };
  if (!summary.lowest_training_loss && Array.isArray(history)) {
    const candidates = history.filter((row) => row.training_loss !== null && row.training_loss !== undefined);
    if (candidates.length) {
      summary.lowest_training_loss = candidates.reduce((lowest, row) => (
        row.training_loss < lowest.training_loss ? row : lowest
      ));
    }
  }
  const rows = [
    formatBestMetric(summary.best_map50_95, "map50_95", "Best mAP50-95"),
    formatBestMetric(summary.best_map50, "map50", "Best mAP50"),
    formatBestMetric(summary.best_f1, "overall_f1", "Best F1"),
    formatBestMetric(summary.lowest_training_loss, "training_loss", "Lowest training loss"),
    formatBestMetric(summary.lowest_validation_loss, "testing_loss", "Lowest validation loss"),
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
  $("download-results-csv").disabled = !isEnabled("results_csv") || state.downloads.has("results_csv");
  $("download-accuracy-graph").disabled = !isEnabled("accuracy_graph") || state.downloads.has("accuracy_graph");
  $("download-loss-graph").disabled = !isEnabled("loss_graph") || state.downloads.has("loss_graph");
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
  updateDatasetNameSuggestion();
  syncTrainingGuide();
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".tab").forEach((button) => {
    const active = button.dataset.source === source;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll(".source-view").forEach((view) => {
    const active = view.id === `source-${source}`;
    view.classList.toggle("active", active);
    view.hidden = !active;
  });
  syncDatasetSourceControls();
  syncActionStates();
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
      classes: classNames(),
      name: $("dataset-name").value,
    }),
  });
}

async function prepareDataset() {
  if (state.isPreparing) {
    return;
  }
  state.isPreparing = true;
  syncActionStates();
  setMessage("Preparing dataset...");
  try {
    if (usesCustomSplit()) {
      validateSplitTotal();
    }

    let result;
    if (state.source === "upload") {
      result = await prepareUploadedDataset();
    } else if (state.source === "folder") {
      result = await prepareFolderDataset();
    } else {
      result = await prepareRoboflowDataset();
    }

    state.datasetYaml = result.dataset_yaml;
    setClassNames(result.classes);
    renderDatasetSummary(result.summary);
    setMessage(result.message);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.isPreparing = false;
    if (!state.running) {
      setStatusPhase("idle");
    }
    syncActionStates();
  }
}

async function detectClasses() {
  if (state.isDetecting) {
    return;
  }
  state.isDetecting = true;
  syncActionStates();
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
  } finally {
    state.isDetecting = false;
    if (!state.running) {
      setStatusPhase("idle");
    }
    syncActionStates();
  }
}

async function startTraining() {
  const datasetYaml = state.datasetYaml;
  const resume = $("resume").checked;
  if (!datasetYaml && !resume) {
    setMessage("Prepare a dataset first.", true);
    return;
  }

  if (state.isStarting || state.running) {
    return;
  }
  if (resume && !state.resumeAvailable) {
    setMessage("No last.pt checkpoint is available for the selected project and run name.", true);
    return;
  }
  state.isStarting = true;
  state.trainingOutcome = "";
  setActivePreset(state.activePreset);
  syncActionStates();
  renderEpochProgress({ total: numberValue("epochs") }, "starting");
  setMessage("Starting training...");
  try {
    const result = await apiJson("/api/train/start", {
      method: "POST",
      body: JSON.stringify({
        dataset_yaml: resume ? null : datasetYaml,
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
        activation: $("activation").value,
        exist_ok: $("exist-ok").checked,
        seed: numberValue("seed"),
        project: $("project").value,
        name: $("run-name").value,
        resume,
      }),
    });
    state.running = true;
    state.trainingStarted = true;
    state.trainingCompleted = false;
    state.lastDataRefresh = 0;
    setMessage(`${result.message}\nPID: ${result.pid}`);
    updateCurrentRunDisplay({ ...(result.training_run || {}), running: true });
    pollStatus();
  } catch (error) {
    setMessage(error.message, true);
    setStatusPhase("failed");
  } finally {
    state.isStarting = false;
    syncActionStates();
  }
}

async function stopTraining() {
  if (!state.running || state.isStopping) {
    return;
  }
  state.isStopping = true;
  state.stopRequested = true;
  syncActionStates();
  try {
    const result = await apiJson("/api/train/stop", { method: "POST", body: "{}" });
    setMessage(result.message);
    pollStatus();
  } catch (error) {
    state.stopRequested = false;
    setMessage(error.message, true);
    setStatusPhase("error");
  } finally {
    state.isStopping = false;
    syncActionStates();
  }
}

async function refreshWeightsStatus(target = weightTarget(), revision = state.targetRevision) {
  try {
    const status = await apiJson("/api/train/weights/status", {
      method: "POST",
      body: JSON.stringify(target),
    });
    if (revision !== state.targetRevision) {
      return;
    }
    updateCurrentRunDisplay({
      run_dir: status.run_dir,
      resolution_type: status.resolution_type,
    });
    $("download-best").disabled = !status.best.available || state.downloads.has("best");
    $("download-last").disabled = !status.last.available || state.downloads.has("last");
    setResumeAvailability(status.last.available, status.last.path);

    const available = ["best", "last"].filter((weight) => status[weight].available);
    if (available.length) {
      const sizes = available.map((weight) => `${weight}.pt ${formatBytes(status[weight].size)}`);
      const runText = status.run_dir ? ` from ${status.run_dir}` : "";
      $("weights-status").textContent = `Available: ${sizes.join(", ")}${runText}`;
    } else {
      $("weights-status").textContent = "No trained weights found for this run yet.";
    }
  } catch (error) {
    if (revision !== state.targetRevision) {
      return;
    }
    $("download-best").disabled = true;
    $("download-last").disabled = true;
    setResumeAvailability(false);
    $("weights-status").textContent = error.message;
  }
}

async function refreshMetrics(target = weightTarget(), revision = state.targetRevision) {
  try {
    const metrics = await apiJson("/api/train/metrics", {
      method: "POST",
      body: JSON.stringify(target),
    });
    if (revision !== state.targetRevision) {
      return;
    }
    updateCurrentRunDisplay({
      run_dir: metrics.run_dir,
      resolution_type: metrics.resolution_type,
    });

    if (!metrics.available) {
      state.metricsAvailable = false;
      $("training-results-panel").classList.remove("has-results");
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
    $("training-results-panel").classList.add("has-results");
    $("metric-f1").textContent = metricText(metrics.overall_f1);
    $("metric-weighted-f1").textContent = metricText(metrics.weighted_f1);
    $("metric-train-loss").textContent = metricText(metrics.training_loss);
    $("metric-test-loss").textContent = metricText(metrics.testing_loss);
    $("metric-map50").textContent = metricText(metrics.map50);
    $("metric-map").textContent = metricText(metrics.map50_95);
    renderBestMetrics(metrics.best, metrics.history);
    renderClassMetrics(metrics.per_class);
    renderMetricCharts(metrics.history);
    setArtifactButtons(metrics.artifacts || true);
    $("metrics-status").textContent = `Epoch ${metrics.epoch}. ${metrics.note}`;
  } catch (error) {
    if (revision !== state.targetRevision) {
      return;
    }
    $("training-results-panel").classList.remove("has-results");
    resetCharts();
    setArtifactButtons(false);
    $("metrics-status").textContent = error.message;
  }
}

async function refreshMetricsFromButton() {
  const button = $("refresh-metrics");
  button.disabled = true;
  button.textContent = "Refreshing...";
  try {
    await refreshMetrics();
  } finally {
    button.disabled = false;
    button.textContent = "Refresh";
  }
}

async function downloadWeight(weight) {
  const button = $(weight === "best" ? "download-best" : "download-last");
  const originalText = button.textContent;
  state.downloads.add(weight);
  button.disabled = true;
  button.textContent = "Downloading...";
  try {
    const filename = `${weight}.pt`;
    let directory = null;
    if ("showDirectoryPicker" in window) {
      try {
        directory = await window.showDirectoryPicker();
      } catch (error) {
        if (error.name === "AbortError") {
          return;
        }
        if (error.name !== "SecurityError") {
          throw error;
        }
      }
    }

    setMessage(`Preparing ${filename}...`);
    const response = await fetch(weightDownloadUrl(weight));
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Download failed: ${response.status}`);
    }

    const blob = await response.blob();
    if (directory) {
      const file = await directory.getFileHandle(filename, { create: true });
      const writable = await file.createWritable();
      await writable.write(blob);
      await writable.close();
      setMessage(`Saved ${filename}.`);
      return;
    }

    saveBlobWithBrowserDownload(blob, filename);
    setMessage(`Downloading ${filename}.`);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    setMessage(error.message, true);
  } finally {
    state.downloads.delete(weight);
    button.textContent = originalText;
    await refreshWeightsStatus();
  }
}

async function downloadArtifact(artifact, filename) {
  const buttonIds = {
    results_csv: "download-results-csv",
    accuracy_graph: "download-accuracy-graph",
    loss_graph: "download-loss-graph",
  };
  const button = $(buttonIds[artifact]);
  const originalText = button.textContent;
  state.downloads.add(artifact);
  button.disabled = true;
  button.textContent = "Downloading...";
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
  } finally {
    state.downloads.delete(artifact);
    button.textContent = originalText;
    await refreshMetrics();
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
    const active = button.dataset.logMode === mode;
    button.classList.toggle("active-control", active);
    button.setAttribute("aria-pressed", String(active));
  });
  refreshLogs();
}

function downloadLog(url) {
  window.location.href = url;
}

async function pollStatus() {
  if (state.pollInFlight) {
    return;
  }
  state.pollInFlight = true;
  try {
    const status = await apiJson("/api/train/status");
    const wasRunning = state.running;
    state.running = Boolean(status.running);

    if (state.running) {
      state.trainingStarted = true;
      state.trainingOutcome = "";
      setStatusPhase("training");
    } else if (wasRunning && state.stopRequested) {
      state.stopRequested = false;
      state.trainingOutcome = "stopped";
      setStatusPhase("stopped");
      setMessage("Training stopped by user. The latest available checkpoint remains in the run folder.");
    } else if (wasRunning && status.returncode === 0) {
      state.trainingCompleted = true;
      state.trainingOutcome = "completed";
      setStatusPhase("completed");
      setMessage("Training completed. Results and model weights are ready to review.");
    } else if (wasRunning && status.returncode !== null && status.returncode !== 0) {
      state.trainingOutcome = "failed";
      setStatusPhase("failed");
      setMessage(`Training stopped with exit code ${status.returncode}. Review the warnings and full log.`, true);
    } else if (!state.isPreparing && !state.isStarting && !state.isStopping) {
      const currentPhase = $("status-pill").className;
      if (!["status-completed", "status-failed", "status-stopped"].some((name) => currentPhase.includes(name))) {
        setStatusPhase("idle");
      }
    }
    syncActionStates();
    renderEpochProgress(
      status.epoch_progress,
      state.running ? "training" : (state.trainingOutcome || "idle"),
    );

    if (state.logMode === "recent") {
      $("logs").textContent = status.log_tail || "";
    }
    if (state.running && status.training_run && Object.keys(status.training_run).length) {
      updateCurrentRunDisplay({ ...status.training_run, running: status.running });
    } else if (wasRunning && status.training_run && Object.keys(status.training_run).length) {
      updateCurrentRunDisplay(status.training_run);
    }
    $("log-status").textContent = status.history_log_file
      ? `Current log: ${status.log_file} | Run log: ${status.history_log_file}`
      : `Current log: ${status.log_file}`;
    const now = Date.now();
    if (now - state.lastDataRefresh >= 6000 || wasRunning !== state.running) {
      state.lastDataRefresh = now;
      const revision = state.targetRevision;
      const target = weightTarget();
      await Promise.all([
        refreshWeightsStatus(target, revision),
        refreshMetrics(target, revision),
      ]);
    }
  } catch (error) {
    setStatusPhase("error");
    setMessage(error.message, true);
  } finally {
    state.pollInFlight = false;
  }
}

async function refreshLogs() {
  const button = $("refresh-logs");
  button.disabled = true;
  button.textContent = "Refreshing...";
  try {
    const response = await fetch(logEndpoint());
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text || `Log request failed: ${response.status}`);
    }
    $("logs").textContent = text || (state.logMode === "errors" ? "No warnings or errors found." : "No log output yet.");
  } catch (error) {
    $("log-status").textContent = `Unable to load logs: ${error.message}`;
    setMessage(`Unable to load logs: ${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "Refresh";
  }
}

function handleTabKeydown(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    return;
  }
  const tabs = Array.from(document.querySelectorAll(".source-tabs [role='tab']"));
  const currentIndex = tabs.indexOf(event.currentTarget);
  let nextIndex = currentIndex;
  if (event.key === "ArrowRight") {
    nextIndex = (currentIndex + 1) % tabs.length;
  } else if (event.key === "ArrowLeft") {
    nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
  } else if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = tabs.length - 1;
  }
  event.preventDefault();
  tabs[nextIndex].focus();
  setSource(tabs[nextIndex].dataset.source);
}

function positionTooltip(element) {
  const rect = element.getBoundingClientRect();
  const tooltipWidth = Math.min(280, window.innerWidth - 48);
  element.classList.toggle("tooltip-align-right", rect.left + tooltipWidth > window.innerWidth - 16);
  element.classList.toggle("tooltip-below", rect.top < 100);
}

function initializeTooltips() {
  document.querySelectorAll(".tooltip-label").forEach((element) => {
    element.addEventListener("mouseenter", () => positionTooltip(element));
    element.addEventListener("focus", () => positionTooltip(element));
  });
}

function redrawChartsSoon() {
  if (state.resizeFrame) {
    window.cancelAnimationFrame(state.resizeFrame);
  }
  state.resizeFrame = window.requestAnimationFrame(() => {
    state.resizeFrame = null;
    redrawCharts();
  });
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
  button.addEventListener("keydown", handleTabKeydown);
});

$("prepare-dataset").addEventListener("click", prepareDataset);
$("detect-classes").addEventListener("click", detectClasses);
$("start-training").addEventListener("click", startTraining);
$("stop-training").addEventListener("click", stopTraining);
$("refresh-logs").addEventListener("click", refreshLogs);
$("refresh-metrics").addEventListener("click", refreshMetricsFromButton);
$("download-best").addEventListener("click", () => downloadWeight("best"));
$("download-last").addEventListener("click", () => downloadWeight("last"));
$("download-results-csv").addEventListener("click", () => downloadArtifact("results_csv", "results.csv"));
$("download-accuracy-graph").addEventListener("click", () => downloadArtifact("accuracy_graph", "accuracy_by_epoch.png"));
$("download-loss-graph").addEventListener("click", () => downloadArtifact("loss_graph", "loss_by_epoch.png"));
$("download-run-log").addEventListener("click", () => downloadLog("/api/train/logs/download"));
$("project").addEventListener("input", scheduleTargetRefresh);
$("run-name").addEventListener("input", scheduleTargetRefresh);
$("upload-file").addEventListener("change", updateFileSelection);
$("folder-files").addEventListener("change", updateFileSelection);
["upload-force-split", "folder-force-split"].forEach((id) => {
  $(id).addEventListener("change", syncDatasetSourceControls);
});
$("dataset-name").addEventListener("input", () => {
  state.datasetNameEdited = true;
});
$("resume").addEventListener("change", syncActionStates);
["rf-project", "rf-version"].forEach((id) => {
  $(id).addEventListener("input", () => {
    updateDatasetNameSuggestion();
    syncTrainingGuide();
  });
});
["split-train", "split-val", "split-test"].forEach((id) => {
  $(id).addEventListener("input", updateSplitTotal);
});
document.querySelectorAll("[data-log-mode]").forEach((button) => {
  button.addEventListener("click", () => setLogMode(button.dataset.logMode));
});
document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});
$("reset-advanced").addEventListener("click", () => {
  applyControlValues(CONTROL_DEFAULTS);
  setActivePreset(null);
  setMessage("Reset training controls to defaults.");
});
const presetControlIds = new Set(Object.values(TRAINING_PRESETS).flatMap((preset) => Object.keys(preset)));
presetControlIds.forEach((id) => {
  const control = $(id);
  if (control) {
    control.addEventListener("input", () => setActivePreset(null));
    control.addEventListener("change", () => setActivePreset(null));
  }
});
window.addEventListener("resize", redrawChartsSoon);

const trainingGuide = $("training-guide");
try {
  const savedGuideState = window.localStorage.getItem("yolov8-training-guide-open");
  if (savedGuideState !== null) {
    trainingGuide.open = savedGuideState === "true";
  }
  trainingGuide.addEventListener("toggle", () => {
    window.localStorage.setItem("yolov8-training-guide-open", String(trainingGuide.open));
  });
} catch (error) {
  // The guide still works when browser storage is unavailable.
}

loadConfig().catch((error) => setMessage(error.message, true));
initializeTooltips();
updateFileSelection();
updateSplitTotal();
syncDatasetSourceControls();
setActivePreset(null);
updateCurrentRunDisplay();
renderEpochProgress();
syncActionStates();
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
