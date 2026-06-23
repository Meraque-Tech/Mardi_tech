const state = {
  source: "upload",
  datasetYaml: "",
  datasetSummary: null,
  preparationPollTimer: null,
  preparationPollRevision: 0,
  pollTimer: null,
  metricsHistory: [],
  metricsAvailable: false,
  gpuSignature: "",
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
  testRunning: false,
  testStarting: false,
  testStopping: false,
  testMetricsAvailable: false,
  testCombinedReportAvailable: false,
  testOutcome: "",
  testLogsMode: "recent",
  testDownloads: new Set(),
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

const PANEL_STORAGE_PREFIX = "yolov8-panel-expanded-";

function setPanelExpanded(panelKey, expanded, { persist = true } = {}) {
  const panel = document.querySelector(`[data-panel-key="${panelKey}"]`);
  const toggle = document.querySelector(`[data-panel-toggle="${panelKey}"]`);
  if (!panel || !toggle) {
    return;
  }

  const sectionName = panel.querySelector("h2")?.textContent?.trim() || "section";
  panel.classList.toggle("is-collapsed", !expanded);
  toggle.setAttribute("aria-expanded", String(expanded));
  toggle.setAttribute("aria-label", `${expanded ? "Collapse" : "Expand"} ${sectionName} section`);

  if (persist) {
    try {
      window.localStorage.setItem(`${PANEL_STORAGE_PREFIX}${panelKey}`, String(expanded));
    } catch (error) {
      // Collapsing still works when browser storage is unavailable.
    }
  }
  if (expanded && panelKey === "results") {
    redrawChartsSoon();
  }
}

function initializeCollapsiblePanels() {
  document.querySelectorAll("[data-panel-key]").forEach((panel) => {
    const panelKey = panel.dataset.panelKey;
    let expanded = panel.dataset.defaultExpanded !== "false";
    try {
      const saved = window.localStorage.getItem(`${PANEL_STORAGE_PREFIX}${panelKey}`);
      if (saved !== null) {
        expanded = saved === "true";
      }
    } catch (error) {
      // Use the markup default when browser storage is unavailable.
    }
    setPanelExpanded(panelKey, expanded, { persist: false });
  });

  document.querySelectorAll("[data-panel-toggle]").forEach((toggle) => {
    toggle.addEventListener("click", () => {
      setPanelExpanded(
        toggle.dataset.panelToggle,
        toggle.getAttribute("aria-expanded") !== "true",
      );
    });
  });
}

function updateDatasetPreparationProgress(stage, percent, detail, indeterminate = false) {
  const progress = $("dataset-preparation-progress");
  const track = $("dataset-preparation-track");
  const safePercent = Math.min(100, Math.max(0, Number(percent) || 0));
  $("dataset-preparation-stage").textContent = stage;
  $("dataset-preparation-percent").textContent = indeterminate ? "In progress" : `${Math.round(safePercent)}%`;
  $("dataset-preparation-detail").textContent = detail;
  track.classList.toggle("indeterminate", indeterminate);
  track.querySelector("span").style.width = indeterminate ? "" : `${safePercent}%`;
  track.setAttribute("aria-valuetext", indeterminate ? `${stage}: in progress` : `${stage}: ${Math.round(safePercent)}%`);
  if (indeterminate) {
    track.removeAttribute("aria-valuenow");
  } else {
    track.setAttribute("aria-valuenow", String(Math.round(safePercent)));
  }
  progress.hidden = false;
}

function setDatasetPreparationProgress(active, source = state.source) {
  const progress = $("dataset-preparation-progress");
  progress.setAttribute("aria-busy", String(active));
  if (!active) {
    progress.hidden = true;
    return;
  }
  if (source === "roboflow") {
    updateDatasetPreparationProgress(
      "Fetching from Roboflow",
      0,
      "Roboflow is exporting, downloading, and inspecting the dataset. This may take several minutes.",
      true,
    );
    return;
  }
  updateDatasetPreparationProgress(
    source === "folder" ? "Uploading folder" : "Uploading ZIP",
    0,
    "Uploading dataset to the server.",
  );
}

function stopDatasetPreparationPolling() {
  state.preparationPollRevision += 1;
  window.clearTimeout(state.preparationPollTimer);
  state.preparationPollTimer = null;
}

function roboflowOverallPercent(stage, stagePercent, forceSplit) {
  const splitRanges = {
    reading_labels: [70, 77],
    calculating_targets: [77, 79],
    assigning: [79, 85],
    finalizing_split: [85, 86],
    copying: [86, 95],
    inspecting: [95, 100],
  };
  const preserveRanges = {
    validating_dataset: [70, 82],
    normalizing_paths: [82, 85],
    inspecting: [85, 100],
  };
  const commonRanges = {
    preparing_roboflow_version: [0, 5],
    preparing_roboflow_export: [5, 10],
    downloading_roboflow: [10, 55],
    extracting_roboflow: [55, 70],
    complete: [100, 100],
  };
  const ranges = { ...commonRanges, ...(forceSplit ? splitRanges : preserveRanges) };
  const range = ranges[stage];
  if (!range) {
    return stagePercent;
  }
  const safeStagePercent = Math.min(100, Math.max(0, Number(stagePercent) || 0));
  return range[0] + ((range[1] - range[0]) * safeStagePercent) / 100;
}

function startDatasetPreparationPolling(jobId, context = {}) {
  stopDatasetPreparationPolling();
  const revision = state.preparationPollRevision;
  const progressSource = context.source || state.source;
  const forceSplit = Boolean(context.forceSplit);
  const stageLabels = {
    fetching_roboflow: "Fetching from Roboflow",
    preparing_roboflow_version: "Preparing Roboflow version",
    preparing_roboflow_export: "Preparing YOLOv8 export",
    downloading_roboflow: "Downloading from Roboflow",
    extracting_roboflow: "Extracting Roboflow ZIP",
    validating_dataset: "Validating dataset",
    normalizing_paths: "Normalizing dataset paths",
    saving: "Saving upload",
    extracting: "Extracting ZIP",
    reading_labels: "Reading labels",
    calculating_targets: "Calculating class targets",
    assigning: "Assigning images",
    finalizing_split: "Finalizing split",
    copying: "Copying split files",
    inspecting: "Inspecting dataset",
    complete: "Complete",
    failed: "Failed",
  };

  const poll = async () => {
    if (revision !== state.preparationPollRevision) {
      return;
    }
    try {
      const response = await fetch(`/api/dataset/preparation/status?job_id=${encodeURIComponent(jobId)}`);
      if (response.status === 404) {
        state.preparationPollTimer = window.setTimeout(poll, 300);
        return;
      }
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Progress request failed: ${response.status}`);
      }
      const percent = progressSource === "roboflow"
        ? roboflowOverallPercent(payload.stage, payload.percent, forceSplit)
        : payload.percent;
      const indeterminate = payload.status === "running"
        && (payload.mode ? payload.mode === "indeterminate" : !payload.total);
      updateDatasetPreparationProgress(
        stageLabels[payload.stage] || payload.stage || "Preparing dataset",
        percent,
        payload.detail || "Preparing dataset.",
        indeterminate,
      );
      if (payload.status === "running") {
        state.preparationPollTimer = window.setTimeout(poll, 300);
      }
    } catch (error) {
      if (revision === state.preparationPollRevision) {
        state.preparationPollTimer = window.setTimeout(poll, 800);
      }
    }
  };
  poll();
}

function uploadWithProgress(url, form, jobId, label) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        updateDatasetPreparationProgress(label, 0, "Uploading dataset to the server.", true);
        return;
      }
      const percent = (event.loaded / event.total) * 100;
      updateDatasetPreparationProgress(
        label,
        percent,
        `Uploaded ${formatBytes(event.loaded)} of ${formatBytes(event.total)}.`,
      );
    });
    request.upload.addEventListener("load", () => {
      updateDatasetPreparationProgress("Upload complete", 100, "The server is starting dataset preparation.");
      startDatasetPreparationPolling(jobId);
    });
    request.addEventListener("load", () => {
      const payload = (() => {
        try {
          return JSON.parse(request.responseText || "{}");
        } catch (error) {
          return {};
        }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(payload.detail || `Upload failed: ${request.status}`));
        return;
      }
      resolve(payload);
    });
    request.addEventListener("error", () => reject(new Error("Dataset upload failed due to a network error.")));
    request.addEventListener("abort", () => reject(new Error("Dataset upload was cancelled.")));
    request.send(form);
  });
}

function preparationJobId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `dataset-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
  const trainingLocked = state.running || state.isStarting || state.isStopping;
  const testLocked = state.testRunning || state.testStarting || state.testStopping;
  const locked = trainingLocked || testLocked;
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
  const datasetDownloadActive = state.downloads.has("dataset");
  $("download-dataset").disabled = !hasDataset || preparing || datasetDownloadActive;
  $("download-dataset").textContent = datasetDownloadActive ? "Preparing ZIP..." : "Download ZIP";
  $("download-dataset").setAttribute("aria-busy", String(datasetDownloadActive));
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
  syncTestActionStates();
  syncTrainingGuide();
}

function usesCustomSplit() {
  if (state.source === "upload") {
    return $("upload-force-split").checked;
  }
  if (state.source === "folder") {
    return $("folder-force-split").checked;
  }
  return state.source === "roboflow" && $("roboflow-force-split").checked;
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

function gpuSeverity(value, warning = 80, critical = 95) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "";
  }
  if (number >= critical) {
    return "critical";
  }
  return number >= warning ? "warning" : "";
}

function renderGpuStatus(status) {
  const signature = JSON.stringify(status || {});
  if (signature === state.gpuSignature) {
    return;
  }
  state.gpuSignature = signature;
  const statusElement = $("gpu-status");
  const cards = $("gpu-cards");
  if (!status || !status.available || !Array.isArray(status.gpus) || !status.gpus.length) {
    statusElement.textContent = "Unavailable";
    statusElement.className = "gpu-status unavailable";
    cards.innerHTML = `<p class="gpu-empty">${escapeHtml(status?.message || "GPU information unavailable.")}</p>`;
    return;
  }

  statusElement.textContent = "● Live";
  statusElement.className = "gpu-status live";
  const optionalNumber = (value) => (
    value === null || value === undefined || value === "" ? Number.NaN : Number(value)
  );
  const meter = (label, value, display) => {
    const numeric = Number(value);
    const available = Number.isFinite(numeric);
    const percent = available ? Math.min(100, Math.max(0, numeric)) : 0;
    const severity = gpuSeverity(numeric);
    const aria = available
      ? `aria-valuenow="${Math.round(percent)}" aria-valuetext="${escapeHtml(display)}"`
      : `aria-valuetext="Not available"`;
    return `
      <div class="gpu-metric">
        <div class="gpu-metric-header"><span>${escapeHtml(label)}</span><strong>${escapeHtml(display)}</strong></div>
        <div class="gpu-meter ${severity}" role="progressbar" aria-valuemin="0" aria-valuemax="100" ${aria}>
          <span style="width: ${percent}%"></span>
        </div>
      </div>`;
  };

  cards.innerHTML = status.gpus.map((gpu) => {
    const utilization = optionalNumber(gpu.utilization_percent);
    const memoryPercent = optionalNumber(gpu.memory_percent);
    const memoryUsed = optionalNumber(gpu.memory_used_mb);
    const memoryTotal = optionalNumber(gpu.memory_total_mb);
    const temperature = optionalNumber(gpu.temperature_c);
    const powerDraw = optionalNumber(gpu.power_draw_w);
    const powerLimit = optionalNumber(gpu.power_limit_w);
    const utilizationText = Number.isFinite(utilization) ? `${utilization.toFixed(1)}%` : "N/A";
    const memoryText = Number.isFinite(memoryUsed) && Number.isFinite(memoryTotal)
      ? (memoryUsed < 1024
        ? `${memoryUsed.toFixed(0)} MB / ${(memoryTotal / 1024).toFixed(1)} GB`
        : `${(memoryUsed / 1024).toFixed(1)} / ${(memoryTotal / 1024).toFixed(1)} GB`)
      : "N/A";
    const temperatureText = Number.isFinite(temperature) ? `${temperature.toFixed(0)}°C` : "N/A";
    const powerText = Number.isFinite(powerDraw)
      ? `${powerDraw.toFixed(1)}${Number.isFinite(powerLimit) ? ` / ${powerLimit.toFixed(1)}` : ""} W`
      : "N/A";
    const temperatureClass = gpuSeverity(temperature, 80, 90);
    return `
      <article class="gpu-card">
        <div class="gpu-card-header">
          <strong>GPU ${escapeHtml(gpu.index)}</strong>
          <span>${escapeHtml(gpu.name || "NVIDIA GPU")}</span>
        </div>
        ${meter("GPU Load", utilization, utilizationText)}
        ${meter("VRAM", memoryPercent, memoryText)}
        <div class="gpu-detail-grid">
          <div><span>Temperature</span><strong class="${temperatureClass}">${escapeHtml(temperatureText)}</strong></div>
          <div><span>Power</span><strong>${escapeHtml(powerText)}</strong></div>
        </div>
      </article>`;
  }).join("");
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
  } else {
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
  }
  syncActionStates();

  const testWeight = $("test-weight-file").files[0];
  $("test-weight-selection").textContent = testWeight
    ? `${testWeight.name} (${formatBytes(testWeight.size)})`
    : "No weights file selected.";

  const testZip = $("test-dataset-zip").files[0];
  $("test-dataset-zip-selection").textContent = testZip
    ? `${testZip.name} (${formatBytes(testZip.size)})`
    : "No ZIP selected.";

  const testFolderFiles = Array.from($("test-dataset-folder").files);
  if (!testFolderFiles.length) {
    $("test-dataset-folder-selection").textContent = "No folder selected.";
  } else {
    const firstPath = testFolderFiles[0].webkitRelativePath || testFolderFiles[0].name;
    const folderLabel = firstPath.split("/")[0];
    const totalFolderBytes = testFolderFiles.reduce((sum, file) => sum + file.size, 0);
    $("test-dataset-folder-selection").textContent = `${folderLabel}: ${testFolderFiles.length} files (${formatBytes(totalFolderBytes)}).`;
  }
  syncTestSourceControls();
}

function selectedTestWeightSource() {
  return document.querySelector('input[name="test-weight-source"]:checked')?.value || "trained";
}

function selectedTestDatasetSource() {
  return document.querySelector('input[name="test-dataset-source"]:checked')?.value || "prepared";
}

function syncTestSourceControls() {
  const weightSource = selectedTestWeightSource();
  const datasetSource = selectedTestDatasetSource();

  $("test-weight-upload-wrap").hidden = weightSource !== "upload";
  $("test-trained-weights-note").hidden = weightSource !== "trained";

  $("test-dataset-zip-wrap").hidden = datasetSource !== "upload_zip";
  $("test-dataset-folder-wrap").hidden = datasetSource !== "upload_folder";
  const datasetNotes = {
    prepared: "Uses the prepared dataset's existing test split.",
    upload_zip: "Upload a labeled YOLO ZIP. A data.yaml is preferred; otherwise the prepared dataset classes are reused.",
    upload_folder: "Upload a labeled YOLO folder. A data.yaml is preferred; otherwise the prepared dataset classes are reused.",
  };
  $("test-dataset-note").textContent = datasetNotes[datasetSource] || datasetNotes.prepared;
  syncTestActionStates();
}

function syncTestActionStates() {
  const trainingBusy = state.running || state.isStarting || state.isStopping;
  const testingBusy = state.testRunning || state.testStarting || state.testStopping;
  const sourcePrepared = selectedTestDatasetSource() === "prepared";
  const sourceUploadZip = selectedTestDatasetSource() === "upload_zip";
  const sourceUploadFolder = selectedTestDatasetSource() === "upload_folder";
  const weightUpload = selectedTestWeightSource() === "upload";
  const hasPreparedDataset = Boolean(state.datasetYaml);
  const hasZip = Boolean($("test-dataset-zip").files[0]);
  const hasFolder = Boolean($("test-dataset-folder").files.length);
  const hasWeightFile = Boolean($("test-weight-file").files[0]);
  const canStart = !trainingBusy
    && !testingBusy
    && (
      (sourcePrepared && hasPreparedDataset)
      || (sourceUploadZip && hasZip)
      || (sourceUploadFolder && hasFolder)
    )
    && (!weightUpload || hasWeightFile);

  $("start-test").disabled = !canStart;
  $("start-test").textContent = state.testStarting ? "Starting..." : "Start Test";
  $("stop-test").disabled = !state.testRunning || state.testStopping;
  $("stop-test").textContent = state.testStopping ? "Stopping..." : "Stop Test";
  $("refresh-test-results").disabled = state.testStarting;

  document.querySelectorAll("#model-testing-panel input, #model-testing-panel select, #model-testing-panel textarea").forEach((control) => {
    if (control.id === "stop-test") {
      return;
    }
    if (control.id === "start-test") {
      return;
    }
    if (control.id === "refresh-test-results") {
      return;
    }
    if (control.type === "button") {
      return;
    }
    control.disabled = trainingBusy || testingBusy;
  });
}

function renderTestProgress(progress = {}, phase = "idle") {
  const percent = Math.min(100, Math.max(0, Number(progress.percent) || 0));
  const stage = progress.stage || phase || "idle";
  const detail = progress.detail || "The current test stage will appear here when testing starts.";
  const stageLabels = {
    idle: "Waiting to start",
    starting: "Starting model testing",
    initializing: "Initializing",
    evaluating: "Evaluating test split",
    saving_metrics: "Saving metrics",
    complete: "Model testing complete",
    failed: "Model testing failed",
  };
  const label = stageLabels[stage] || stage.replace(/_/g, " ");
  $("test-progress-label").textContent = label;
  $("test-progress-percent").textContent = `${Math.round(percent)}%`;
  $("test-progress-fill").style.width = `${percent}%`;
  $("test-progress-detail").textContent = detail;
  const track = $("test-progress-track");
  track.setAttribute("aria-valuenow", String(Math.round(percent)));
  track.setAttribute("aria-valuetext", `${label}: ${Math.round(percent)}%`);
}

function testArtifactViewUrl(artifact, status) {
  const params = new URLSearchParams();
  params.set("v", String(status.modified_at || status.size || 0));
  return `/api/test/artifacts/view/${encodeURIComponent(artifact)}?${params.toString()}`;
}

function renderTestConfusionMatrices(artifacts = {}) {
  const variants = [
    {
      artifact: "confusion_matrix_normalized",
      card: "test-confusion-matrix-normalized-card",
      image: "test-confusion-matrix-normalized-image",
      link: "test-confusion-matrix-normalized-link",
    },
    {
      artifact: "confusion_matrix",
      card: "test-confusion-matrix-card",
      image: "test-confusion-matrix-image",
      link: "test-confusion-matrix-link",
    },
  ];
  let availableCount = 0;
  variants.forEach((variant) => {
    const status = artifacts?.[variant.artifact] || {};
    const available = Boolean(status.available);
    const card = $(variant.card);
    const image = $(variant.image);
    const link = $(variant.link);
    card.hidden = !available;
    if (!available) {
      image.removeAttribute("src");
      link.removeAttribute("href");
      return;
    }
    availableCount += 1;
    const url = testArtifactViewUrl(variant.artifact, status);
    if (image.getAttribute("src") !== url) {
      image.src = url;
    }
    link.href = url;
  });
  $("test-confusion-matrix-status").textContent = availableCount
    ? "Click a matrix to open the full-resolution test plot."
    : "The confusion matrix will appear after the test evaluation finishes.";
}

function renderTestRocAuc(rocAuc = {}, artifacts = {}) {
  const status = artifacts?.roc_auc_curve || {};
  const available = Boolean(status.available);
  const card = $("test-roc-auc-card");
  const image = $("test-roc-auc-image");
  const link = $("test-roc-auc-link");
  card.hidden = !available;
  if (!available) {
    image.removeAttribute("src");
    link.removeAttribute("href");
  } else {
    const url = testArtifactViewUrl("roc_auc_curve", status);
    if (image.getAttribute("src") !== url) {
      image.src = url;
    }
    link.href = url;
  }

  const classes = Array.isArray(rocAuc.classes) ? rocAuc.classes : [];
  const container = $("test-roc-auc-summary");
  if (!classes.length) {
    container.innerHTML = "";
  } else {
    const rows = classes.map((item) => `
      <tr>
        <td>${escapeHtml(item.class_name)}</td>
        <td>${item.positive_images || 0}</td>
        <td>${item.negative_images || 0}</td>
        <td>${metricText(item.auc)}</td>
      </tr>
    `).join("");
    container.innerHTML = `
      <h4>Per-Class AUC</h4>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Positive images</th>
            <th>Negative images</th>
            <th>AUC</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  $("test-roc-auc-status").textContent = available
    ? `Click the ROC plot to open the full-resolution image. ${rocAuc.note || ""}`.trim()
    : (rocAuc.note || "ROC-AUC will appear after the test evaluation finishes.");
}

function renderTestClassMetrics(classes) {
  const container = $("test-class-metrics");
  if (!Array.isArray(classes) || !classes.length) {
    container.innerHTML = "";
    return;
  }

  const rows = classes.map((item) => `
    <tr>
      <td>${escapeHtml(item.class_name)}</td>
      <td>${item.instances}</td>
      <td>${metricText(item.map50)}</td>
      <td>${metricText(item.map50_95)}</td>
      <td>${metricText(item.f1)}</td>
      <td>${metricText(item.precision)}</td>
      <td>${metricText(item.recall)}</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <h4>Per-Class Test Metrics</h4>
    <table>
      <thead>
        <tr>
          <th>Class</th>
          <th>Instances</th>
          <th>AP50</th>
          <th>AP50-95</th>
          <th>F1</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function setTestArtifactButtons(artifacts) {
  const metricsAvailable = Boolean(artifacts?.metrics_json?.available);
  const rocAvailable = Boolean(artifacts?.roc_auc_curve?.available);
  $("download-test-metrics-json").disabled = !metricsAvailable || state.testDownloads.has("metrics_json");
  $("download-test-roc-auc-graph").disabled = !rocAvailable || state.testDownloads.has("roc_auc_curve");
  $("download-combined-report").disabled = !state.testMetricsAvailable
    || !state.testCombinedReportAvailable
    || state.testRunning
    || state.testDownloads.has("combined_report");
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
      <td>${metricText(item.map50)}</td>
      <td>${metricText(item.map50_95)}</td>
      <td>${metricText(item.f1)}</td>
      <td>${metricText(item.precision)}</td>
      <td>${metricText(item.recall)}</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <h4>Per-Class Metrics</h4>
    <table>
      <thead>
        <tr>
          <th>Class</th>
          <th>Instances</th>
          <th>AP50</th>
          <th>AP50-95</th>
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
  const classBalance = Array.isArray(summary.class_balance) ? summary.class_balance : [];
  const balanceRows = classBalance.map((item) => {
    const splitCells = ["train", "val", "test"].map((split) => {
      const row = item.splits?.[split] || {};
      const imageShare = Number(row.image_share) || 0;
      const targetShare = Number(row.target_share) || 0;
      return `
        <td>
          <strong>${row.images || 0} img / ${row.instances || 0} obj</strong>
          <small>${imageShare.toFixed(1)}% vs ${targetShare.toFixed(1)}% target</small>
        </td>
      `;
    }).join("");
    return `
      <tr>
        <th>${escapeHtml(item.class_name)}</th>
        <td>${item.images || 0} img / ${item.instances || 0} obj</td>
        ${splitCells}
        <td>${Number(item.max_image_deviation || 0).toFixed(1)} pp</td>
      </tr>
    `;
  }).join("");
  const balanceTable = balanceRows
    ? `
      <div class="split-balance">
        <h4>Split Class Balance</h4>
        <p>Image presence drives stratification; object instances are used as a tie-breaker.</p>
        <table>
          <thead>
            <tr><th>Class</th><th>Total</th><th>Train</th><th>Val</th><th>Test</th><th>Max Δ</th></tr>
          </thead>
          <tbody>${balanceRows}</tbody>
        </table>
      </div>
    `
    : "";
  const warnings = Array.isArray(summary.warnings) && summary.warnings.length
    ? `<ul>${summary.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
    : "<p>No dataset warnings found.</p>";

  container.innerHTML = `
    <details class="dataset-summary-details" open>
      <summary>
        <span>Dataset Summary</span>
        <small>${summary.total_images || 0} images, ${summary.class_count || 0} classes${summary.split_strategy === "multi_label_stratified" ? " · stratified" : ""}</small>
      </summary>
      <div class="dataset-summary-content">
        <div class="summary-grid">${splitRows}</div>
        ${distributionTable}
        ${balanceTable}
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
  $("download-roc-auc-graph").disabled = !isEnabled("roc_auc_curve") || state.downloads.has("roc_auc_curve");
  $("download-training-report").disabled = !state.metricsAvailable
    || state.running
    || state.downloads.has("training_report");
}

function artifactViewUrl(artifact, target, status) {
  const params = new URLSearchParams({
    project: target.project,
    name: target.name,
  });
  params.set("v", String(status.modified_at || status.size || 0));
  return `/api/train/artifacts/view/${encodeURIComponent(artifact)}?${params.toString()}`;
}

function renderConfusionMatrices(artifacts = {}, target = weightTarget()) {
  const variants = [
    {
      artifact: "confusion_matrix_normalized",
      card: "confusion-matrix-normalized-card",
      image: "confusion-matrix-normalized-image",
      link: "confusion-matrix-normalized-link",
    },
    {
      artifact: "confusion_matrix",
      card: "confusion-matrix-card",
      image: "confusion-matrix-image",
      link: "confusion-matrix-link",
    },
  ];
  let availableCount = 0;
  variants.forEach((variant) => {
    const status = artifacts?.[variant.artifact] || {};
    const available = Boolean(status.available);
    const card = $(variant.card);
    const image = $(variant.image);
    const link = $(variant.link);
    card.hidden = !available;
    if (!available) {
      image.removeAttribute("src");
      link.removeAttribute("href");
      return;
    }

    availableCount += 1;
    const url = artifactViewUrl(variant.artifact, target, status);
    if (image.getAttribute("src") !== url) {
      image.src = url;
    }
    link.href = url;
  });

  $("confusion-matrix-status").textContent = availableCount
    ? "Click a matrix to open the full-resolution validation plot."
    : "The confusion matrix will appear after validation plots are generated.";
}

function renderRocAuc(rocAuc = {}, artifacts = {}, target = weightTarget()) {
  const status = artifacts?.roc_auc_curve || {};
  const available = Boolean(status.available);
  const card = $("roc-auc-card");
  const image = $("roc-auc-image");
  const link = $("roc-auc-link");
  card.hidden = !available;
  if (!available) {
    image.removeAttribute("src");
    link.removeAttribute("href");
  } else {
    const url = artifactViewUrl("roc_auc_curve", target, status);
    if (image.getAttribute("src") !== url) {
      image.src = url;
    }
    link.href = url;
  }

  const classes = Array.isArray(rocAuc.classes) ? rocAuc.classes : [];
  const container = $("roc-auc-summary");
  if (!classes.length) {
    container.innerHTML = "";
  } else {
    const rows = classes.map((item) => `
      <tr>
        <td>${escapeHtml(item.class_name)}</td>
        <td>${item.positive_images || 0}</td>
        <td>${item.negative_images || 0}</td>
        <td>${metricText(item.auc)}</td>
      </tr>
    `).join("");
    container.innerHTML = `
      <h4>Per-Class AUC</h4>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Positive images</th>
            <th>Negative images</th>
            <th>AUC</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  const note = rocAuc.note || "ROC-AUC will appear after post-training evaluation finishes.";
  $("roc-auc-status").textContent = available
    ? `Click the ROC plot to open the full-resolution image. ${note}`
    : note;
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
  $("test-device").value = config.default_device || "";
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
  const jobId = preparationJobId();
  form.append("job_id", jobId);
  return uploadWithProgress("/api/dataset/upload", form, jobId, "Uploading ZIP");
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
  const jobId = preparationJobId();
  form.append("job_id", jobId);
  return uploadWithProgress("/api/dataset/folder", form, jobId, "Uploading folder");
}

async function prepareRoboflowDataset() {
  const jobId = preparationJobId();
  startDatasetPreparationPolling(jobId, {
    source: "roboflow",
    forceSplit: $("roboflow-force-split").checked,
  });
  return apiJson("/api/dataset/roboflow", {
    method: "POST",
    body: JSON.stringify({
      workspace: $("rf-workspace").value,
      project: $("rf-project").value,
      version: $("rf-version").value,
      classes: classNames(),
      name: $("dataset-name").value,
      train: numberValue("split-train"),
      val: numberValue("split-val"),
      test: numberValue("split-test"),
      force_split: $("roboflow-force-split").checked,
      job_id: jobId,
    }),
  });
}

async function prepareDataset() {
  if (state.isPreparing) {
    return;
  }
  const preparingRoboflow = state.source === "roboflow";
  const preparationSource = state.source;
  let preparationSucceeded = false;
  state.isPreparing = true;
  setDatasetPreparationProgress(true, preparationSource);
  syncActionStates();
  setMessage(preparingRoboflow
    ? "In progress: fetching and preparing the Roboflow dataset..."
    : "Preparing dataset...");
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
    preparationSucceeded = true;
    updateDatasetPreparationProgress("Complete", 100, "Dataset preparation complete.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    stopDatasetPreparationPolling();
    if (preparationSucceeded) {
      await new Promise((resolve) => window.setTimeout(resolve, 450));
    }
    state.isPreparing = false;
    setDatasetPreparationProgress(false);
    if (!state.running) {
      setStatusPhase("idle");
    }
    syncActionStates();
  }
}

async function downloadPreparedDataset() {
  if (!state.datasetYaml || state.downloads.has("dataset")) {
    return;
  }

  state.downloads.add("dataset");
  syncActionStates();
  setMessage("Preparing the dataset ZIP. Large datasets may take several minutes...");
  try {
    const result = await apiJson("/api/dataset/download/prepare", {
      method: "POST",
      body: JSON.stringify({ dataset_yaml: state.datasetYaml }),
    });
    const link = document.createElement("a");
    link.href = result.download_url;
    link.download = result.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setMessage(`Downloading ${result.filename} (${formatBytes(result.size)}).`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.downloads.delete("dataset");
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
    setPanelExpanded("logs", true);
    setPanelExpanded("results", true);
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
      $("metric-macro-f1").textContent = "-";
      $("metric-weighted-f1").textContent = "-";
      $("metric-train-loss").textContent = "-";
      $("metric-test-loss").textContent = "-";
      $("metric-map50").textContent = "-";
      $("metric-map").textContent = "-";
      renderBestMetrics(null);
      renderClassMetrics([]);
      resetCharts();
      setArtifactButtons(metrics.artifacts || false);
      renderConfusionMatrices(metrics.artifacts || {}, target);
      renderRocAuc({}, metrics.artifacts || {}, target);
      $("metrics-status").textContent = "No results.csv found for this run yet.";
      return;
    }

    state.metricsAvailable = true;
    $("training-results-panel").classList.add("has-results");
    $("metric-macro-f1").textContent = metricText(metrics.macro_f1);
    $("metric-weighted-f1").textContent = metricText(metrics.weighted_f1);
    $("metric-train-loss").textContent = metricText(metrics.training_loss);
    $("metric-test-loss").textContent = metricText(metrics.testing_loss);
    $("metric-map50").textContent = metricText(metrics.map50);
    $("metric-map").textContent = metricText(metrics.map50_95);
    renderBestMetrics(metrics.best, metrics.history);
    renderClassMetrics(metrics.per_class);
    renderMetricCharts(metrics.history);
    setArtifactButtons(metrics.artifacts || true);
    renderConfusionMatrices(metrics.artifacts || {}, target);
    renderRocAuc(metrics.roc_auc || {}, metrics.artifacts || {}, target);
    $("metrics-status").textContent = `Epoch ${metrics.epoch}. ${metrics.note}`;
  } catch (error) {
    if (revision !== state.targetRevision) {
      return;
    }
    $("training-results-panel").classList.remove("has-results");
    resetCharts();
    setArtifactButtons(false);
    renderConfusionMatrices({}, target);
    renderRocAuc({}, {}, target);
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

async function downloadTrainingReport() {
  const button = $("download-training-report");
  const originalText = button.textContent;
  state.downloads.add("training_report");
  button.disabled = true;
  button.textContent = "Generating report...";
  try {
    const response = await fetch("/api/train/report/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(weightTarget()),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Report generation failed: ${response.status}`);
    }
    saveBlobWithBrowserDownload(await response.blob(), "training_report.pdf");
    setMessage("Downloading training_report.pdf.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.downloads.delete("training_report");
    button.textContent = originalText;
    await refreshMetrics();
  }
}

async function startTest() {
  if (state.testStarting || state.testRunning) {
    return;
  }

  const weightSource = selectedTestWeightSource();
  const datasetSource = selectedTestDatasetSource();
  if (datasetSource === "prepared" && !state.datasetYaml) {
    setMessage("Prepare a dataset first so the test split is available.", true);
    return;
  }
  if (weightSource === "upload" && !$("test-weight-file").files[0]) {
    setMessage("Choose a .pt weights file to test.", true);
    return;
  }
  if (datasetSource === "upload_zip" && !$("test-dataset-zip").files[0]) {
    setMessage("Choose a labeled YOLO ZIP file for testing.", true);
    return;
  }
  if (datasetSource === "upload_folder" && !$("test-dataset-folder").files.length) {
    setMessage("Choose a labeled YOLO folder for testing.", true);
    return;
  }

  state.testStarting = true;
  state.testOutcome = "";
  syncActionStates();
  renderTestProgress({ percent: 0, stage: "starting", detail: "Uploading inputs and launching the test job." }, "starting");
  setMessage("Starting model testing...");

  try {
    const form = new FormData();
    form.append("weight_source", weightSource);
    form.append("project", $("project").value || "runs/detect");
    form.append("name", $("run-name").value || "train");
    form.append("prepared_dataset_yaml", state.datasetYaml || "");
    form.append("reference_dataset_yaml", state.datasetYaml || "");
    form.append("dataset_source", datasetSource);
    form.append("imgsz", $("test-imgsz").value);
    form.append("batch", $("test-batch").value);
    form.append("workers", $("test-workers").value);
    form.append("device", $("test-device").value || $("device").value || "");

    const weightFile = $("test-weight-file").files[0];
    if (weightSource === "upload" && weightFile) {
      form.append("weight_file", weightFile);
    }

    const datasetZip = $("test-dataset-zip").files[0];
    if (datasetSource === "upload_zip" && datasetZip) {
      form.append("dataset_zip", datasetZip);
    }

    if (datasetSource === "upload_folder") {
      Array.from($("test-dataset-folder").files).forEach((file) => {
        form.append("dataset_files", file, file.webkitRelativePath || file.name);
      });
    }

    const response = await fetch("/api/test/start", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed: ${response.status}`);
    }

    state.testRunning = true;
    state.testMetricsAvailable = false;
    state.testCombinedReportAvailable = false;
    $("model-testing-panel").classList.remove("has-results");
    setPanelExpanded("testing", true);
    setMessage(`${payload.message}\nPID: ${payload.pid}`);
    $("test-logs").textContent = "";
    await Promise.all([refreshTestResults(), pollStatus()]);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.testStarting = false;
    syncActionStates();
  }
}

async function stopTest() {
  if (!state.testRunning || state.testStopping) {
    return;
  }
  state.testStopping = true;
  syncActionStates();
  try {
    const payload = await apiJson("/api/test/stop", { method: "POST", body: "{}" });
    setMessage(payload.message);
    await pollStatus();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.testStopping = false;
    syncActionStates();
  }
}

async function refreshTestResults() {
  const button = $("refresh-test-results");
  button.disabled = true;
  button.textContent = "Refreshing...";
  try {
    const results = await apiJson("/api/test/results");
    if (!results.available) {
      state.testMetricsAvailable = false;
      state.testCombinedReportAvailable = false;
      $("model-testing-panel").classList.remove("has-results");
      $("test-metric-macro-f1").textContent = "-";
      $("test-metric-weighted-f1").textContent = "-";
      $("test-metric-precision").textContent = "-";
      $("test-metric-recall").textContent = "-";
      $("test-metric-map50").textContent = "-";
      $("test-metric-map").textContent = "-";
      renderTestClassMetrics([]);
      renderTestConfusionMatrices({});
      renderTestRocAuc({}, {});
      setTestArtifactButtons({});
      $("test-results-status").textContent = "No test results available yet.";
      return;
    }

    state.testMetricsAvailable = true;
    state.testCombinedReportAvailable = Boolean(results.combined_report_available);
    $("model-testing-panel").classList.add("has-results");
    $("test-metric-macro-f1").textContent = metricText(results.macro_f1);
    $("test-metric-weighted-f1").textContent = metricText(results.weighted_f1);
    $("test-metric-precision").textContent = metricText(results.precision);
    $("test-metric-recall").textContent = metricText(results.recall);
    $("test-metric-map50").textContent = metricText(results.map50);
    $("test-metric-map").textContent = metricText(results.map50_95);
    renderTestClassMetrics(results.per_class);
    renderTestConfusionMatrices(results.artifacts || {});
    renderTestRocAuc(results.roc_auc || {}, results.artifacts || {});
    setTestArtifactButtons(results.artifacts || {});
    $("test-results-status").textContent = `Evaluated ${results.split || "test"} split from ${results.run_dir}.`;
  } catch (error) {
    state.testCombinedReportAvailable = false;
    $("model-testing-panel").classList.remove("has-results");
    setTestArtifactButtons({});
    $("test-results-status").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Refresh Results";
  }
}

async function downloadTestArtifact(artifact, filename) {
  const buttonIds = {
    metrics_json: "download-test-metrics-json",
    roc_auc_curve: "download-test-roc-auc-graph",
  };
  const button = $(buttonIds[artifact]);
  const originalText = button.textContent;
  state.testDownloads.add(artifact);
  button.disabled = true;
  button.textContent = "Downloading...";
  try {
    const response = await fetch("/api/test/artifacts/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artifact }),
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
    state.testDownloads.delete(artifact);
    button.textContent = originalText;
    await refreshTestResults();
  }
}

async function downloadCombinedReport() {
  const button = $("download-combined-report");
  const originalText = button.textContent;
  state.testDownloads.add("combined_report");
  button.disabled = true;
  button.textContent = "Generating report...";
  try {
    const response = await fetch("/api/test/report/download", { method: "POST", body: "{}" });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Report generation failed: ${response.status}`);
    }
    saveBlobWithBrowserDownload(await response.blob(), "training_and_test_report.pdf");
    setMessage("Downloading training_and_test_report.pdf.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.testDownloads.delete("combined_report");
    button.textContent = originalText;
    setTestArtifactButtons({});
    await refreshTestResults();
  }
}

function downloadTestLog() {
  window.location.href = "/api/test/logs/download";
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
    const [status, testStatus] = await Promise.all([
      apiJson("/api/train/status"),
      apiJson("/api/test/status"),
    ]);
    const wasRunning = state.running;
    state.running = Boolean(status.running);
    const wasTestRunning = state.testRunning;
    state.testRunning = Boolean(testStatus.running);

    if (state.running) {
      state.trainingStarted = true;
      state.trainingOutcome = "";
      if (!wasRunning) {
        setPanelExpanded("logs", true);
        setPanelExpanded("results", true);
      }
      setStatusPhase("training");
    } else if (wasRunning && state.stopRequested) {
      state.stopRequested = false;
      state.trainingOutcome = "stopped";
      setStatusPhase("stopped");
      setMessage("Training stopped by user. The latest available checkpoint remains in the run folder.");
    } else if (wasRunning && status.returncode === 0) {
      state.trainingCompleted = true;
      state.trainingOutcome = "completed";
      setPanelExpanded("results", true);
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
    renderGpuStatus(status.gpu);

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
    const shouldRefreshData = now - state.lastDataRefresh >= 6000
      || wasRunning !== state.running
      || wasTestRunning !== state.testRunning;
    if (shouldRefreshData) {
      state.lastDataRefresh = now;
      const revision = state.targetRevision;
      const target = weightTarget();
      await Promise.all([
        refreshWeightsStatus(target, revision),
        refreshMetrics(target, revision),
      ]);
    }

    if (state.testRunning) {
      state.testOutcome = "";
      setPanelExpanded("testing", true);
    } else if (wasTestRunning && testStatus.returncode === 0) {
      state.testOutcome = "completed";
      setPanelExpanded("testing", true);
      setMessage("Model testing completed. Test metrics and plots are ready to review.");
    } else if (wasTestRunning && testStatus.returncode !== null && testStatus.returncode !== 0) {
      state.testOutcome = "failed";
      setPanelExpanded("testing", true);
      setMessage(`Model testing stopped with exit code ${testStatus.returncode}. Review the test log.`, true);
    }

    renderTestProgress(
      testStatus.progress,
      state.testRunning ? "running" : (state.testOutcome || "idle"),
    );
    $("test-logs").textContent = testStatus.log_tail || "No model-testing log output yet.";
    $("test-log-status").textContent = testStatus.history_log_file
      ? `Current log: ${testStatus.log_file} | Run log: ${testStatus.history_log_file}`
      : `Current log: ${testStatus.log_file}`;
    if (shouldRefreshData) {
      await refreshTestResults();
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
$("download-dataset").addEventListener("click", downloadPreparedDataset);
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
$("download-roc-auc-graph").addEventListener("click", () => downloadArtifact("roc_auc_curve", "roc_auc_curve.png"));
$("download-training-report").addEventListener("click", downloadTrainingReport);
$("download-run-log").addEventListener("click", () => downloadLog("/api/train/logs/download"));
$("start-test").addEventListener("click", startTest);
$("stop-test").addEventListener("click", stopTest);
$("refresh-test-results").addEventListener("click", refreshTestResults);
$("download-test-log").addEventListener("click", downloadTestLog);
$("download-test-metrics-json").addEventListener("click", () => downloadTestArtifact("metrics_json", "test_metrics.json"));
$("download-test-roc-auc-graph").addEventListener("click", () => downloadTestArtifact("roc_auc_curve", "test_roc_auc_curve.png"));
$("download-combined-report").addEventListener("click", downloadCombinedReport);
$("project").addEventListener("input", scheduleTargetRefresh);
$("run-name").addEventListener("input", scheduleTargetRefresh);
$("upload-file").addEventListener("change", updateFileSelection);
$("folder-files").addEventListener("change", updateFileSelection);
$("test-weight-file").addEventListener("change", updateFileSelection);
$("test-dataset-zip").addEventListener("change", updateFileSelection);
$("test-dataset-folder").addEventListener("change", updateFileSelection);
["upload-force-split", "folder-force-split", "roboflow-force-split"].forEach((id) => {
  $(id).addEventListener("change", syncDatasetSourceControls);
});
document.querySelectorAll('input[name="test-weight-source"]').forEach((input) => {
  input.addEventListener("change", syncTestSourceControls);
});
document.querySelectorAll('input[name="test-dataset-source"]').forEach((input) => {
  input.addEventListener("change", syncTestSourceControls);
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
initializeCollapsiblePanels();
updateFileSelection();
updateSplitTotal();
syncDatasetSourceControls();
setActivePreset(null);
updateCurrentRunDisplay();
renderEpochProgress();
syncActionStates();
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
