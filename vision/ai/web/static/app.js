const state = {
  source: "upload",
  datasetYaml: "",
  datasetSummary: null,
  preparationPollTimer: null,
  preparationPollRevision: 0,
  pollTimer: null,
  metricsHistory: [],
  latestMetrics: null,
  magicChoice: null,
  magicAdjustments: [],
  metricLabels: {},
  performanceChartTitle: "Detection Performance by Epoch",
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
  storageCleanup: new Set(),
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
  inferenceWeights: [],
  inferenceRunning: false,
  inferenceStopping: false,
  inferenceClearing: false,
  inferenceStoppable: false,
  inferenceInputObjectUrl: "",
  inferencePollTimer: null,
  inferencePollRevision: 0,
  inferenceJobId: "",
  inferencePreviewStreamJobId: "",
  inferenceWebRtcJobId: "",
  inferenceWebRtcStartingJobId: "",
  inferenceWebRtcFailedJobId: "",
  inferenceWebRtcPeer: null,
  trainingSessions: [],
  annotationQaRunning: false,
  annotationQaStopping: false,
  annotationQaApplyingFixes: false,
  annotationQaJobId: "",
  annotationQaPollTimer: null,
  annotationQaPollRevision: 0,
  annotationQaReport: null,
  annotationQaActiveIssueId: "",
  annotationQaReviewSeverity: "",
  annotationQaCorrectedDatasetYaml: "",
  annotationQaQueue: "needs_review",
  annotationQaPage: 1,
  annotationQaPageSize: 50,
  annotationQaFilters: { search: "", severity: "", split: "", className: "", issueType: "" },
  annotationQaUndo: null,
  annotationQaUndoTimer: null,
  annotationQaReviewReturnFocus: null,
  annotationQaZoom: 1,
  annotationQaSelected: new Set(),
  annotationQaCanvasRevision: 0,
};

const $ = (id) => document.getElementById(id);
const ANNOTATION_QA_REPORT_VERSION = 4;

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

const AUGMENTATION_DEFAULTS = {
  "disable-ultralytics-albumentations": true,
  mosaic: 0,
  "close-mosaic": 0,
  "hsv-h": 0,
  "hsv-s": 0,
  "hsv-v": 0,
  degrees: 0.0,
  translate: 0.05,
  scale: 0.25,
  shear: 0.0,
  perspective: 0.0,
  flipud: 0.0,
  fliplr: 0,
  bgr: 0.0,
  mixup: 0.0,
  cutmix: 0.0,
  "copy-paste": 0.0,
  "auto-augment": "",
  erasing: 0.0,
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
  optimizer: "auto",
  seed: 42,
  lr0: 0.001,
  lrf: 0.01,
  "weight-decay": 0.0005,
  "warmup-epochs": 3.0,
  freeze: "",
  "cos-lr": false,
  "exist-ok": false,
  resume: false,
  "augmentation-enabled": false,
  ...AUGMENTATION_DEFAULTS,
};

const RFDETR_MODEL_SIZE = "rfdetr-nano";
const DFINE_MODEL_SIZE = "dfine-n";
const RFDETR_DEFAULTS = {
  imgsz: 512,
  batch: 4,
  lr0: 0.0001,
  "weight-decay": 0.0001,
  "warmup-epochs": 0,
  "cos-lr": false,
};
const DFINE_DEFAULTS = {
  imgsz: 640,
  batch: 4,
  lr0: 0.0004,
  "weight-decay": 0.0001,
  "warmup-epochs": 500,
  "cos-lr": false,
};

const MODEL_TASKS = [
  { id: "detect", label: "Detection", badge: "detection", summary: "Bounding-box object detection." },
  { id: "segment", label: "Instance Segmentation", badge: "instance-segmentation", summary: "Object masks and boxes." },
  { id: "semantic", label: "Semantic Segmentation", badge: "semantic-segmentation", summary: "Pixel-level class maps." },
  { id: "classify", label: "Classification", badge: "classification", summary: "One label per image." },
];

const MODEL_FAMILIES = [
  { id: "yolov8", label: "YOLOv8", backend: "ultralytics", tasks: ["detect", "segment", "classify"], summary: "Stable Ultralytics baseline." },
  { id: "yolo11", label: "YOLO11", backend: "ultralytics", tasks: ["detect", "segment", "classify"], summary: "Newer Ultralytics YOLO family." },
  { id: "yolo26", label: "YOLO26", backend: "ultralytics", tasks: ["detect", "segment", "semantic", "classify"], summary: "Ultralytics family with semantic segmentation options." },
  { id: "rfdetr", label: "RF-DETR", backend: "rfdetr", tasks: ["detect"], summary: "Transformer detector; Nano is enabled for now." },
  { id: "dfine", label: "D-FINE", backend: "dfine", tasks: ["detect"], summary: "COCO-format transformer detector; Nano is enabled for now." },
];

const MODEL_SIZES = [
  { id: "nano", label: "Nano", code: "n", summary: "Lowest VRAM and fastest training." },
  { id: "small", label: "Small", code: "s", summary: "Balanced speed and accuracy." },
  { id: "medium", label: "Medium", code: "m", summary: "More capacity, higher VRAM." },
  { id: "large", label: "Large", code: "l", summary: "High capacity, slower training." },
  { id: "xlarge", label: "Extra large", code: "x", summary: "Largest YOLO option." },
];

const MODEL_TASK_BY_ID = Object.fromEntries(MODEL_TASKS.map((task) => [task.id, task]));
const MODEL_FAMILY_BY_ID = Object.fromEntries(MODEL_FAMILIES.map((family) => [family.id, family]));
const MODEL_SIZE_BY_ID = Object.fromEntries(MODEL_SIZES.map((size) => [size.id, size]));

function yoloModelValue(familyId, taskId, sizeId) {
  const familyPrefix = familyId === "yolov8" ? "" : `${familyId}-`;
  const taskSuffix = taskId === "segment" ? "-seg" : taskId === "semantic" ? "-sem" : taskId === "classify" ? "-cls" : "";
  return `${familyPrefix}${sizeId}${taskSuffix}`;
}

function yoloCheckpoint(familyId, taskId, sizeId) {
  const familyPrefix = familyId === "yolov8" ? "yolov8" : familyId;
  const taskSuffix = taskId === "segment" ? "-seg" : taskId === "semantic" ? "-sem" : taskId === "classify" ? "-cls" : "";
  return `${familyPrefix}${MODEL_SIZE_BY_ID[sizeId].code}${taskSuffix}.pt`;
}

function yoloModelLabel(familyId, taskId, sizeId) {
  const family = MODEL_FAMILY_BY_ID[familyId].label;
  const size = MODEL_SIZE_BY_ID[sizeId].label;
  const task = MODEL_TASK_BY_ID[taskId].label.toLowerCase();
  return taskId === "detect" || taskId === "classify"
    ? `${family} ${size}`
    : `${family} ${size} ${task}`;
}

function buildYoloModelCatalog() {
  return MODEL_FAMILIES
    .filter((family) => family.backend === "ultralytics")
    .flatMap((family) => family.tasks.flatMap((taskId) => MODEL_SIZES.map((size) => ({
      value: yoloModelValue(family.id, taskId, size.id),
      task: taskId,
      projectTask: taskId,
      family: family.id,
      size: size.id,
      backend: family.backend,
      label: yoloModelLabel(family.id, taskId, size.id),
      checkpoint: yoloCheckpoint(family.id, taskId, size.id),
      summary: `${MODEL_TASK_BY_ID[taskId].summary} ${MODEL_SIZE_BY_ID[size.id].summary}`,
      compatibility: "Ultralytics training controls and runtime augmentations are available.",
    }))));
}

const MODEL_CATALOG = [
  ...buildYoloModelCatalog(),
  {
    value: RFDETR_MODEL_SIZE,
    task: "detect",
    projectTask: "rfdetr",
    family: "rfdetr",
    size: "nano",
    backend: "rfdetr",
    label: "RF-DETR Nano",
    checkpoint: RFDETR_MODEL_SIZE,
    summary: "Transformer object detector. Lowest RF-DETR VRAM option.",
    compatibility: "Detection only. YOLO-format datasets are supported. RF-DETR training defaults are applied.",
  },
  {
    value: DFINE_MODEL_SIZE,
    task: "detect",
    projectTask: "dfine",
    family: "dfine",
    size: "nano",
    backend: "dfine",
    label: "D-FINE Nano",
    checkpoint: DFINE_MODEL_SIZE,
    summary: "D-FINE Nano detector. YOLO datasets are converted to COCO at training time.",
    compatibility: "Detection only. Requires an official D-FINE checkout in the training container.",
  },
];
const MODEL_BY_VALUE = Object.fromEntries(MODEL_CATALOG.map((model) => [model.value, model]));

const TASK_PROJECT_DEFAULTS = {
  detect: "runs/detect",
  rfdetr: "runs/rfdetr",
  dfine: "runs/dfine",
  segment: "runs/segment",
  semantic: "runs/semantic",
  classify: "runs/classify",
};

const KNOWN_TRAINING_PROJECTS = new Set(
  Object.values(TASK_PROJECT_DEFAULTS).flatMap((project) => [project, `/app/${project}`]),
);

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
    ...AUGMENTATION_DEFAULTS,
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
    ...AUGMENTATION_DEFAULTS,
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
    ...AUGMENTATION_DEFAULTS,
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
    ...AUGMENTATION_DEFAULTS,
  },
};

function taskForModelSize(modelSize) {
  const value = String(modelSize || "");
  const model = MODEL_BY_VALUE[value];
  if (model) {
    return model.projectTask;
  }
  if (value.startsWith("rfdetr-")) {
    return "rfdetr";
  }
  if (value.startsWith("dfine-")) {
    return "dfine";
  }
  if (value.endsWith("-seg")) {
    return "segment";
  }
  if (value.endsWith("-sem")) {
    return "semantic";
  }
  if (value.endsWith("-cls")) {
    return "classify";
  }
  return "detect";
}

function defaultProjectForModelSize(modelSize) {
  return TASK_PROJECT_DEFAULTS[taskForModelSize(modelSize)] || TASK_PROJECT_DEFAULTS.detect;
}

function modelSizeForTask(task, currentModelSize) {
  if (task === "rfdetr") {
    return RFDETR_MODEL_SIZE;
  }
  let base = String(currentModelSize || CONTROL_DEFAULTS["model-size"])
    .replace(/-(seg|sem|cls)$/i, "");
  if (task === "segment") {
    return `${base}-seg`;
  }
  if (task === "semantic") {
    if (!base.startsWith("yolo26-")) {
      base = "yolo26-nano";
    }
    return `${base}-sem`;
  }
  if (task === "classify") {
    return `${base}-cls`;
  }
  return base;
}

function normalizedProjectValue(value) {
  return normalizedPath(value || "").replace(/^\.\//, "");
}

function isKnownTrainingProject(value) {
  return !String(value || "").trim() || KNOWN_TRAINING_PROJECTS.has(normalizedProjectValue(value));
}

function syncProjectWithModelTask() {
  const project = $("project");
  if (!project || !isKnownTrainingProject(project.value)) {
    return;
  }
  project.value = defaultProjectForModelSize($("model-size").value);
  updateCurrentRunDisplay();
  scheduleTargetRefresh();
}

function applyModelFamilyDefaults() {
  const projectTask = taskForModelSize($("model-size").value);
  if (projectTask === "rfdetr") {
    Object.entries(RFDETR_DEFAULTS).forEach(([id, value]) => setControlValue(id, value));
  } else if (projectTask === "dfine") {
    Object.entries(DFINE_DEFAULTS).forEach(([id, value]) => setControlValue(id, value));
  } else {
    setControlValue("warmup-epochs", CONTROL_DEFAULTS["warmup-epochs"]);
    setControlValue("cos-lr", CONTROL_DEFAULTS["cos-lr"]);
  }
}

function syncScheduleControlLabels() {
  const isDfine = taskForModelSize($("model-size").value) === "dfine";
  const warmupLabel = $("warmup-label");
  const warmupInput = $("warmup-epochs");
  const cosLabel = $("cos-lr-label");
  if (warmupLabel) {
    warmupLabel.textContent = isDfine ? "Warmup steps" : "Warmup epochs";
    warmupLabel.dataset.tooltip = isDfine
      ? "D-FINE LinearWarmup duration in optimizer steps. Official D-FINE configs use warmup_duration."
      : "Warmup duration in epochs. Helps stabilize training at the beginning.";
  }
  if (warmupInput) {
    warmupInput.step = isDfine ? "1" : "0.1";
  }
  if (cosLabel) {
    cosLabel.textContent = isDfine ? "Cosine LR scheduler" : "Cosine LR";
  }
}

function syncModelFamilyControls() {
  const trainingLocked = state.running || state.isStarting || state.isStopping;
  const locked = trainingLocked || state.testRunning || state.testStarting || state.testStopping
    || state.annotationQaRunning || state.annotationQaStopping;
  const backend = selectedModelSpec()?.backend || "ultralytics";
  const isUltralytics = backend === "ultralytics";
  document.querySelectorAll("[data-ultralytics-only]").forEach((element) => {
    element.hidden = !isUltralytics;
    element.querySelectorAll("input, select, textarea, button").forEach((control) => {
      control.disabled = locked || !isUltralytics;
    });
  });
  syncScheduleControlLabels();
  syncAugmentationControls();
}

function augmentationConfigurationSummary() {
  if ((selectedModelSpec()?.backend || "ultralytics") !== "ultralytics") {
    return "Runtime augmentation: Not available for the selected model";
  }
  if (!$("augmentation-enabled").checked) {
    return "Augmentation: Off";
  }
  const labels = {
    mosaic: "mosaic",
    "hsv-h": "HSV hue",
    "hsv-s": "HSV saturation",
    "hsv-v": "HSV value",
    degrees: "rotation",
    translate: "translation",
    scale: "scale",
    shear: "shear",
    perspective: "perspective",
    flipud: "vertical flip",
    fliplr: "horizontal flip",
    bgr: "BGR swap",
    mixup: "MixUp",
    cutmix: "CutMix",
    "copy-paste": "copy-paste",
    erasing: "erasing",
  };
  const active = Object.entries(labels)
    .filter(([id]) => Number($(id).value) !== 0)
    .map(([id, label]) => `${label} ${$(id).value}`);
  const autoAugment = $("auto-augment").value.trim();
  if (autoAugment) {
    active.push(`auto augment ${autoAugment}`);
  }
  if (!$("disable-ultralytics-albumentations").checked) {
    active.push("optional Albumentations enabled");
  }
  return active.length
    ? `Augmentation: On — ${active.join(", ")}`
    : "Augmentation: On — all explicit values are zero";
}

function syncAugmentationControls() {
  const toggle = $("augmentation-enabled");
  const controls = $("augmentation-controls");
  if (!toggle || !controls) {
    return;
  }
  const locked = state.running || state.isStarting || state.isStopping
    || state.testRunning || state.testStarting || state.testStopping
    || state.annotationQaRunning || state.annotationQaStopping;
  const isUltralytics = (selectedModelSpec()?.backend || "ultralytics") === "ultralytics";
  const enabled = toggle.checked;
  toggle.disabled = locked || !isUltralytics;
  toggle.setAttribute("aria-checked", String(enabled));
  controls.disabled = locked || !isUltralytics || !enabled;
  $("augmentation-toggle-label").textContent = enabled ? "On" : "Off";
  $("augmentation-status").textContent = enabled
    ? "On — the configured values will be applied during training."
    : "Off — no augmentation will be applied. Configured values are preserved.";
  toggle.closest(".advanced-group")?.classList.toggle("augmentation-is-enabled", enabled);
}

function selectedModelSpec() {
  return MODEL_BY_VALUE[$("model-size").value] || MODEL_BY_VALUE[CONTROL_DEFAULTS["model-size"]] || MODEL_CATALOG[0];
}

function syncModelSelectorDisplay() {
  const model = selectedModelSpec();
  if ($("model-selector-value") && model) {
    $("model-selector-value").textContent = `${model.label} - ${model.checkpoint}`;
  }
  if ($("selected-model-summary") && model) {
    $("selected-model-summary").textContent = model.compatibility;
  }
  if ($("model-compatibility") && model) {
    $("model-compatibility").textContent = model.compatibility;
  }
}

function modelOptionSearchText(model) {
  const task = MODEL_TASK_BY_ID[model.task]?.label || "";
  const family = MODEL_FAMILY_BY_ID[model.family]?.label || "";
  const size = MODEL_SIZE_BY_ID[model.size]?.label || "";
  return `${task} ${family} ${size} ${model.label} ${model.checkpoint} ${model.value} ${model.summary}`.toLowerCase();
}

function renderSelectOptions(select, options, selectedValue) {
  if (!select) {
    return;
  }
  select.innerHTML = "";
  options.forEach((option) => {
    const element = document.createElement("option");
    element.value = option.id;
    element.textContent = option.label;
    select.appendChild(element);
  });
  if (options.some((option) => option.id === selectedValue)) {
    select.value = selectedValue;
  } else if (options.length) {
    select.value = options[0].id;
  }
}

function availableModelFamilies(taskId) {
  return MODEL_FAMILIES.filter((family) => MODEL_CATALOG.some((model) => (
    model.task === taskId && model.family === family.id
  )));
}

function guidedModelValue(taskId, familyId) {
  const currentSize = selectedModelSpec()?.size;
  const matchingModels = MODEL_CATALOG.filter((model) => (
    model.task === taskId && model.family === familyId
  ));
  return (
    matchingModels.find((model) => model.size === currentSize)?.value
    || matchingModels[0]?.value
    || ""
  );
}

function syncGuidedControlsFromModel() {
  const current = selectedModelSpec();
  const taskSelect = $("model-task");
  const familySelect = $("model-family");
  if (!taskSelect || !familySelect || !current) {
    return;
  }
  renderSelectOptions(taskSelect, MODEL_TASKS, current.task);
  const families = availableModelFamilies(taskSelect.value);
  renderSelectOptions(familySelect, families, current.family);
}

function chooseGuidedModel() {
  const taskSelect = $("model-task");
  const familySelect = $("model-family");
  if (!taskSelect || !familySelect) {
    return;
  }
  const families = availableModelFamilies(taskSelect.value);
  if (!families.some((family) => family.id === familySelect.value)) {
    renderSelectOptions(familySelect, families, families[0]?.id || "");
  }
  const value = guidedModelValue(taskSelect.value, familySelect.value);
  if (value) {
    chooseModelOption(value, { close: false, focusToggle: false });
  }
}

function closeModelSelector() {
  $("model-selector-menu").hidden = true;
  $("model-selector-toggle").setAttribute("aria-expanded", "false");
}

function openModelSelector() {
  $("model-selector-menu").hidden = false;
  $("model-selector-toggle").setAttribute("aria-expanded", "true");
  filterModelOptions();
  $("model-search").focus();
}

function chooseModelOption(value, { close = true, focusToggle = true } = {}) {
  const select = $("model-size");
  if (select.value !== value) {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  } else {
    syncModelSelectorDisplay();
    syncGuidedControlsFromModel();
    filterModelOptions();
  }
  if (close) {
    closeModelSelector();
  }
  if (focusToggle) {
    $("model-selector-toggle").focus();
  }
}

function filterModelOptions() {
  const search = $("model-search");
  const optionsContainer = $("model-options");
  const query = String(search?.value || "").trim().toLowerCase();
  if (!search || !optionsContainer) {
    return;
  }
  syncGuidedControlsFromModel();
  const taskId = $("model-task").value;
  const familyId = $("model-family").value;
  optionsContainer.innerHTML = "";
  const matches = MODEL_CATALOG.filter((model) => (
    query
      ? modelOptionSearchText(model).includes(query)
      : model.task === taskId && model.family === familyId
  ));
  matches.forEach((model) => {
    const task = MODEL_TASK_BY_ID[model.task];
    const family = MODEL_FAMILY_BY_ID[model.family];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(model.value === $("model-size").value));
    button.classList.toggle("active", model.value === $("model-size").value);
    button.dataset.value = model.value;

    const header = document.createElement("span");
    header.className = "model-option-header";
    const taskBadge = document.createElement("span");
    taskBadge.className = "model-task-badge";
    taskBadge.dataset.task = task.badge;
    taskBadge.textContent = task.label;
    const familyLabel = document.createElement("span");
    familyLabel.className = "model-family-label";
    familyLabel.textContent = family.label;
    header.append(taskBadge, familyLabel);

    const title = document.createElement("span");
    title.className = "model-option-title";
    title.textContent = `${model.label} - ${model.checkpoint}`;

    button.append(header, title);
    button.addEventListener("click", () => chooseModelOption(model.value));
    optionsContainer.appendChild(button);
  });
  if (!matches.length) {
    const empty = document.createElement("div");
    empty.className = "model-options-empty";
    empty.textContent = "No matching models.";
    optionsContainer.appendChild(empty);
  }
  syncModelSelectorDisplay();
}

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
    reading_labels: [82, 87],
    calculating_targets: [87, 89],
    assigning: [89, 93],
    finalizing_split: [93, 94],
    copying: [94, 98],
    rebuilding_split: [98, 99],
    normalizing_paths: [82, 85],
    inspecting: [98, 100],
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
    rebuilding_split: "Rebuilding local split",
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

function closeDatasetCleanupMenu() {
  const menu = $("dataset-cleanup-menu");
  const toggle = $("dataset-cleanup-toggle");
  menu.classList.remove("is-open");
  toggle.setAttribute("aria-expanded", "false");
}

function toggleDatasetCleanupMenu() {
  const menu = $("dataset-cleanup-menu");
  const toggle = $("dataset-cleanup-toggle");
  const expanded = toggle.getAttribute("aria-expanded") === "true";
  menu.classList.toggle("is-open", !expanded);
  toggle.setAttribute("aria-expanded", String(!expanded));
}

function syncActionStates() {
  const trainingLocked = state.running || state.isStarting || state.isStopping;
  const testLocked = state.testRunning || state.testStarting || state.testStopping;
  const qaLocked = state.annotationQaRunning || state.annotationQaStopping;
  const locked = trainingLocked || testLocked || qaLocked;
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
  const annotatedDatasetDownloadActive = state.downloads.has("annotated_dataset");
  $("download-dataset").disabled = !hasDataset || preparing || datasetDownloadActive;
  $("download-dataset").textContent = datasetDownloadActive ? "Preparing ZIP..." : "Download ZIP";
  $("download-dataset").setAttribute("aria-busy", String(datasetDownloadActive));
  $("download-annotated-dataset").disabled = !hasDataset || preparing || annotatedDatasetDownloadActive;
  $("download-annotated-dataset").textContent = annotatedDatasetDownloadActive
    ? "Preparing Annotated ZIP..."
    : "Download Annotated ZIP";
  $("download-annotated-dataset").setAttribute("aria-busy", String(annotatedDatasetDownloadActive));
  [
    ["clear-dataset-uploads", "dataset_uploads", "Clear Uploaded ZIPs"],
    ["clear-dataset-extracted", "dataset_extracted", "Clear Extracted"],
    ["clear-dataset-prepared", "dataset_prepared", "Clear Prepared"],
  ].forEach(([id, key, label]) => {
    const active = state.storageCleanup.has(key);
    $(id).disabled = locked || preparing || active;
    $(id).textContent = active ? "Clearing..." : label;
    $(id).setAttribute("aria-busy", String(active));
  });
  $("dataset-cleanup-toggle").disabled = locked || preparing;
  if ($("dataset-cleanup-toggle").disabled) {
    closeDatasetCleanupMenu();
  }
  $("start-training").disabled = locked || preparing || (!hasDataset && !canResume);
  $("start-training").textContent = state.isStarting ? "Starting..." : "Start";
  $("start-training").setAttribute("aria-busy", String(state.isStarting));
  $("stop-training").disabled = !state.running || state.isStopping;
  $("stop-training").textContent = state.isStopping ? "Stopping..." : "Stop";
  $("stop-training").setAttribute("aria-busy", String(state.isStopping));
  syncAnnotationQaActionStates();

  document.querySelectorAll(".training-panel input, .training-panel select, .advanced-panel input, .advanced-panel select").forEach((control) => {
    control.disabled = locked;
  });
  $("model-selector-toggle").disabled = locked;
  if (locked) {
    closeModelSelector();
  }
  $("resume").disabled = locked || !state.resumeAvailable;
  document.querySelectorAll("[data-preset], #reset-advanced").forEach((button) => {
    button.disabled = locked;
  });
  syncModelFamilyControls();

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
    upload: "Class names are read from data.yaml during dataset preparation.",
    folder: "Use Auto Fetch to preview class names from data.yaml, or prepare the dataset to fill them automatically.",
    roboflow: "Class names are supplied by the Roboflow dataset version.",
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

function optionalTextValue(id) {
  const value = $(id).value.trim();
  return value === "" ? null : value;
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
  if (id === "model-size") {
    filterModelOptions();
  }
}

function applyControlValues(values) {
  Object.entries(values).forEach(([id, value]) => setControlValue(id, value));
  applyModelFamilyDefaults();
  if (Object.hasOwn(values, "model-size")) {
    syncProjectWithModelTask();
  }
  updateCurrentRunDisplay();
  syncModelFamilyControls();
  syncActionStates();
  if (Object.hasOwn(values, "project") || Object.hasOwn(values, "run-name") || Object.hasOwn(values, "model-size")) {
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

function isDefaultTrainingTarget() {
  const target = weightTarget();
  return normalizedProjectValue(target.project) === normalizedProjectValue(CONTROL_DEFAULTS.project)
    && target.name === CONTROL_DEFAULTS["run-name"]
    && $("model-size").value === CONTROL_DEFAULTS["model-size"];
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
  const progressDetail = typeof progress.detail === "string" ? progress.detail.trim() : "";
  const percent = total
    ? Math.min(100, Math.max(0, Number(progress.percent) || (current / total) * 100))
    : 0;

  let label = "Waiting to start";
  let detail = "The current epoch will appear here when training starts.";
  if (phase === "starting") {
    label = total ? `Starting a ${total}-epoch run` : "Starting training";
    detail = "Loading the model and preparing the dataloaders.";
  } else if (phase === "training") {
    label = current && total ? `Running epoch ${current} of ${total}` : "Training running";
    detail = progressDetail || (completed
      ? `${completed} ${completed === 1 ? "epoch has" : "epochs have"} finished validation.`
      : "Training is running. Waiting for the first validation update.");
  } else if (phase === "completed") {
    label = total ? `Training completed: ${completed || current} of ${total}` : "Training completed";
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
  updateInferenceFileSelection();
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

function numberText(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "N/A";
  }
  return number.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function millisecondsText(value) {
  const text = numberText(value, 1);
  return text === "N/A" ? text : `${text} ms/image`;
}

function durationText(seconds) {
  const number = Number(seconds);
  if (!Number.isFinite(number)) {
    return "N/A";
  }
  if (number >= 60) {
    const minutes = Math.floor(number / 60);
    const remaining = number - (minutes * 60);
    return `${minutes}m ${remaining.toFixed(1)}s`;
  }
  return `${number.toFixed(1)}s`;
}

function renderTestTiming(timing) {
  const payload = timing || {};
  $("test-timing-images").textContent = numberText(payload.image_count, 0);
  $("test-timing-inference").textContent = millisecondsText(payload.inference_ms_per_image);
  $("test-timing-preprocess").textContent = millisecondsText(payload.preprocess_ms_per_image);
  $("test-timing-postprocess").textContent = millisecondsText(payload.postprocess_ms_per_image);
  $("test-timing-eval-processing").textContent = millisecondsText(payload.evaluation_ms_per_image);
  $("test-timing-total").textContent = durationText(payload.total_seconds);
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

function resetAnnotationQaForDataset() {
  state.annotationQaJobId = "";
  state.annotationQaReport = null;
  state.annotationQaRunning = false;
  state.annotationQaStopping = false;
  state.annotationQaApplyingFixes = false;
  state.annotationQaCorrectedDatasetYaml = "";
  state.annotationQaQueue = "needs_review";
  state.annotationQaPage = 1;
  state.annotationQaFilters = { search: "", severity: "", split: "", className: "", issueType: "" };
  state.annotationQaUndo = null;
  state.annotationQaSelected.clear();
  window.clearTimeout(state.annotationQaUndoTimer);
  window.clearTimeout(state.annotationQaPollTimer);
  state.annotationQaPollTimer = null;
  state.annotationQaPollRevision += 1;
  $("annotation-qa-status").textContent = state.datasetYaml ? "Not run" : "Not available";
  $("annotation-qa-status-heading").textContent = state.datasetYaml
    ? "Ready to scan"
    : "Ready when your dataset is prepared";
  $("annotation-qa-detail").textContent = state.datasetYaml
    ? "SAM QA is optional. Run it before training when you want an annotation quality check."
    : "Prepare a dataset to enable optional SAM QA.";
  $("annotation-qa-dataset-ready").textContent = state.datasetYaml
    ? "Prepared dataset selected. The scan will not modify its labels."
    : "Prepare a dataset to enable Annotation QA.";
  $("annotation-qa-summary").innerHTML = "";
  $("annotation-qa-issues").innerHTML = "";
  $("annotation-qa-review-stage").hidden = true;
  $("annotation-qa-finalize-stage").hidden = true;
  $("annotation-qa-config-stage").open = true;
  $("annotation-qa-run-stage").open = true;
  $("annotation-qa-tab-count").hidden = true;
  $("annotation-qa-search").value = "";
  ["annotation-qa-filter-severity", "annotation-qa-filter-split", "annotation-qa-filter-class", "annotation-qa-filter-type"].forEach((id) => {
    $(id).value = "";
  });
  hideAnnotationQaUndo();
  setAnnotationQaProgress(false);
  updateAnnotationQaConfigurationSummary();
  updateAnnotationQaWorkflow();
  syncAnnotationQaActionStates();
}

function annotationQaIsResolved(issue) {
  return ["fix_accepted", "accepted", "false_positive", "ignored"].includes(issue?.review_status)
    && !(issue?.audit_required && issue?.audit_status === "pending");
}

function annotationQaIssueCounts(issues = annotationQaIssuesList()) {
  const counts = {
    all: issues.length,
    needsReview: 0,
    reviewed: 0,
    highPriority: 0,
    audits: 0,
    safeSuggestions: 0,
    manual: 0,
  };
  issues.forEach((issue) => {
    const unresolved = !annotationQaIsResolved(issue) && issue.review_status !== "needs_fix";
    if (unresolved) {
      counts.needsReview += 1;
    } else {
      counts.reviewed += 1;
    }
    if (unresolved && annotationQaSeverityKey(issue) === "high") {
      counts.highPriority += 1;
    }
    if (issue.audit_required && issue.audit_status === "pending") {
      counts.audits += 1;
    }
    if (unresolved && issueCanAcceptSamBox(issue)) {
      counts.safeSuggestions += 1;
    }
    if (issue.review_status === "needs_fix" || (!annotationQaIsResolved(issue) && issue.difference_band === "large_disagreement")) {
      counts.manual += 1;
    }
  });
  return counts;
}

function updateAnnotationQaConfigurationSummary() {
  if (!$("annotation-qa-scope")) {
    return;
  }
  const scopeLabels = { all: "all splits", val: "validation", train: "training", test: "test" };
  const presetLabels = { balanced: "Balanced review", lenient: "Quick check", strict: "Thorough review" };
  const modeLabels = { shadow: "suggestions only", automatic: "safe corrections + audit", manual: "human decisions only" };
  const scope = $("annotation-qa-scope").value;
  const preset = $("annotation-qa-preset").value;
  const mode = $("annotation-qa-auto-mode").value;
  $("annotation-qa-config-summary").textContent = `${presetLabels[preset]} · ${scopeLabels[scope]} · ${modeLabels[mode]}`;
  const modeText = mode === "automatic"
    ? "Policy-approved SAM replacements may be queued automatically; sampled decisions must be audited before finalization."
    : mode === "manual"
      ? "Every flagged annotation requires a human decision; no automatic corrections will be queued."
      : "SAM may suggest safer boxes, but nothing will be queued without a reviewer.";
  $("annotation-qa-configuration-explainer").textContent = `Scan ${scopeLabels[scope]} with ${presetLabels[preset].toLowerCase()} sensitivity. ${modeText}`;
}

function updateAnnotationQaWorkflow() {
  const hasReport = Boolean(state.annotationQaReport);
  const running = state.annotationQaRunning || state.annotationQaStopping;
  const counts = annotationQaIssueCounts();
  const current = running ? "run" : hasReport ? (counts.needsReview || counts.audits ? "review" : "finalize") : "config";
  const order = ["config", "run", "review", "finalize"];
  order.forEach((stage, index) => {
    const indicator = $(`qa-stage-indicator-${stage}`);
    if (!indicator) {
      return;
    }
    const currentIndex = order.indexOf(current);
    indicator.classList.toggle("is-active", stage === current);
    indicator.classList.toggle("is-complete", index < currentIndex || (stage === "review" && hasReport && !counts.needsReview && !counts.audits));
  });
  $("annotation-qa-review-stage").hidden = !hasReport;
  $("annotation-qa-finalize-stage").hidden = !hasReport;
  $("annotation-qa-review-progress-label").textContent = `${counts.reviewed} of ${counts.all} reviewed`;
  const tabCount = $("annotation-qa-tab-count");
  tabCount.textContent = String(counts.needsReview);
  tabCount.hidden = !hasReport || counts.needsReview === 0;
}

function setAnnotationQaProgress(visible, job = {}) {
  const progress = $("annotation-qa-progress");
  progress.hidden = !visible;
  progress.setAttribute("aria-busy", String(visible));
  if (!visible) {
    return;
  }
  const percent = Math.max(0, Math.min(100, Number(job.percent) || 0));
  const stage = job.stage || "running";
  $("annotation-qa-stage").textContent = stage.replace(/_/g, " ");
  $("annotation-qa-percent").textContent = `${Math.round(percent)}%`;
  const track = $("annotation-qa-track");
  track.setAttribute("aria-valuenow", String(Math.round(percent)));
  track.setAttribute("aria-valuetext", `${stage}: ${Math.round(percent)}%`);
  track.querySelector("span").style.width = `${percent}%`;
}

function annotationQaAcceptedFixCount() {
  const issues = Array.isArray(state.annotationQaReport?.issues) ? state.annotationQaReport.issues : [];
  return issues.filter((issue) => (
    issue.accepted_fix === "sam_box"
    || (issue.accepted_class_id !== null && issue.accepted_class_id !== undefined)
  )).length;
}

function annotationQaPendingAuditCount() {
  const issues = Array.isArray(state.annotationQaReport?.issues) ? state.annotationQaReport.issues : [];
  return issues.filter((issue) => issue.audit_required && issue.audit_status === "pending").length;
}

function annotationQaFixCounts() {
  const issues = Array.isArray(state.annotationQaReport?.issues) ? state.annotationQaReport.issues : [];
  return issues.reduce((counts, issue) => {
    if (issue.accepted_fix === "sam_box") {
      counts.box += 1;
    }
    if (issue.accepted_class_id !== null && issue.accepted_class_id !== undefined) {
      counts.class += 1;
    }
    return counts;
  }, { box: 0, class: 0 });
}

function annotationQaHasLegacySamFixes() {
  const reportVersion = Number(state.annotationQaReport?.summary?.report_version || 1);
  const issues = Array.isArray(state.annotationQaReport?.issues) ? state.annotationQaReport.issues : [];
  return reportVersion < ANNOTATION_QA_REPORT_VERSION
    && issues.some((issue) => issue.accepted_fix === "sam_box");
}

function renderAnnotationQaFixSummary() {
  const summary = $("annotation-qa-fix-summary");
  if (!summary) {
    return;
  }
  if (!state.annotationQaReport) {
    summary.textContent = "";
    return;
  }
  const counts = annotationQaFixCounts();
  if (annotationQaHasLegacySamFixes()) {
    summary.textContent = "Queued SAM fixes use an older QA report. Rerun SAM QA to apply the maximum-difference safety gate before creating a corrected dataset.";
    return;
  }
  const pendingAudits = annotationQaPendingAuditCount();
  if (pendingAudits) {
    summary.textContent = `${pendingAudits} sampled automatic decision${pendingAudits === 1 ? "" : "s"} require an audit before creating a corrected dataset.`;
    return;
  }
  const total = counts.box + counts.class;
  if (!total) {
    summary.textContent = state.annotationQaCorrectedDatasetYaml
      ? "Corrected dataset is ready."
      : "No corrections queued.";
    return;
  }
  const parts = [];
  if (counts.box) {
    parts.push(`${counts.box} box ${counts.box === 1 ? "fix" : "fixes"}`);
  }
  if (counts.class) {
    parts.push(`${counts.class} class ${counts.class === 1 ? "change" : "changes"}`);
  }
  summary.textContent = `Queued corrections: ${parts.join(", ")}.`;
}

function renderAnnotationQaCompletionChecklist() {
  const container = $("annotation-qa-completion-checklist");
  if (!container || !state.annotationQaReport) {
    if (container) {
      container.innerHTML = "";
    }
    return;
  }
  const counts = annotationQaIssueCounts();
  const corrections = annotationQaAcceptedFixCount();
  const audits = annotationQaPendingAuditCount();
  const reviewComplete = counts.needsReview === 0;
  const auditComplete = audits === 0;
  const hasCorrections = corrections > 0;
  const item = (complete, text, blockedText = text, queue = "") => {
    const tag = !complete && queue ? "button" : "div";
    const attributes = !complete && queue ? ` type="button" data-qa-open-queue="${queue}"` : "";
    return `
    <${tag} class="qa-checklist-item ${complete ? "is-complete" : "is-blocked"}"${attributes}>
      <span>${complete ? "✓" : "!"}</span>
      <strong>${escapeHtml(complete ? text : blockedText)}</strong>
    </${tag}>
  `;
  };
  container.innerHTML = [
    item(reviewComplete, "All review decisions complete", `${counts.needsReview} annotation${counts.needsReview === 1 ? "" : "s"} still need review`, "needs_review"),
    item(auditComplete, "Required audits complete", `${audits} required audit${audits === 1 ? "" : "s"} pending`, "audits"),
    item(hasCorrections || Boolean(state.annotationQaCorrectedDatasetYaml), `${corrections} correction${corrections === 1 ? "" : "s"} ready`, "No corrections have been queued"),
  ].join("");
  container.onclick = (event) => {
    const trigger = event.target.closest("[data-qa-open-queue]");
    if (!trigger) {
      return;
    }
    state.annotationQaQueue = trigger.dataset.qaOpenQueue;
    state.annotationQaPage = 1;
    renderAnnotationQaIssues();
    $("annotation-qa-review-stage").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  $("annotation-qa-finalize-state").textContent = state.annotationQaCorrectedDatasetYaml
    ? "Corrected dataset ready"
    : audits
      ? `${audits} audit${audits === 1 ? "" : "s"} blocking finalization`
      : hasCorrections
        ? `${corrections} correction${corrections === 1 ? "" : "s"} ready`
        : "No corrections queued";
}

function syncAnnotationQaActionStates() {
  const hasDataset = Boolean(state.datasetYaml);
  const trainingLocked = state.running || state.isStarting || state.isStopping;
  const testLocked = state.testRunning || state.testStarting || state.testStopping;
  const preparing = state.isPreparing || state.isDetecting;
  const running = state.annotationQaRunning || state.annotationQaStopping || state.annotationQaApplyingFixes;
  $("run-annotation-qa").disabled = !hasDataset || preparing || trainingLocked || testLocked || running;
  $("run-annotation-qa").textContent = state.annotationQaRunning ? "Running scan..." : "Run Annotation QA";
  $("run-annotation-qa").setAttribute("aria-busy", String(state.annotationQaRunning));
  $("stop-annotation-qa").disabled = !state.annotationQaRunning || state.annotationQaStopping;
  $("stop-annotation-qa").hidden = !state.annotationQaRunning && !state.annotationQaStopping;
  $("stop-annotation-qa").textContent = state.annotationQaStopping ? "Stopping..." : "Stop";
  $("stop-annotation-qa").setAttribute("aria-busy", String(state.annotationQaStopping));
  [
    "annotation-qa-model", "annotation-qa-scope", "annotation-qa-preset", "annotation-qa-tolerance",
    "annotation-qa-max-difference", "annotation-qa-auto-mode", "annotation-qa-audit",
    "annotation-qa-prompt-expansion", "annotation-qa-prompt-jitter", "annotation-qa-stability-iou",
    "annotation-qa-stability-edge", "annotation-qa-auto-quality", "annotation-qa-auto-iou",
    "annotation-qa-auto-center", "annotation-qa-auto-neighbor",
  ].forEach((id) => {
    $(id).disabled = !hasDataset || preparing || trainingLocked || testLocked || running;
  });
  const reportReady = Boolean(state.annotationQaJobId && state.annotationQaReport);
  $("download-annotation-qa-csv").disabled = !reportReady;
  $("download-annotation-qa-json").disabled = !reportReady;
  $("apply-annotation-qa-fixes").disabled = !reportReady
    || annotationQaAcceptedFixCount() === 0
    || annotationQaPendingAuditCount() > 0
    || annotationQaHasLegacySamFixes()
    || preparing
    || trainingLocked
    || testLocked
    || running;
  $("apply-annotation-qa-fixes").textContent = state.annotationQaApplyingFixes ? "Creating..." : "Create Corrected Dataset";
  $("download-corrected-dataset").disabled = !state.annotationQaCorrectedDatasetYaml || state.downloads.has("corrected_dataset");
  renderAnnotationQaFixSummary();
  renderAnnotationQaCompletionChecklist();
  updateAnnotationQaWorkflow();
}

function annotationQaStatusLabel(status) {
  const labels = {
    queued: "Queued",
    running: "Running",
    stopping: "Stopping",
    stopped: "Stopped",
    completed: "Complete",
    failed: "Failed",
  };
  return labels[status] || "Not run";
}

function annotationQaReviewStatusLabel(status) {
  const labels = {
    unreviewed: "unreviewed",
    fix_accepted: "correction queued",
    needs_fix: "manual fix",
    accepted: "YOLO kept",
    false_positive: "suspected false positive",
    ignored: "ignored",
  };
  return labels[status] || String(status || "").replace(/_/g, " ");
}

function renderAnnotationQaSummary(summary = {}) {
  const container = $("annotation-qa-summary");
  if (!summary || !Object.keys(summary).length) {
    container.innerHTML = "";
    return;
  }
  const issues = annotationQaIssuesList();
  const counts = annotationQaIssueCounts(issues);
  const percent = counts.all ? Math.round((counts.reviewed / counts.all) * 100) : 100;
  container.innerHTML = `
    <div class="qa-summary-grid">
      <div class="is-actionable"><span>Needs review</span><strong>${counts.needsReview}</strong></div>
      <div class="is-priority"><span>High priority</span><strong>${counts.highPriority}</strong></div>
      <div><span>Safe SAM suggestions</span><strong>${counts.safeSuggestions}</strong></div>
      <div><span>Audits pending</span><strong>${counts.audits}</strong></div>
      <div><span>Reviewed</span><strong>${counts.reviewed}/${counts.all}</strong></div>
      <div class="qa-review-progress">
        <span>Review progress</span>
        <strong>${percent}%</strong>
        <div class="qa-review-progress-track" role="progressbar" aria-label="Annotation review progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"><span style="width:${percent}%"></span></div>
      </div>
    </div>
  `;
  const runDetails = $("annotation-qa-run-detail-content");
  if (runDetails) {
    runDetails.innerHTML = `
      <div><span>Images scanned</span><strong>${summary.images_scanned || 0}</strong></div>
      <div><span>Labels checked</span><strong>${summary.labels_checked || 0}</strong></div>
      <div><span>Model</span><strong>${escapeHtml(summary.model || summary.sam_model || $("annotation-qa-model").value)}</strong></div>
      <div><span>Scope</span><strong>${escapeHtml(summary.scope || $("annotation-qa-scope").value)}</strong></div>
      <div><span>Box tolerance</span><strong>${Number(summary.box_tolerance_percent ?? 5).toFixed(1)}%</strong></div>
      <div><span>SAM difference limit</span><strong>${Number(summary.sam_max_difference_percent ?? 25).toFixed(1)}%</strong></div>
      <div><span>YOLO kept automatically</span><strong>${summary.qa_decisions?.auto_keep_yolo || summary.yolo_boxes_accepted || 0}</strong></div>
      <div><span>SAM replacements blocked</span><strong>${summary.sam_replacements_blocked || 0}</strong></div>
    `;
  }
}

function annotationQaDecisionLabel(decision) {
  const labels = {
    auto_keep_yolo: "Auto kept YOLO",
    auto_replace_sam: "Auto replace candidate",
    human_review: "Human review",
    manual_only: "Manual only",
  };
  return labels[String(decision || "")] || "Human review";
}

function annotationQaPreviewAssetUrl(issue, field = "preview") {
  const preview = String(issue?.[field] || "");
  if (!preview || !state.annotationQaJobId) {
    return "";
  }
  const name = preview.split("/").pop();
  return `/api/annotation-qa/preview/${encodeURIComponent(state.annotationQaJobId)}/${encodeURIComponent(name)}`;
}

function annotationQaPreviewUrl(issue) {
  return annotationQaPreviewAssetUrl(issue, "preview");
}

function annotationQaSeverityKey(issue) {
  if (issue?.severity === "high") {
    return "high";
  }
  if (issue?.severity === "medium") {
    return "medium";
  }
  return "low";
}

function annotationQaIssueTypeLabel(issueType) {
  const labels = {
    low_box_agreement: "Low box agreement",
    low_confidence_mask: "Low-confidence SAM mask",
    low_mask_coverage: "Low SAM mask coverage",
    loose_box: "Possibly loose YOLO box",
    shifted_box: "Shifted box",
    possibly_tight_box: "Possibly tight YOLO box",
    moderate_box_difference: "Moderate box difference",
    large_box_disagreement: "Large box disagreement",
    unstable_sam_prompt: "Unstable SAM prompt result",
    auto_keep_audit: "Automatic keep audit",
    empty_mask: "Empty SAM mask",
    sam_mapping_error: "SAM mapping error",
    duplicate_box: "Possible duplicate box",
    invalid_label: "Invalid label",
    unsupported_annotation: "Unsupported annotation",
    image_read_error: "Image read error",
  };
  const normalized = String(issueType || "");
  if (labels[normalized]) {
    return labels[normalized];
  }
  const readable = normalized.replace(/_/g, " ");
  return readable ? `${readable.charAt(0).toUpperCase()}${readable.slice(1)}` : "Unknown issue";
}

function issueCanBulkAcceptSamBox(issue) {
  return issueCanAcceptSamBox(issue) && !(issue.audit_required && issue.audit_status === "pending");
}

function annotationQaIssueRows(issues) {
  return issues.map((issue) => {
    const previewUrl = annotationQaPreviewUrl(issue);
    const imageName = escapeHtml(issue.image_name || "image");
    const imageIssueCount = annotationQaIssuesList().filter((item) => item.image === issue.image).length;
    const splitClass = escapeHtml(`${issue.split || ""} · ${issue.class_name || ""}${imageIssueCount > 1 ? ` · ${imageIssueCount} issues on image` : ""}`);
    const issueType = escapeHtml(annotationQaIssueTypeLabel(issue.issue_type));
    const decision = escapeHtml(annotationQaDecisionLabel(issue.qa_decision));
    const issueId = escapeHtml(issue.issue_id);
    const severity = escapeHtml(annotationQaSeverityKey(issue));
    const thumb = previewUrl
      ? `<img src="${previewUrl}" alt="">`
      : '<span class="qa-no-preview">No preview</span>';
    const selector = state.annotationQaQueue === "safe_suggestions" && issueCanBulkAcceptSamBox(issue)
      ? `<label class="qa-select-issue"><input type="checkbox" data-qa-select="${issueId}"${state.annotationQaSelected.has(issue.issue_id) ? " checked" : ""}> Select</label>`
      : "";
    return `
      <tr class="qa-issue-row" data-qa-open="${issueId}" data-qa-severity="${severity}">
        <td class="qa-preview-cell">
          <button class="qa-preview-button" type="button" data-qa-open="${issueId}" data-qa-severity="${severity}" aria-label="Review ${imageName}">
            ${thumb}
          </button>
          ${selector}
        </td>
        <td>
          <strong title="${imageName}">${imageName}</strong>
          <small title="${splitClass}">${splitClass}</small>
        </td>
        <td title="${issueType}"><span class="qa-severity ${severity}">${severity}</span><strong>${issueType}</strong></td>
        <td>${metricText(issue.score)}</td>
        <td title="${decision}">${decision}</td>
        <td>
          <span class="qa-status-label ${escapeHtml(issue.review_status || "unreviewed")}">${escapeHtml(annotationQaReviewStatusLabel(issue.review_status))}</span>
          <button class="secondary compact qa-review-inline-button" type="button" data-qa-open="${issueId}" data-qa-severity="${severity}">Review</button>
        </td>
      </tr>
    `;
  }).join("");
}

function annotationQaIssueTable(issues, severity) {
  const rows = annotationQaIssueRows(issues);
  const title = `${severity.charAt(0).toUpperCase()}${severity.slice(1)} Severity`;
  return `
    <section class="qa-severity-section ${severity}" data-qa-severity="${severity}">
      <div class="qa-severity-header">
        <h3>${title}</h3>
        <span>${issues.length}</span>
      </div>
      <div class="qa-table-wrap">
        <table class="qa-table">
          <thead>
            <tr>
              <th>Preview</th>
              <th>Image</th>
              <th>Issue</th>
              <th>Score</th>
              <th>Decision</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function annotationQaQueueMatches(issue, queue) {
  const unresolved = !annotationQaIsResolved(issue) && issue.review_status !== "needs_fix";
  if (queue === "needs_review") {
    return unresolved;
  }
  if (queue === "audits") {
    return issue.audit_required && issue.audit_status === "pending";
  }
  if (queue === "safe_suggestions") {
    return unresolved && issueCanAcceptSamBox(issue);
  }
  if (queue === "manual") {
    return issue.review_status === "needs_fix" || (!annotationQaIsResolved(issue) && issue.difference_band === "large_disagreement");
  }
  if (queue === "resolved") {
    return annotationQaIsResolved(issue);
  }
  return true;
}

function annotationQaFilteredIssues() {
  const filters = state.annotationQaFilters;
  const severityRank = { high: 0, medium: 1, low: 2 };
  return annotationQaIssuesList()
    .filter((issue) => annotationQaQueueMatches(issue, state.annotationQaQueue))
    .filter((issue) => !filters.search || String(issue.image_name || "").toLowerCase().includes(filters.search))
    .filter((issue) => !filters.severity || annotationQaSeverityKey(issue) === filters.severity)
    .filter((issue) => !filters.split || String(issue.split || "") === filters.split)
    .filter((issue) => !filters.className || String(issue.class_name || "") === filters.className)
    .filter((issue) => !filters.issueType || String(issue.issue_type || "") === filters.issueType)
    .sort((left, right) => (
      (severityRank[annotationQaSeverityKey(left)] - severityRank[annotationQaSeverityKey(right)])
      || (Number(right.score || 0) - Number(left.score || 0))
      || String(left.image_name || "").localeCompare(String(right.image_name || ""))
    ));
}

function setAnnotationQaFilterOptions(id, emptyLabel, values, labeler = (value) => value) {
  const select = $(id);
  const selected = select.value;
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = emptyLabel;
  select.appendChild(empty);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labeler(value);
    select.appendChild(option);
  });
  select.value = values.includes(selected) ? selected : "";
}

function renderAnnotationQaFilterOptions() {
  const issues = annotationQaIssuesList();
  const values = (key) => [...new Set(issues.map((issue) => String(issue[key] || "")).filter(Boolean))].sort();
  setAnnotationQaFilterOptions("annotation-qa-filter-severity", "All severities", ["high", "medium", "low"]);
  setAnnotationQaFilterOptions("annotation-qa-filter-split", "All splits", values("split"));
  setAnnotationQaFilterOptions("annotation-qa-filter-class", "All classes", values("class_name"));
  setAnnotationQaFilterOptions("annotation-qa-filter-type", "All issue types", values("issue_type"), annotationQaIssueTypeLabel);
}

function renderAnnotationQaQueueCounts() {
  const issues = annotationQaIssuesList();
  document.querySelectorAll("[data-qa-queue]").forEach((button) => {
    const queue = button.dataset.qaQueue;
    const count = issues.filter((issue) => annotationQaQueueMatches(issue, queue)).length;
    button.querySelector("span").textContent = String(count);
    const active = queue === state.annotationQaQueue;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

function renderAnnotationQaBulkToolbar(pageIssues = []) {
  const toolbar = $("annotation-qa-bulk-toolbar");
  const available = state.annotationQaQueue === "safe_suggestions";
  toolbar.hidden = !available;
  if (!available) {
    return;
  }
  const pageIds = pageIssues.filter(issueCanBulkAcceptSamBox).map((issue) => issue.issue_id);
  const selectedOnPage = pageIds.filter((id) => state.annotationQaSelected.has(id)).length;
  $("annotation-qa-select-page").checked = pageIds.length > 0 && selectedOnPage === pageIds.length;
  $("annotation-qa-select-page").indeterminate = selectedOnPage > 0 && selectedOnPage < pageIds.length;
  $("annotation-qa-select-page").disabled = pageIds.length === 0;
  const selectedCount = state.annotationQaSelected.size;
  $("annotation-qa-selected-count").textContent = `${selectedCount} selected`;
  $("annotation-qa-bulk-accept-sam").disabled = selectedCount === 0;
}

async function bulkAcceptAnnotationQaSamBoxes() {
  const issueIds = [...state.annotationQaSelected].filter((issueId) => {
    const issue = annotationQaIssueById(issueId);
    return issue && issueCanBulkAcceptSamBox(issue);
  });
  if (!issueIds.length || !window.confirm(`Queue ${issueIds.length} policy-approved SAM suggestion${issueIds.length === 1 ? "" : "s"}?`)) {
    return;
  }
  const button = $("annotation-qa-bulk-accept-sam");
  button.disabled = true;
  button.textContent = `Queueing 0/${issueIds.length}...`;
  try {
    for (let index = 0; index < issueIds.length; index += 1) {
      await apiJson(`/api/annotation-qa/fix/${encodeURIComponent(state.annotationQaJobId)}`, {
        method: "POST",
        body: JSON.stringify({ issue_id: issueIds[index], fix: "sam_box" }),
      });
      button.textContent = `Queueing ${index + 1}/${issueIds.length}...`;
    }
    state.annotationQaSelected.clear();
    await loadAnnotationQaResults(state.annotationQaJobId);
    setMessage(`${issueIds.length} safety-gated SAM suggestion${issueIds.length === 1 ? " was" : "s were"} queued.`);
  } catch (error) {
    await loadAnnotationQaResults(state.annotationQaJobId);
    setMessage(`Bulk SAM review stopped: ${error.message}`, true);
  } finally {
    button.textContent = "Queue selected SAM suggestions";
    renderAnnotationQaBulkToolbar();
  }
}

function renderAnnotationQaIssues() {
  const container = $("annotation-qa-issues");
  const allIssues = annotationQaIssuesList();
  renderAnnotationQaQueueCounts();
  if (!allIssues.length) {
    container.innerHTML = '<p class="qa-empty">No annotation QA issues were flagged.</p>';
    $("annotation-qa-queue-summary").textContent = "The scan did not find any annotations requiring review.";
    $("annotation-qa-pagination").hidden = true;
    renderAnnotationQaBulkToolbar();
    return;
  }
  const issues = annotationQaFilteredIssues();
  if (!issues.length) {
    container.innerHTML = '<p class="qa-empty">No issues match this queue and filter combination.</p>';
    $("annotation-qa-queue-summary").textContent = "0 matching issues";
    $("annotation-qa-pagination").hidden = true;
    renderAnnotationQaBulkToolbar();
    return;
  }
  const totalPages = Math.max(1, Math.ceil(issues.length / state.annotationQaPageSize));
  state.annotationQaPage = Math.min(Math.max(1, state.annotationQaPage), totalPages);
  const start = (state.annotationQaPage - 1) * state.annotationQaPageSize;
  const pageIssues = issues.slice(start, start + state.annotationQaPageSize);
  const validSelected = new Set(issues.filter(issueCanBulkAcceptSamBox).map((issue) => issue.issue_id));
  [...state.annotationQaSelected].forEach((issueId) => {
    if (!validSelected.has(issueId)) {
      state.annotationQaSelected.delete(issueId);
    }
  });
  const groups = {
    high: pageIssues.filter((issue) => issue.severity === "high"),
    medium: pageIssues.filter((issue) => issue.severity === "medium"),
    low: pageIssues.filter((issue) => issue.severity !== "high" && issue.severity !== "medium"),
  };
  container.innerHTML = ["high", "medium", "low"]
    .filter((severity) => groups[severity].length)
    .map((severity) => annotationQaIssueTable(groups[severity], severity))
    .join("");
  $("annotation-qa-queue-summary").textContent = `Showing ${start + 1}–${Math.min(start + state.annotationQaPageSize, issues.length)} of ${issues.length} matching issues.`;
  const pagination = $("annotation-qa-pagination");
  pagination.hidden = totalPages <= 1;
  $("annotation-qa-page-label").textContent = `Page ${state.annotationQaPage} of ${totalPages}`;
  $("annotation-qa-page-prev").disabled = state.annotationQaPage <= 1;
  $("annotation-qa-page-next").disabled = state.annotationQaPage >= totalPages;
  renderAnnotationQaBulkToolbar(pageIssues);
  container.querySelectorAll("[data-qa-select]").forEach((checkbox) => {
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.annotationQaSelected.add(checkbox.dataset.qaSelect);
      } else {
        state.annotationQaSelected.delete(checkbox.dataset.qaSelect);
      }
      renderAnnotationQaBulkToolbar(pageIssues);
    });
  });
  container.onclick = (event) => {
    const trigger = event.target.closest("[data-qa-open]");
    if (!trigger || !container.contains(trigger)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const severity = trigger.dataset.qaSeverity || trigger.closest("[data-qa-severity]")?.dataset.qaSeverity || "";
    state.annotationQaReviewReturnFocus = trigger;
    openAnnotationQaReview(trigger.dataset.qaOpen, severity);
  };
}

async function loadAnnotationQaResults(jobId) {
  const report = await apiJson(`/api/annotation-qa/results/${encodeURIComponent(jobId)}`);
  state.annotationQaReport = report;
  state.annotationQaCorrectedDatasetYaml = report.summary?.corrected_dataset_yaml || state.annotationQaCorrectedDatasetYaml || "";
  renderAnnotationQaFilterOptions();
  renderAnnotationQaSummary(report.summary || {});
  renderAnnotationQaIssues();
  $("annotation-qa-config-stage").open = false;
  $("annotation-qa-run-stage").open = false;
  syncAnnotationQaActionStates();
}

function renderAnnotationQaJob(job) {
  state.annotationQaJobId = job.job_id || state.annotationQaJobId;
  const status = job.status || "queued";
  state.annotationQaRunning = status === "queued" || status === "running";
  state.annotationQaStopping = status === "stopping";
  $("annotation-qa-status").textContent = annotationQaStatusLabel(status);
  $("annotation-qa-status-heading").textContent = status === "completed"
    ? "Scan complete"
    : status === "failed"
      ? "Scan failed"
      : status === "stopped"
        ? "Scan stopped"
        : state.annotationQaStopping
          ? "Stopping scan"
          : state.annotationQaRunning
            ? "Scanning annotations"
            : "Ready to scan";
  $("annotation-qa-detail").textContent = job.detail || "";
  $("annotation-qa-detail").title = job.detail || "";
  setAnnotationQaProgress(state.annotationQaRunning || state.annotationQaStopping, job);
  renderAnnotationQaSummary(job.summary || {
    images_scanned: job.images_scanned,
    labels_checked: job.labels_checked,
    box_tolerance_percent: job.box_tolerance_percent,
    sam_max_difference_percent: job.sam_max_difference_percent,
    qa_decisions: job.summary?.qa_decisions,
    audits_pending: job.summary?.audits_pending,
    high: job.high,
    medium: job.medium,
    low: job.low,
  });
  if (state.annotationQaRunning || state.annotationQaStopping) {
    $("annotation-qa-config-stage").open = false;
    $("annotation-qa-run-stage").open = true;
  }
  syncActionStates();
}

function pollAnnotationQa(jobId) {
  state.annotationQaPollRevision += 1;
  const revision = state.annotationQaPollRevision;
  window.clearTimeout(state.annotationQaPollTimer);
  const poll = async () => {
    try {
      const job = await apiJson(`/api/annotation-qa/status/${encodeURIComponent(jobId)}`);
      if (revision !== state.annotationQaPollRevision) {
        return;
      }
      renderAnnotationQaJob(job);
      if (["queued", "running", "stopping"].includes(job.status)) {
        state.annotationQaPollTimer = window.setTimeout(poll, 900);
        return;
      }
      state.annotationQaRunning = false;
      state.annotationQaStopping = false;
      setAnnotationQaProgress(false);
      if (job.report_available) {
        await loadAnnotationQaResults(jobId);
      }
      syncActionStates();
    } catch (error) {
      if (revision !== state.annotationQaPollRevision) {
        return;
      }
      state.annotationQaRunning = false;
      state.annotationQaStopping = false;
      setAnnotationQaProgress(false);
      $("annotation-qa-status").textContent = "Failed";
      $("annotation-qa-status-heading").textContent = "Scan failed";
      $("annotation-qa-detail").textContent = error.message;
      syncActionStates();
    }
  };
  poll();
}

const ANNOTATION_QA_PRESET_DIFFERENCES = {
  lenient: { tolerance: 8, maximum: 35 },
  balanced: { tolerance: 5, maximum: 25 },
  strict: { tolerance: 3, maximum: 15 },
};

function applyAnnotationQaDifferencePreset() {
  const values = ANNOTATION_QA_PRESET_DIFFERENCES[$("annotation-qa-preset").value];
  if (!values) {
    return;
  }
  $("annotation-qa-tolerance").value = String(values.tolerance);
  $("annotation-qa-max-difference").value = String(values.maximum);
}

async function runAnnotationQa() {
  if (!state.datasetYaml || state.annotationQaRunning) {
    return;
  }
  state.annotationQaReport = null;
  state.annotationQaQueue = "needs_review";
  state.annotationQaPage = 1;
  $("annotation-qa-issues").innerHTML = "";
  $("annotation-qa-summary").innerHTML = "";
  $("annotation-qa-review-stage").hidden = true;
  $("annotation-qa-finalize-stage").hidden = true;
  try {
    const tolerance = Number($("annotation-qa-tolerance").value);
    const maximumDifference = Number($("annotation-qa-max-difference").value);
    if (!Number.isFinite(tolerance) || tolerance < 0 || tolerance > 50) {
      throw new Error("Box tolerance must be between 0% and 50%.");
    }
    if (!Number.isFinite(maximumDifference) || maximumDifference < 0 || maximumDifference > 100) {
      throw new Error("Maximum SAM difference must be between 0% and 100%.");
    }
    if (maximumDifference <= tolerance) {
      throw new Error("Maximum SAM difference must be greater than the YOLO box tolerance.");
    }
    const mode = $("annotation-qa-auto-mode").value;
    if (!["manual", "shadow", "automatic"].includes(mode)) {
      throw new Error("Choose a valid automatic-correction mode.");
    }
    const policy = {
      sam_prompt_expansion_percent: Number($("annotation-qa-prompt-expansion").value),
      sam_prompt_jitter_percent: Number($("annotation-qa-prompt-jitter").value),
      sam_stability_bbox_iou_min: Number($("annotation-qa-stability-iou").value),
      sam_stability_edge_percent_max: Number($("annotation-qa-stability-edge").value),
      sam_auto_quality_min: Number($("annotation-qa-auto-quality").value),
      sam_auto_yolo_iou_min: Number($("annotation-qa-auto-iou").value),
      sam_auto_center_shift_max: Number($("annotation-qa-auto-center").value),
      sam_auto_neighbor_iou_max: Number($("annotation-qa-auto-neighbor").value),
      auto_audit_percent: Number($("annotation-qa-audit").value),
    };
    const policyLimits = {
      sam_prompt_expansion_percent: [0, 50],
      sam_prompt_jitter_percent: [0, 20],
      sam_stability_bbox_iou_min: [0, 1],
      sam_stability_edge_percent_max: [0, 50],
      sam_auto_quality_min: [0, 1],
      sam_auto_yolo_iou_min: [0, 1],
      sam_auto_center_shift_max: [0, 1],
      sam_auto_neighbor_iou_max: [0, 1],
      auto_audit_percent: [0, 100],
    };
    Object.entries(policyLimits).forEach(([key, limits]) => {
      const value = policy[key];
      if (!Number.isFinite(value) || value < limits[0] || value > limits[1]) {
        throw new Error(`${key} must be between ${limits[0]} and ${limits[1]}.`);
      }
    });
    setMessage("Starting annotation QA...");
    const job = await apiJson("/api/annotation-qa/start", {
      method: "POST",
      body: JSON.stringify({
        dataset_yaml: state.datasetYaml,
        model: $("annotation-qa-model").value,
        scope: $("annotation-qa-scope").value,
        preset: $("annotation-qa-preset").value,
        box_tolerance_percent: tolerance,
        sam_max_difference_percent: maximumDifference,
        auto_correction_mode: mode,
        ...policy,
      }),
    });
    renderAnnotationQaJob(job);
    pollAnnotationQa(job.job_id);
  } catch (error) {
    setMessage(error.message, true);
    $("annotation-qa-status").textContent = "Failed";
    $("annotation-qa-status-heading").textContent = "Scan could not start";
    $("annotation-qa-detail").textContent = error.message;
    syncActionStates();
  }
}

async function stopAnnotationQa() {
  if (!state.annotationQaJobId || !state.annotationQaRunning) {
    return;
  }
  state.annotationQaStopping = true;
  syncActionStates();
  try {
    const job = await apiJson(`/api/annotation-qa/stop/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: "{}",
    });
    renderAnnotationQaJob(job);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function markAnnotationQaIssue(issueId, status) {
  if (!state.annotationQaJobId || !issueId) {
    return false;
  }
  try {
    await apiJson(`/api/annotation-qa/mark/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: JSON.stringify({ issue_id: issueId, status }),
    });
    if (state.annotationQaReport?.issues) {
      const issue = state.annotationQaReport.issues.find((item) => item.issue_id === issueId);
      if (issue) {
        issue.review_status = status;
        if (issue.audit_required) {
          issue.audit_status = status === "fix_accepted" ? "passed" : "failed";
        }
        if (status !== "fix_accepted") {
          issue.accepted_fix = "";
          issue.accepted_class_id = null;
          issue.accepted_class_name = "";
          issue.accepted_fix_source = "";
        }
      }
    }
    if (state.annotationQaActiveIssueId === issueId && $("qa-review-status")) {
      $("qa-review-status").value = status;
    }
    if (state.annotationQaActiveIssueId === issueId) {
      const issue = state.annotationQaReport?.issues?.find((item) => item.issue_id === issueId);
      if (issue) {
        renderAnnotationQaReview(issue);
      }
    }
    renderAnnotationQaSummary(state.annotationQaReport?.summary || {});
    renderAnnotationQaIssues();
    syncAnnotationQaActionStates();
    return true;
  } catch (error) {
    setMessage(error.message, true);
    return false;
  }
}

async function acceptAnnotationQaSamBox(issueId) {
  if (!state.annotationQaJobId || !issueId) {
    return false;
  }
  try {
    const result = await apiJson(`/api/annotation-qa/fix/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: JSON.stringify({ issue_id: issueId, fix: "sam_box" }),
    });
    if (state.annotationQaReport?.issues) {
      const issue = state.annotationQaReport.issues.find((item) => item.issue_id === issueId);
      if (issue) {
        issue.accepted_fix = result.accepted_fix || "sam_box";
        issue.accepted_fix_source = result.accepted_fix_source || issue.accepted_fix_source || "human";
        issue.audit_status = result.audit_status || issue.audit_status || "not_required";
        issue.review_status = result.review_status || "fix_accepted";
        if (state.annotationQaActiveIssueId === issueId) {
          renderAnnotationQaReview(issue);
        }
      }
    }
    setMessage("SAM box queued for this issue. Create the corrected dataset when you are ready.");
    renderAnnotationQaSummary(state.annotationQaReport?.summary || {});
    renderAnnotationQaIssues();
    syncAnnotationQaActionStates();
    return true;
  } catch (error) {
    setMessage(error.message, true);
    return false;
  }
}

async function acceptAnnotationQaClassChange(issueId) {
  if (!state.annotationQaJobId || !issueId || !$("qa-review-class-fix")) {
    return false;
  }
  const classId = Number($("qa-review-class-fix").value);
  if (!Number.isInteger(classId) || classId < 0) {
    setMessage("Choose a valid corrected class first.", true);
    return false;
  }
  try {
    const result = await apiJson(`/api/annotation-qa/fix/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: JSON.stringify({ issue_id: issueId, fix: "class", class_id: classId }),
    });
    if (state.annotationQaReport?.issues) {
      const issue = state.annotationQaReport.issues.find((item) => item.issue_id === issueId);
      if (issue) {
        issue.accepted_class_id = result.accepted_class_id;
        issue.accepted_class_name = result.accepted_class_name || "";
        issue.review_status = result.review_status || "fix_accepted";
        if (state.annotationQaActiveIssueId === issueId) {
          renderAnnotationQaReview(issue);
        }
      }
    }
    setMessage("Class change queued for this issue. Create the corrected dataset when you are ready.");
    renderAnnotationQaSummary(state.annotationQaReport?.summary || {});
    renderAnnotationQaIssues();
    syncAnnotationQaActionStates();
    return true;
  } catch (error) {
    setMessage(error.message, true);
    return false;
  }
}

async function applyAnnotationQaFixes() {
  if (!state.annotationQaJobId || state.annotationQaApplyingFixes) {
    return;
  }
  state.annotationQaApplyingFixes = true;
  syncAnnotationQaActionStates();
  setMessage("Applying accepted SAM fixes into a corrected dataset copy...");
  try {
    const result = await apiJson(`/api/annotation-qa/apply/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: "{}",
    });
    state.datasetYaml = result.dataset_yaml;
    state.annotationQaCorrectedDatasetYaml = result.dataset_yaml;
    setClassNames(result.classes || []);
    renderDatasetSummary(result.summary);
    if (state.annotationQaReport?.summary) {
      state.annotationQaReport.summary.corrected_dataset_yaml = result.dataset_yaml;
      state.annotationQaReport.summary.corrected_dataset_root = result.corrected_dataset_root;
      state.annotationQaReport.summary.applied_fixes = result.applied_fixes;
    }
    await loadAnnotationQaResults(state.annotationQaJobId);
    setMessage(`${result.message} You can train or download this corrected dataset now.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.annotationQaApplyingFixes = false;
    syncActionStates();
  }
}

async function downloadCorrectedDataset() {
  if (!state.annotationQaCorrectedDatasetYaml || state.downloads.has("corrected_dataset")) {
    return;
  }
  state.downloads.add("corrected_dataset");
  syncAnnotationQaActionStates();
  const previousDatasetYaml = state.datasetYaml;
  state.datasetYaml = state.annotationQaCorrectedDatasetYaml;
  try {
    await downloadPreparedDataset();
  } finally {
    state.datasetYaml = previousDatasetYaml || state.annotationQaCorrectedDatasetYaml;
    state.downloads.delete("corrected_dataset");
    syncAnnotationQaActionStates();
  }
}

function annotationQaIssuesList() {
  return Array.isArray(state.annotationQaReport?.issues) ? state.annotationQaReport.issues : [];
}

function annotationQaDecisionSnapshot(issue) {
  return {
    issueId: issue.issue_id,
    reviewStatus: issue.review_status || "unreviewed",
    acceptedFix: issue.accepted_fix || "",
    acceptedClassId: issue.accepted_class_id,
    acceptedClassName: issue.accepted_class_name || "",
  };
}

function hideAnnotationQaUndo() {
  window.clearTimeout(state.annotationQaUndoTimer);
  state.annotationQaUndoTimer = null;
  state.annotationQaUndo = null;
  if ($("qa-undo-toast")) {
    $("qa-undo-toast").hidden = true;
  }
}

function showAnnotationQaUndo(snapshot, message) {
  state.annotationQaUndo = snapshot;
  $("qa-undo-message").textContent = message;
  $("qa-undo-toast").hidden = false;
  window.clearTimeout(state.annotationQaUndoTimer);
  state.annotationQaUndoTimer = window.setTimeout(hideAnnotationQaUndo, 8000);
}

async function undoAnnotationQaDecision() {
  const snapshot = state.annotationQaUndo;
  if (!snapshot) {
    return;
  }
  hideAnnotationQaUndo();
  try {
    await apiJson(`/api/annotation-qa/mark/${encodeURIComponent(state.annotationQaJobId)}`, {
      method: "POST",
      body: JSON.stringify({ issue_id: snapshot.issueId, status: snapshot.reviewStatus }),
    });
    if (snapshot.acceptedFix === "sam_box") {
      await apiJson(`/api/annotation-qa/fix/${encodeURIComponent(state.annotationQaJobId)}`, {
        method: "POST",
        body: JSON.stringify({ issue_id: snapshot.issueId, fix: "sam_box" }),
      });
    }
    if (snapshot.acceptedClassId !== null && snapshot.acceptedClassId !== undefined) {
      await apiJson(`/api/annotation-qa/fix/${encodeURIComponent(state.annotationQaJobId)}`, {
        method: "POST",
        body: JSON.stringify({ issue_id: snapshot.issueId, fix: "class", class_id: snapshot.acceptedClassId }),
      });
    }
    await loadAnnotationQaResults(state.annotationQaJobId);
    openAnnotationQaReview(snapshot.issueId);
    setMessage("Annotation QA decision restored.");
  } catch (error) {
    setMessage(`Could not undo the QA decision: ${error.message}`, true);
  }
}

function annotationQaNextIssueId(issueId) {
  const issues = annotationQaReviewIssuesList();
  const index = issues.findIndex((issue) => issue.issue_id === issueId);
  const after = index >= 0 ? issues.slice(index + 1) : issues;
  const before = index > 0 ? issues.slice(0, index) : [];
  return [...after, ...before].find((issue) => !annotationQaIsResolved(issue) && issue.review_status !== "needs_fix")?.issue_id
    || after[0]?.issue_id
    || before[0]?.issue_id
    || "";
}

function advanceAnnotationQaReview(nextIssueId) {
  if (!$("qa-review-auto-advance").checked) {
    return;
  }
  if (nextIssueId && annotationQaIssueById(nextIssueId)) {
    openAnnotationQaReview(nextIssueId);
  } else {
    closeAnnotationQaReview();
    setMessage("This review queue is complete.");
  }
}

async function decideAnnotationQaStatus(status, message) {
  const issue = annotationQaIssueById(state.annotationQaActiveIssueId);
  if (!issue) {
    return;
  }
  const snapshot = annotationQaDecisionSnapshot(issue);
  const nextIssueId = annotationQaNextIssueId(issue.issue_id);
  const saved = await markAnnotationQaIssue(issue.issue_id, status);
  if (saved) {
    showAnnotationQaUndo(snapshot, `${message} Saved.`);
    advanceAnnotationQaReview(nextIssueId);
  }
}

async function decideAnnotationQaSamBox() {
  const issue = annotationQaIssueById(state.annotationQaActiveIssueId);
  if (!issue || !issueCanAcceptSamBox(issue)) {
    return;
  }
  const snapshot = annotationQaDecisionSnapshot(issue);
  const nextIssueId = annotationQaNextIssueId(issue.issue_id);
  const saved = await acceptAnnotationQaSamBox(issue.issue_id);
  if (saved) {
    showAnnotationQaUndo(snapshot, "SAM suggestion queued. Saved.");
    advanceAnnotationQaReview(nextIssueId);
  }
}

async function decideAnnotationQaClassChange() {
  const issue = annotationQaIssueById(state.annotationQaActiveIssueId);
  if (!issue) {
    return;
  }
  const snapshot = annotationQaDecisionSnapshot(issue);
  const nextIssueId = annotationQaNextIssueId(issue.issue_id);
  const saved = await acceptAnnotationQaClassChange(issue.issue_id);
  if (saved) {
    showAnnotationQaUndo(snapshot, "Class correction queued. Saved.");
    advanceAnnotationQaReview(nextIssueId);
  }
}

function annotationQaReviewIssuesList(currentIssue = null) {
  const filtered = annotationQaFilteredIssues();
  if (currentIssue && !filtered.some((issue) => issue.issue_id === currentIssue.issue_id)) {
    return [currentIssue, ...filtered];
  }
  if (filtered.length || !currentIssue) {
    return filtered;
  }
  return annotationQaIssuesList();
}

function annotationQaIssueById(issueId) {
  return annotationQaIssuesList().find((issue) => issue.issue_id === issueId) || null;
}

function bboxText(bbox) {
  return Array.isArray(bbox) && bbox.length === 4 ? bbox.map((value) => Math.round(Number(value) || 0)).join(", ") : "none";
}

function metricsText(metrics = {}) {
  const entries = Object.entries(metrics || {});
  if (!entries.length) {
    return "none";
  }
  return entries
    .map(([key, value]) => {
      if (value && typeof value === "object") {
        const nested = Object.entries(value)
          .map(([nestedKey, nestedValue]) => `${nestedKey.replace(/_/g, " ")}: ${metricText(nestedValue)}`)
          .join(", ");
        return `${key.replace(/_/g, " ")}: ${nested}`;
      }
      return `${key.replace(/_/g, " ")}: ${metricText(value)}`;
    })
    .join(" · ");
}

function bboxArea(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return 0;
  }
  const [x1, y1, x2, y2] = bbox.map((value) => Number(value) || 0);
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

function qaLevel(value, mediumAt, highAt) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "Unknown";
  }
  if (numeric >= highAt) {
    return "High";
  }
  if (numeric >= mediumAt) {
    return "Medium";
  }
  return "Low";
}

function annotationQaBoxDifferenceText(issue) {
  const originalArea = bboxArea(issue.original_bbox);
  const samArea = bboxArea(issue.sam_bbox || issue.recommended_bbox);
  if (!originalArea || !samArea) {
    return "Box difference: Unknown";
  }
  const ratio = samArea / originalArea;
  if (ratio < 0.7) {
    return "Box difference: SAM is smaller";
  }
  if (ratio > 1.3) {
    return "Box difference: SAM is larger";
  }
  return "Box difference: Similar size";
}

function annotationQaReviewSummary(issue) {
  const type = String(issue.issue_type || "");
  const summaries = {
    low_box_agreement: "YOLO and SAM boxes do not agree closely.",
    loose_box: "YOLO box may include too much background.",
    shifted_box: "SAM found the object center in a different place.",
    low_mask_coverage: "SAM mask covers only a small part of the YOLO box.",
    possibly_tight_box: "YOLO box may be too tight around the object.",
    empty_mask: "SAM could not find a usable mask inside the box.",
    low_confidence_mask: "SAM returned a mask, but its confidence is too low for an automatic replacement.",
    moderate_box_difference: "YOLO and SAM differ beyond the acceptance tolerance but remain inside the reviewable correction band.",
    large_box_disagreement: "YOLO and SAM differ beyond the configured maximum for automatic correction.",
    sam_mapping_error: "SAM output could not be safely matched to the requested label boxes.",
  };
  return summaries[type] || issue.message || "Review this annotation.";
}

function annotationQaReviewSuggestion(issue) {
  const type = String(issue.issue_type || "");
  if (type === "low_box_agreement" || type === "loose_box") {
    return "Check whether the yellow YOLO box includes too much background. Use the blue SAM box only if it fits the plant better.";
  }
  if (type === "shifted_box") {
    return "Confirm which box is centered on the correct plant. Send it to manual fix if either box targets the wrong object.";
  }
  if (type === "low_mask_coverage" || type === "empty_mask") {
    return "The SAM result is uncertain. Keep the YOLO box if it is correct, otherwise send this to manual fix.";
  }
  if (type === "low_confidence_mask") {
    return "SAM produced a low-confidence mask. Keep the YOLO box unless a human reviewer redraws the annotation.";
  }
  if (type === "possibly_tight_box") {
    return "Check whether the yellow YOLO box cuts off part of the plant.";
  }
  if (type === "moderate_box_difference") {
    return "Compare both boxes. The SAM box can be queued only when it clearly fits the intended object better.";
  }
  if (type === "large_box_disagreement") {
    return "Automatic SAM replacement is blocked because the disagreement is too large. Preserve YOLO or send the annotation for manual correction.";
  }
  if (type === "unstable_sam_prompt") {
    return "SAM changed when the prompt was expanded or shifted. Keep YOLO or review the object manually.";
  }
  if (type === "auto_keep_audit") {
    return "YOLO and SAM agree within tolerance. Confirm that the original YOLO box is correct for this audit sample.";
  }
  return "Choose the annotation to keep, or send it to manual fix when neither box is reliable.";
}

function annotationQaMetricSummaryText(issue) {
  const metrics = issue.metrics || {};
  const parts = [];
  if (metrics.bbox_iou !== null && metrics.bbox_iou !== undefined) {
    parts.push(`Overlap: ${qaLevel(metrics.bbox_iou, 0.45, 0.75)}`);
  }
  parts.push(annotationQaBoxDifferenceText(issue));
  if (metrics.center_shift !== null && metrics.center_shift !== undefined) {
    const shift = Number(metrics.center_shift);
    let label = "Unknown";
    if (Number.isFinite(shift)) {
      label = shift < 0.08 ? "Small" : shift < 0.18 ? "Medium" : "Large";
    }
    parts.push(`Center shift: ${label}`);
  }
  const edgeDifferences = metrics.edge_differences;
  if (edgeDifferences && edgeDifferences.max_percent !== undefined) {
    const maximum = Number(edgeDifferences.max_difference_percent ?? issue.sam_max_difference_percent ?? 25);
    parts.push(`Edge difference: ${Number(edgeDifferences.max_percent).toFixed(1)}% · keep at ${Number(edgeDifferences.tolerance_percent ?? 0).toFixed(1)}% · block above ${maximum.toFixed(1)}%`);
    if (Number(edgeDifferences.pixel_floor) > 0) {
      parts.push(`${Number(edgeDifferences.pixel_floor)} px minimum allowance`);
    }
  }
  if (issue.difference_band) {
    const bandLabels = {
      within_tolerance: "YOLO kept",
      reviewable: "SAM reviewable",
      large_disagreement: "manual only",
    };
    parts.push(`Decision: ${bandLabels[issue.difference_band] || issue.difference_band.replace(/_/g, " ")}`);
  }
  const qualityChecks = metrics.sam_quality_checks;
  if (qualityChecks && qualityChecks.passed === false) {
    const failed = Object.entries(qualityChecks)
      .filter(([name, passed]) => name !== "passed" && passed === false)
      .map(([name]) => name.replace(/_/g, " "));
    parts.push(`SAM quality gate: blocked${failed.length ? ` (${failed.join(", ")})` : ""}`);
  }
  if (metrics.sam_confidence !== null && metrics.sam_confidence !== undefined) {
    parts.push(`SAM confidence: ${(Number(metrics.sam_confidence) * 100).toFixed(1)}%`);
  }
  const stability = metrics.prompt_stability;
  if (stability) {
    parts.push(`Prompt stability: ${stability.passed ? "passed" : "failed"} · bbox IoU ${Number(stability.minimum_bbox_iou || 0).toFixed(2)} · edge spread ${Number(stability.max_edge_spread_percent || 0).toFixed(1)}%`);
  }
  if (issue.qa_decision) {
    parts.push(`Policy: ${annotationQaDecisionLabel(issue.qa_decision)}`);
  }
  if (Array.isArray(issue.decision_reasons) && issue.decision_reasons.length) {
    parts.push(`Reason: ${issue.decision_reasons.join(", ").replace(/_/g, " ")}`);
  }
  if (issue.audit_required) {
    parts.push(`Audit: ${issue.audit_status || "pending"}`);
  }
  return parts.join(" · ");
}

function issueCanAcceptSamBox(issue) {
  const edgeDifferences = issue?.metrics?.edge_differences || {};
  const promptStability = issue?.metrics?.prompt_stability || {};
  return issue?.auto_fix_eligible === true
    && issue?.quality_gate_passed === true
    && issue?.difference_band === "reviewable"
    && issue?.fix_type === "replace_box"
    && Array.isArray(issue.recommended_bbox)
    && issue.recommended_bbox.length === 4
    && promptStability.passed !== false
    && edgeDifferences.within_tolerance === false
    && edgeDifferences.within_max_difference === true;
}

function acceptedFixText(issue) {
  const fixes = [];
  if (issue.accepted_fix === "sam_box") {
    fixes.push("SAM box");
  }
  if (issue.accepted_class_id !== null && issue.accepted_class_id !== undefined) {
    fixes.push(`class ${issue.accepted_class_name || `ID ${issue.accepted_class_id}`}`);
  }
  return fixes.length ? fixes.join(" + ") : "none";
}

function pendingCorrectionText(issue) {
  const fixes = [];
  if (issue.accepted_fix === "sam_box") {
    fixes.push(`box ${bboxText(issue.original_bbox)} -> ${bboxText(issue.recommended_bbox)}`);
  }
  if (issue.accepted_class_id !== null && issue.accepted_class_id !== undefined) {
    const originalClass = issue.class_name || (issue.class_id !== null && issue.class_id !== undefined ? `ID ${issue.class_id}` : "unknown");
    fixes.push(`class ${originalClass} -> ${issue.accepted_class_name || `ID ${issue.accepted_class_id}`}`);
  }
  return fixes.length ? `Queued correction: ${fixes.join(" + ")}` : "No correction queued.";
}

function renderAnnotationQaPendingFix(issue) {
  const container = $("qa-review-pending-fix");
  if (!container) {
    return;
  }
  const hasFix = issue.accepted_fix === "sam_box"
    || (issue.accepted_class_id !== null && issue.accepted_class_id !== undefined);
  container.classList.toggle("is-empty", !hasFix);
  container.textContent = pendingCorrectionText(issue);
}

function renderAnnotationQaClassSelector(issue) {
  const select = $("qa-review-class-fix");
  const button = $("qa-review-accept-class");
  if (!select || !button) {
    return;
  }
  const classes = classNames();
  select.innerHTML = "";
  if (!classes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = issue.class_name || "No classes available";
    select.appendChild(option);
    select.disabled = true;
    button.disabled = true;
    return;
  }
  classes.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${name} (ID ${index})`;
    select.appendChild(option);
  });
  const selectedClassId = issue.accepted_class_id !== null && issue.accepted_class_id !== undefined
    ? Number(issue.accepted_class_id)
    : Number(issue.class_id);
  if (Number.isInteger(selectedClassId) && selectedClassId >= 0 && selectedClassId < classes.length) {
    select.value = String(selectedClassId);
  }
  select.disabled = false;
  button.disabled = !Number.isInteger(Number(select.value));
  button.textContent = issue.accepted_class_id !== null && issue.accepted_class_id !== undefined
    ? "Class Queued"
    : "Use Selected Class";
}

function loadAnnotationQaImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load preview asset: ${url}`));
    image.src = url;
  });
}

function drawAnnotationQaBox(context, bbox, color, label) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return;
  }
  const [x1, y1, x2, y2] = bbox.map((value) => Number(value) || 0);
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, context.canvas.width / 600);
  context.strokeRect(x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1));
  context.font = `700 ${Math.max(12, context.canvas.width / 80)}px sans-serif`;
  const labelWidth = context.measureText(label).width + 10;
  const labelHeight = Math.max(19, context.canvas.width / 45);
  context.fillStyle = color;
  context.fillRect(x1, Math.max(0, y1 - labelHeight), labelWidth, labelHeight);
  context.fillStyle = "#101820";
  context.fillText(label, x1 + 5, Math.max(14, y1 - 5));
  context.restore();
}

async function renderAnnotationQaCanvas(issue) {
  const rawUrl = annotationQaPreviewAssetUrl(issue, "raw_preview");
  const maskUrl = annotationQaPreviewAssetUrl(issue, "mask_preview");
  const canvas = $("qa-review-canvas");
  const fallback = $("qa-review-image");
  const controls = $("qa-overlay-controls");
  state.annotationQaCanvasRevision += 1;
  const revision = state.annotationQaCanvasRevision;
  if (!rawUrl) {
    canvas.hidden = true;
    controls.hidden = true;
    fallback.hidden = !annotationQaPreviewUrl(issue);
    return;
  }
  try {
    const [rawImage, maskImage] = await Promise.all([
      loadAnnotationQaImage(rawUrl),
      maskUrl ? loadAnnotationQaImage(maskUrl).catch(() => null) : Promise.resolve(null),
    ]);
    if (revision !== state.annotationQaCanvasRevision || state.annotationQaActiveIssueId !== issue.issue_id) {
      return;
    }
    canvas.width = rawImage.naturalWidth;
    canvas.height = rawImage.naturalHeight;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(rawImage, 0, 0, canvas.width, canvas.height);
    $("qa-overlay-mask").disabled = !maskImage;
    if (maskImage && $("qa-overlay-mask").checked) {
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;
      const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
      maskContext.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
      const pixels = maskContext.getImageData(0, 0, canvas.width, canvas.height);
      for (let index = 0; index < pixels.data.length; index += 4) {
        const visible = pixels.data[index] > 127;
        pixels.data[index] = 255;
        pixels.data[index + 1] = 220;
        pixels.data[index + 2] = 70;
        pixels.data[index + 3] = visible ? 105 : 0;
      }
      maskContext.putImageData(pixels, 0, 0);
      context.drawImage(maskCanvas, 0, 0);
    }
    if ($("qa-overlay-yolo").checked) {
      drawAnnotationQaBox(context, issue.original_bbox, "rgb(255, 210, 0)", "YOLO");
    }
    if ($("qa-overlay-sam").checked) {
      drawAnnotationQaBox(context, issue.sam_bbox, "rgb(0, 150, 255)", "SAM");
    }
    fallback.hidden = true;
    canvas.hidden = false;
    controls.hidden = false;
    setAnnotationQaZoom(state.annotationQaZoom);
  } catch (_error) {
    canvas.hidden = true;
    controls.hidden = true;
    fallback.hidden = !annotationQaPreviewUrl(issue);
  }
}

function renderAnnotationQaReview(issue) {
  const previewUrl = annotationQaPreviewUrl(issue);
  const severity = String(issue.severity || "low");
  const hasClassId = issue.class_id !== null && issue.class_id !== undefined;
  const classLabel = issue.class_name || (hasClassId ? `class_${issue.class_id}` : "unknown");
  const classDetail = hasClassId ? `${classLabel} (ID ${issue.class_id})` : classLabel;
  $("qa-review-severity").className = `qa-severity ${severity}`;
  $("qa-review-severity").textContent = severity;
  $("qa-review-title").textContent = issue.issue_type
    ? annotationQaIssueTypeLabel(issue.issue_type)
    : "Annotation QA Review";
  $("qa-review-subtitle").textContent = `${issue.image_name || "image"} · ${issue.split || "split"} · ${issue.class_name || "class"}`;
  $("qa-review-subtitle").title = $("qa-review-subtitle").textContent;
  $("qa-review-status").value = issue.review_status || "unreviewed";
  $("qa-review-issue-summary").textContent = annotationQaReviewSummary(issue);
  $("qa-review-suggestion").textContent = annotationQaReviewSuggestion(issue);

  const reviewIssues = annotationQaReviewIssuesList(issue);
  const reviewIndex = reviewIssues.findIndex((item) => item.issue_id === issue.issue_id);
  $("qa-review-position").textContent = reviewIndex >= 0
    ? `Issue ${reviewIndex + 1} of ${reviewIssues.length}`
    : `Issue review`;

  const image = $("qa-review-image");
  const empty = $("qa-review-empty");
  if (previewUrl) {
    image.src = previewUrl;
    image.hidden = false;
    empty.hidden = true;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
    empty.hidden = false;
  }
  renderAnnotationQaCanvas(issue);

  $("qa-review-details").innerHTML = `
    <div><dt>Message</dt><dd>${escapeHtml(issue.message || issue.issue_type || "")}</dd></div>
    <div><dt>Summary</dt><dd>${escapeHtml(annotationQaMetricSummaryText(issue))}</dd></div>
    <div><dt>Score</dt><dd>${metricText(issue.score)}</dd></div>
    <div><dt>Class</dt><dd>${escapeHtml(classDetail)}</dd></div>
    <div><dt>YOLO box</dt><dd>${escapeHtml(bboxText(issue.original_bbox))}</dd></div>
    <div><dt>SAM box</dt><dd>${escapeHtml(bboxText(issue.sam_bbox))}</dd></div>
    <div><dt>Recommended box</dt><dd>${escapeHtml(bboxText(issue.recommended_bbox))}</dd></div>
    <div><dt>Metrics</dt><dd>${escapeHtml(metricsText(issue.metrics))}</dd></div>
    <div><dt>Queued correction</dt><dd>${escapeHtml(acceptedFixText(issue))}</dd></div>
  `;
  renderAnnotationQaPendingFix(issue);
  renderAnnotationQaClassSelector(issue);
  $("qa-review-keep-yolo").innerHTML = issue.review_status === "accepted"
    ? "<span>1</span> Original box kept"
    : "<span>1</span> Keep original box";
  const canAcceptSam = issueCanAcceptSamBox(issue);
  $("qa-review-accept-sam").disabled = !canAcceptSam;
  const samActionText = issue.accepted_fix === "sam_box"
    ? issue.audit_required && issue.audit_status !== "passed"
      ? "Confirm audited SAM box"
      : "SAM box queued"
    : issue.difference_band === "large_disagreement"
      ? "SAM replacement blocked"
      : issue.difference_band === "reviewable" && !canAcceptSam
        ? "SAM quality checks failed"
        : "Use SAM suggestion";
  $("qa-review-accept-sam").innerHTML = `<span>2</span> ${escapeHtml(samActionText)}`;

  $("qa-review-prev").disabled = reviewIndex <= 0;
  $("qa-review-next").disabled = reviewIssues.length <= 1;
}

function openAnnotationQaReview(issueId, severity = "") {
  if (!$("qa-review-modal")) {
    setMessage("Refresh the page to load the annotation review UI.", true);
    return;
  }
  const issue = annotationQaIssueById(issueId);
  if (!issue) {
    return;
  }
  state.annotationQaActiveIssueId = issueId;
  state.annotationQaReviewSeverity = ["high", "medium", "low"].includes(severity)
    ? severity
    : state.annotationQaReviewSeverity || annotationQaSeverityKey(issue);
  renderAnnotationQaReview(issue);
  setAnnotationQaZoom(1);
  $("qa-review-modal").hidden = false;
  $("qa-review-modal").querySelector(".qa-review-dialog")?.focus();
}

function closeAnnotationQaReview() {
  state.annotationQaActiveIssueId = "";
  state.annotationQaReviewSeverity = "";
  $("qa-review-modal").hidden = true;
  $("qa-review-image").removeAttribute("src");
  $("qa-review-canvas").hidden = true;
  $("qa-overlay-controls").hidden = true;
  state.annotationQaCanvasRevision += 1;
  if (state.annotationQaReviewReturnFocus?.isConnected) {
    state.annotationQaReviewReturnFocus.focus();
  } else {
    document.querySelector(`[data-qa-queue="${CSS.escape(state.annotationQaQueue)}"]`)?.focus();
  }
  state.annotationQaReviewReturnFocus = null;
}

function stepAnnotationQaReview(direction) {
  const currentIssue = annotationQaIssueById(state.annotationQaActiveIssueId);
  const issues = annotationQaReviewIssuesList(currentIssue);
  const index = issues.findIndex((issue) => issue.issue_id === state.annotationQaActiveIssueId);
  let next = null;
  if (direction > 0) {
    const ordered = [...issues.slice(index + 1), ...issues.slice(0, Math.max(0, index))];
    next = ordered.find((issue) => !annotationQaIsResolved(issue) && issue.review_status !== "needs_fix") || ordered[0];
  } else {
    next = issues[index - 1];
  }
  if (next) {
    openAnnotationQaReview(next.issue_id, state.annotationQaReviewSeverity || annotationQaSeverityKey(next));
  }
}

function setAnnotationQaZoom(value) {
  state.annotationQaZoom = Math.max(0.5, Math.min(3, Number(value) || 1));
  if ($("qa-review-image")) {
    $("qa-review-image").style.transform = `scale(${state.annotationQaZoom})`;
  }
  if ($("qa-review-canvas")) {
    $("qa-review-canvas").style.transform = `scale(${state.annotationQaZoom})`;
  }
  if ($("qa-review-zoom-reset")) {
    $("qa-review-zoom-reset").textContent = state.annotationQaZoom === 1 ? "Fit" : `${Math.round(state.annotationQaZoom * 100)}%`;
  }
}

function handleAnnotationQaReviewKeydown(event) {
  if (!$("qa-review-modal") || $("qa-review-modal").hidden || !state.annotationQaActiveIssueId) {
    return;
  }
  if (event.key === "Tab") {
    const dialog = $("qa-review-modal").querySelector(".qa-review-dialog");
    const focusable = [...dialog.querySelectorAll("button:not([disabled]), select:not([disabled]), input:not([disabled]), summary, [tabindex='0']")]
      .filter((element) => element.offsetParent !== null);
    if (focusable.length) {
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    return;
  }
  const key = event.key.toLowerCase();
  const target = event.target;
  if (target instanceof HTMLElement && (target.matches("input, textarea, select") || target.isContentEditable)) {
    return;
  }
  if (key === "1") {
    event.preventDefault();
    decideAnnotationQaStatus("accepted", "Original YOLO box kept.");
  } else if (key === "2") {
    if (!$("qa-review-accept-sam").disabled) {
      event.preventDefault();
      decideAnnotationQaSamBox();
    }
  } else if (key === "3") {
    event.preventDefault();
    decideAnnotationQaStatus("needs_fix", "Annotation flagged for manual correction.");
  } else if (key === "a" || event.key === "ArrowLeft") {
    if (!$("qa-review-prev").disabled) {
      event.preventDefault();
      stepAnnotationQaReview(-1);
    }
  } else if (key === "d" || event.key === "ArrowRight") {
    if (!$("qa-review-next").disabled) {
      event.preventDefault();
      stepAnnotationQaReview(1);
    }
  }
}

function setActiveAnnotationQaReviewStatus(status) {
  if (!state.annotationQaActiveIssueId) {
    return;
  }
  decideAnnotationQaStatus(status, `${annotationQaReviewStatusLabel(status)} selected.`);
}

function downloadAnnotationQaReport(kind) {
  if (!state.annotationQaJobId) {
    return;
  }
  window.location.href = `/api/annotation-qa/download/${encodeURIComponent(state.annotationQaJobId)}/${kind}`;
}

const DEFAULT_METRIC_LABELS = {
  precision: "Precision",
  recall: "Recall",
  map50: "mAP50",
  map50_95: "mAP50-95",
};

const MAGIC_OVERALL_OPTIONS = [
  { scope: "overall", key: "map50_95", label: "mAP50-95" },
  { scope: "overall", key: "map50", label: "mAP50" },
  { scope: "overall", key: "precision", label: "Precision" },
  { scope: "overall", key: "recall", label: "Recall" },
  { scope: "overall", key: "macro_f1", label: "Macro F1" },
  { scope: "overall", key: "weighted_f1", label: "Weighted F1" },
];

const MAGIC_PER_CLASS_OPTIONS = [
  { scope: "per_class", key: "map50_95", label: "AP50-95" },
  { scope: "per_class", key: "map50", label: "AP50" },
  { scope: "per_class", key: "precision", label: "Precision" },
  { scope: "per_class", key: "recall", label: "Recall" },
  { scope: "per_class", key: "f1", label: "F1" },
];

function metricLabels(labels = {}) {
  return { ...DEFAULT_METRIC_LABELS, ...(labels || {}) };
}

function applyMetricLabels(labels = {}, chartTitle = "Detection Performance by Epoch") {
  const merged = metricLabels(labels);
  state.metricLabels = merged;
  state.performanceChartTitle = chartTitle;
  $("metric-precision-label").textContent = merged.precision;
  $("metric-recall-label").textContent = merged.recall;
  $("metric-map50-label").textContent = merged.map50;
  $("metric-map-label").textContent = merged.map50_95;
  $("performance-chart-title").textContent = chartTitle;
}

function formatBestMetric(row, key, label) {
  if (!row || row[key] === null || row[key] === undefined) {
    return "";
  }
  return `<div><span>${label}</span><strong>${metricText(row[key])}</strong><small>Epoch ${row.epoch}</small></div>`;
}

function renderBestMetrics(best, history = [], labels = {}) {
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
  const mergedLabels = metricLabels(labels);
  const rows = [
    formatBestMetric(summary.best_map50_95, "map50_95", `Best ${mergedLabels.map50_95}`),
    formatBestMetric(summary.best_map50, "map50", `Best ${mergedLabels.map50}`),
    formatBestMetric(summary.lowest_training_loss, "training_loss", "Lowest train loss"),
    formatBestMetric(summary.lowest_validation_loss, "testing_loss", "Lowest val loss"),
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
  $("magic-metrics").disabled = !state.metricsAvailable
    || state.running
    || state.downloads.has("magic_metrics");
}

function setMagicAdjustedState(metrics = null) {
  const adjusted = Boolean(metrics?.magic_adjusted);
  const count = Array.isArray(metrics?.magic_adjustments) ? metrics.magic_adjustments.length : (adjusted ? 1 : 0);
  $("training-results-panel").classList.toggle("has-magic-metrics", adjusted);
  $("magic-metrics").textContent = adjusted && count
    ? `Magic Button (${count})`
    : "Magic Button";
}

function artifactViewUrl(artifact, target, status) {
  const params = new URLSearchParams({
    project: target.project,
    name: target.name,
  });
  params.set("v", String(status.modified_at || status.size || 0));
  return `/api/train/artifacts/view/${encodeURIComponent(artifact)}?${params.toString()}`;
}

function renderConfusionMatrices(artifacts = {}, target = weightTarget(), metrics = {}) {
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
    : metrics.backend === "rfdetr"
      ? "RF-DETR training does not generate validation confusion matrices in this runner."
      : metrics.backend === "dfine"
        ? "D-FINE training does not generate validation confusion matrices in this runner yet."
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
    container.innerHTML = rocAuc.note
      ? `<p>${escapeHtml(rocAuc.note)}</p>`
      : "";
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

function chartTooltip() {
  let tooltip = $("chart-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "chart-tooltip";
    tooltip.className = "chart-tooltip";
    tooltip.setAttribute("role", "status");
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function hideChartTooltip() {
  const tooltip = $("chart-tooltip");
  if (tooltip) {
    tooltip.classList.remove("is-visible");
  }
  document.querySelectorAll(".chart-panel canvas").forEach((canvas) => {
    canvas.classList.remove("has-hover-point");
  });
}

function metricTooltipValue(value) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(3);
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function showChartTooltip(canvas, point, clientX, clientY) {
  const tooltip = chartTooltip();
  tooltip.innerHTML = `
    <strong>Epoch ${point.epoch}</strong>
    <span>${escapeHtml(point.label)}: ${metricTooltipValue(point.value)}</span>
  `;
  tooltip.classList.add("is-visible");

  const tooltipRect = tooltip.getBoundingClientRect();
  const offset = 12;
  let left = clientX + offset;
  let top = clientY - tooltipRect.height - offset;
  if (left + tooltipRect.width > window.innerWidth - 8) {
    left = clientX - tooltipRect.width - offset;
  }
  if (top < 8) {
    top = clientY + offset;
  }
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.min(window.innerHeight - tooltipRect.height - 8, top)}px`;

  canvas.classList.add("has-hover-point");
}

function nearestChartPoint(canvas, event) {
  const points = canvas._chartPoints || [];
  if (!points.length) {
    return null;
  }

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hitRadius = 10;
  let nearest = null;
  points.forEach((point) => {
    const distance = Math.hypot(point.x - x, point.y - y);
    if (distance <= hitRadius && (!nearest || distance < nearest.distance)) {
      nearest = { ...point, distance };
    }
  });
  return nearest;
}

function handleChartPointerMove(event) {
  const canvas = event.currentTarget;
  const point = nearestChartPoint(canvas, event);
  if (!point) {
    hideChartTooltip();
    return;
  }
  showChartTooltip(canvas, point, event.clientX, event.clientY);
}

function handleChartFocus(event) {
  const canvas = event.currentTarget;
  const point = (canvas._chartPoints || [])[0];
  if (!point) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  showChartTooltip(canvas, point, rect.left + point.x, rect.top + point.y);
}

function initializeChartTooltips() {
  document.querySelectorAll(".chart-panel canvas").forEach((canvas) => {
    canvas.tabIndex = 0;
    canvas.addEventListener("mousemove", handleChartPointerMove);
    canvas.addEventListener("mouseleave", hideChartTooltip);
    canvas.addEventListener("focus", handleChartFocus);
    canvas.addEventListener("blur", hideChartTooltip);
  });
  window.addEventListener("scroll", hideChartTooltip, { passive: true });
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
  canvas._chartPoints = [];
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
      const x = xFor(point.epoch);
      const y = yFor(point.value);
      canvas._chartPoints.push({
        x,
        y,
        epoch: point.epoch,
        value: point.value,
        label: line.label,
      });
      context.fillStyle = line.color;
      context.beginPath();
      context.arc(x, y, 2.5, 0, Math.PI * 2);
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

function renderMetricCharts(history, labels = {}) {
  const rows = Array.isArray(history) ? history : [];
  state.metricsHistory = rows;
  const mergedLabels = metricLabels(labels);
  drawLineChart("accuracy-chart", rows, [
    { key: "map50", label: mergedLabels.map50, color: "#16745f" },
    { key: "map50_95", label: mergedLabels.map50_95, color: "#5b6ee1" },
  ]);
  drawLineChart("loss-chart", rows, [
    { key: "training_loss", label: "Train loss", color: "#a43d3d" },
    { key: "testing_loss", label: "Val loss", color: "#16745f" },
  ]);
}

function redrawCharts() {
  renderMetricCharts(state.metricsHistory, state.metricLabels);
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

function responseDownloadFilename(response, fallback) {
  const disposition = response.headers.get("Content-Disposition") || "";
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match) {
    try {
      return decodeURIComponent(utf8Match[1].trim());
    } catch (_error) {
      return utf8Match[1].trim();
    }
  }
  const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
  return filenameMatch ? filenameMatch[1].trim() : fallback;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(errorDetailText(payload, `Request failed: ${response.status}`));
  }
  return payload;
}

function errorDetailText(payload, fallback) {
  const detail = payload?.detail || payload?.message;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== "object") {
        return String(item);
      }
      const location = Array.isArray(item.loc) ? item.loc.join(".") : item.loc;
      return [location, item.msg].filter(Boolean).join(": ");
    }).join("; ") || fallback;
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return detail || fallback;
}

function setTrainingSessionStatus(text, isError = false) {
  const message = $("training-session-status");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function renderTrainingSessions(sessions) {
  const select = $("training-session-select");
  const currentValue = select.value;
  const currentTarget = {
    project: $("project").value.trim() || "runs/detect",
    name: $("run-name").value.trim() || "train",
  };
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Choose a training run";
  select.appendChild(placeholder);

  sessions.forEach((session) => {
    const option = document.createElement("option");
    option.value = JSON.stringify({ project: session.project, name: session.name, task: session.task });
    option.textContent = session.label;
    select.appendChild(option);
  });

  if (currentValue && Array.from(select.options).some((option) => option.value === currentValue)) {
    select.value = currentValue;
  } else {
    const currentOption = Array.from(select.options).find((option) => {
      if (!option.value) {
        return false;
      }
      try {
        const value = JSON.parse(option.value);
        return value.project === currentTarget.project && value.name === currentTarget.name;
      } catch (error) {
        return false;
      }
    });
    if (currentOption) {
      select.value = currentOption.value;
    }
  }
  setTrainingSessionStatus(
    sessions.length
      ? `${sessions.length} previous training session${sessions.length === 1 ? "" : "s"} found.`
      : "No completed training sessions were found.",
    false,
  );
}

async function loadTrainingSessions() {
  setTrainingSessionStatus("Loading training runs...");
  const payload = await apiJson("/api/train/sessions");
  state.trainingSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
  renderTrainingSessions(state.trainingSessions);
  if (isDefaultTrainingTarget() && state.trainingSessions.length) {
    const latest = state.trainingSessions[0];
    $("training-session-select").value = JSON.stringify({
      project: latest.project,
      name: latest.name,
      task: latest.task,
    });
    loadSelectedTrainingSession();
  }
}

function trainingSessionErrorMessage(error) {
  const message = error?.message || "Previous training sessions could not be loaded.";
  if (message.includes("404")) {
    return "Previous training sessions endpoint is unavailable. Restart the web app so /api/train/sessions is registered.";
  }
  return message;
}

function loadSelectedTrainingSession() {
  const raw = $("training-session-select").value;
  if (!raw) {
    setTrainingSessionStatus("Choose a previous training session first.", true);
    return;
  }
  let session;
  try {
    session = JSON.parse(raw);
  } catch (error) {
    setTrainingSessionStatus("The selected training session could not be read.", true);
    return;
  }
  if (session.task) {
    $("model-size").value = modelSizeForTask(session.task, $("model-size").value);
  }
  $("project").value = session.project || defaultProjectForModelSize($("model-size").value);
  $("run-name").value = session.name || "train";
  updateCurrentRunDisplay();
  setPanelExpanded("results", true);
  scheduleTargetRefresh();
  setTrainingSessionStatus(`Loaded ${session.project}/${session.name}.`, false);
  setMessage(`Loaded previous training session ${session.project}/${session.name}.`);
}

function setAppTab(tab) {
  document.querySelectorAll("[data-app-tab]").forEach((button) => {
    const active = button.dataset.appTab === tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".app-view").forEach((view) => {
    const active = tab === "annotation-qa"
      ? view.id === "training-view"
      : view.id === `${tab}-view`;
    view.classList.toggle("active", active);
    view.hidden = !active;
  });
  const trainingView = $("training-view");
  trainingView.classList.toggle("qa-mode", tab === "annotation-qa");
  trainingView.setAttribute("aria-labelledby", tab === "annotation-qa" ? "app-tab-annotation-qa" : "app-tab-training");
  if (tab === "inference") {
    loadInferenceWeights().catch((error) => setInferenceMessage(error.message, true));
  } else if (tab === "annotation-qa") {
    setPanelExpanded("annotation-qa", true);
  } else {
    redrawChartsSoon();
  }
}

function setInferenceMessage(text, isError = false) {
  const message = $("inference-message");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setInferenceUploadProgress(active, percent = 0, detail = "Waiting to upload.") {
  const shell = $("inference-upload-status");
  const safePercent = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  shell.hidden = !active;
  $("inference-upload-percent").textContent = `${safePercent}%`;
  $("inference-upload-detail").textContent = detail;
  $("inference-upload-track").setAttribute("aria-valuenow", String(safePercent));
  $("inference-upload-track").setAttribute("aria-valuetext", `Upload progress ${safePercent}%`);
  $("inference-upload-fill").style.width = `${safePercent}%`;
}

function selectedInferenceWeightSource() {
  return document.querySelector('input[name="inference-weight-source"]:checked')?.value || "selected";
}

function inferenceWeightSuffix(file) {
  const name = file?.name || "";
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

function syncInferenceControls() {
  const source = selectedInferenceWeightSource();
  $("inference-selected-weight-wrap").hidden = source !== "selected";
  $("inference-upload-weight-wrap").hidden = source !== "upload";
  const hasSelectedWeight = $("inference-weight-select").value !== "";
  const uploadedWeight = $("inference-weight-file").files[0];
  const uploadedSuffix = inferenceWeightSuffix(uploadedWeight);
  const uploadFamily = $("inference-upload-family").value;
  const isDetrUpload = uploadFamily === "rfdetr" || uploadFamily === "dfine";
  const hasUploadedWeight = isDetrUpload
    ? uploadedSuffix === ".pt"
    : uploadedSuffix === ".pt" || uploadedSuffix === ".onnx";
  $("inference-convert-onnx-row").hidden = source !== "upload" || uploadedSuffix !== ".pt" || isDetrUpload;
  $("inference-convert-onnx").disabled = source !== "upload" || uploadedSuffix !== ".pt" || isDetrUpload;
  if (uploadedSuffix !== ".pt" || isDetrUpload) {
    $("inference-convert-onnx").checked = false;
  }
  const hasMedia = Boolean($("inference-media-file").files[0]);
  $("run-inference").disabled = state.inferenceRunning
    || !hasMedia
    || (source === "selected" ? !hasSelectedWeight : !hasUploadedWeight);
  $("stop-inference").disabled = !state.inferenceRunning
    || state.inferenceStopping
    || !state.inferenceJobId
    || !state.inferenceStoppable;
  $("stop-inference").textContent = state.inferenceStopping ? "Stopping..." : "Stop";
  $("stop-inference").setAttribute("aria-busy", String(state.inferenceStopping));
  $("clear-inference-output").disabled = state.inferenceRunning || state.inferenceClearing;
  $("clear-inference-output").textContent = state.inferenceClearing ? "Clearing..." : "Clear Inference Output";
  $("clear-inference-output").setAttribute("aria-busy", String(state.inferenceClearing));
  const clearingInferenceUploads = state.storageCleanup.has("inference_uploads");
  $("clear-inference-uploads").disabled = state.inferenceRunning || clearingInferenceUploads;
  $("clear-inference-uploads").textContent = clearingInferenceUploads ? "Clearing..." : "Clear Uploaded Weights";
  $("clear-inference-uploads").setAttribute("aria-busy", String(clearingInferenceUploads));
}

function renderInferenceWeights(weights) {
  const select = $("inference-weight-select");
  select.innerHTML = "";
  if (!weights.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No weights found";
    select.appendChild(option);
    $("inference-weights-status").textContent = "No .pt or .onnx weights were found under runs/detect, runs/rfdetr, runs/dfine, runs/segment, runs/semantic, or runs/classify.";
    syncInferenceControls();
    return;
  }
  weights.forEach((weight) => {
    const option = document.createElement("option");
    option.value = weight.path;
    const format = weight.format ? weight.format.toUpperCase() : "PT";
    option.textContent = `${weight.label} [${format}] (${formatBytes(weight.size)})`;
    select.appendChild(option);
  });
  $("inference-weights-status").textContent = `${weights.length} .pt/.onnx weight file${weights.length === 1 ? "" : "s"} available from runs.`;
  syncInferenceControls();
}

async function loadInferenceWeights() {
  $("inference-weights-status").textContent = "Loading weights from runs...";
  const payload = await apiJson("/api/inference/weights");
  state.inferenceWeights = Array.isArray(payload.weights) ? payload.weights : [];
  renderInferenceWeights(state.inferenceWeights);
}

function updateInferenceFileSelection() {
  const weight = $("inference-weight-file").files[0];
  const weightSuffix = inferenceWeightSuffix(weight);
  $("inference-weight-selection").textContent = weight
    ? `${weight.name} (${formatBytes(weight.size)})${[".pt", ".onnx"].includes(weightSuffix) ? "" : " - unsupported"}`
    : "No weights file selected.";

  const media = $("inference-media-file").files[0];
  $("inference-media-selection").textContent = media
    ? `${media.name} (${formatBytes(media.size)})`
    : "No image or video selected.";
  renderInferenceInputPreview(media);
  if (!media) {
    setInferenceUploadProgress(false);
  }
  syncInferenceControls();
}

function renderInferenceInputPreview(media) {
  const shell = $("inference-input-preview-shell");
  const image = $("inference-input-image");
  const video = $("inference-input-video");
  const status = $("inference-input-preview-status");
  if (state.inferenceInputObjectUrl) {
    URL.revokeObjectURL(state.inferenceInputObjectUrl);
    state.inferenceInputObjectUrl = "";
  }
  if (!media) {
    shell.hidden = true;
    image.hidden = true;
    image.removeAttribute("src");
    video.hidden = true;
    video.pause();
    video.onerror = null;
    video.removeAttribute("src");
    $("inference-input-summary").textContent = "No uploaded media selected yet.";
    status.textContent = "";
    return;
  }

  const objectUrl = URL.createObjectURL(media);
  state.inferenceInputObjectUrl = objectUrl;
  shell.hidden = false;
  $("inference-input-summary").textContent = `${media.name} (${formatBytes(media.size)})`;
  status.textContent = "";
  if ((media.type || "").startsWith("video/") || /\.(mov|mp4|avi|mkv|webm)$/i.test(media.name)) {
    image.hidden = true;
    image.removeAttribute("src");
    video.hidden = false;
    video.src = objectUrl;
    video.onerror = () => {
      status.textContent = "This video could not be previewed in the browser, but it can still be uploaded for inference.";
    };
    if (/\.mov$/i.test(media.name)) {
      status.textContent = "MOV preview may appear black if the browser cannot decode the camera codec. The server will still process it.";
    }
    video.load();
  } else {
    video.hidden = true;
    video.pause();
    video.onerror = null;
    video.removeAttribute("src");
    image.hidden = false;
    image.src = objectUrl;
  }
}

function inferenceTaskLabel(task) {
  const labels = {
    detect: "Detection",
    segment: "Segmentation",
    semantic: "Semantic Segmentation",
    classify: "Classification",
    pose: "Pose",
    obb: "Oriented Detection",
  };
  return labels[task] || "YOLO";
}

function inferenceResultTask(result) {
  if (result.task) {
    return result.task;
  }
  if (Array.isArray(result.image_classifications) && result.image_classifications.length) {
    return "classify";
  }
  const detections = Array.isArray(result.image_detections) ? result.image_detections : [];
  if (detections.some((row) => row.mask_area !== undefined && row.mask_area !== null)) {
    return "segment";
  }
  return "detect";
}

function renderInferencePredictions(result) {
  const container = $("inference-detections");
  const task = inferenceResultTask(result || {});
  if (task === "classify") {
    const rows = Array.isArray(result.image_classifications) ? result.image_classifications : [];
    if (!rows.length) {
      container.innerHTML = "<p>No image classifications were returned.</p>";
      return;
    }
    container.innerHTML = `
      <h4>Image Classifications</h4>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Class</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${Math.round(Number(row.rank) || 0)}</td>
              <td>${escapeHtml(row.class_name || `class_${row.class_id}`)}</td>
              <td>${metricText(row.confidence)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
    return;
  }

  const rows = Array.isArray(result.image_detections) ? result.image_detections : [];
  if (!rows.length) {
    container.innerHTML = "<p>No image predictions passed the selected confidence threshold.</p>";
    return;
  }
  const hasMasks = rows.some((row) => row.mask_area !== undefined && row.mask_area !== null);
  container.innerHTML = `
    <h4>${task === "segment" || hasMasks ? "Image Instances" : "Image Detections"}</h4>
    <table>
      <thead>
        <tr>
          <th>Class</th>
          <th>Confidence</th>
          <th>Box</th>
          ${hasMasks ? "<th>Mask Area</th>" : ""}
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(row.class_name || `class_${row.class_id}`)}</td>
            <td>${metricText(row.confidence)}</td>
            <td>${(row.box || []).map((value) => Math.round(Number(value) || 0)).join(", ")}</td>
            ${hasMasks ? `<td>${row.mask_area === null || row.mask_area === undefined ? "N/A" : Math.round(Number(row.mask_area) || 0)}</td>` : ""}
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function inferenceStageLabel(stage) {
  const labels = {
    queued: "Queued",
    starting: "Starting",
    converting: "Converting weights",
    loading_model: "Loading model",
    processing: "Processing frames",
    encoding: "Encoding video",
    complete: "Complete",
    stopped: "Stopped",
    failed: "Failed",
  };
  return labels[stage] || String(stage || "Processing");
}

function resetInferenceResultForRun() {
  $("inference-results").hidden = true;
  $("download-inference-result").removeAttribute("href");
  $("inference-result-summary").textContent = "No inference result yet.";
  $("inference-result-image-shell").hidden = true;
  $("inference-result-video-shell").hidden = true;
  $("inference-result-image").hidden = true;
  $("inference-result-image").removeAttribute("src");
  $("inference-result-video").hidden = true;
  $("inference-result-video").pause();
  $("inference-result-video").removeAttribute("src");
  $("inference-detections").innerHTML = "";
  stopInferenceWebRtc();
  $("inference-live-preview-shell").hidden = true;
  $("inference-live-preview-image").removeAttribute("src");
  $("inference-live-preview-image").hidden = false;
  state.inferencePreviewStreamJobId = "";
  state.inferenceWebRtcFailedJobId = "";
  $("inference-live-preview-summary").textContent = "Waiting for annotated frames.";
}

function stopInferencePolling() {
  state.inferencePollRevision += 1;
  window.clearTimeout(state.inferencePollTimer);
  state.inferencePollTimer = null;
}

function stopInferenceWebRtc() {
  const video = $("inference-live-preview-video");
  if (state.inferenceWebRtcPeer) {
    state.inferenceWebRtcPeer.close();
    state.inferenceWebRtcPeer = null;
  }
  state.inferenceWebRtcJobId = "";
  state.inferenceWebRtcStartingJobId = "";
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
  }
  video.pause();
  video.srcObject = null;
  video.hidden = true;
}

function waitForIceGatheringComplete(peer) {
  if (peer.iceGatheringState === "complete") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", checkState);
      resolve();
    }, 1500);
    const checkState = () => {
      if (peer.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        peer.removeEventListener("icegatheringstatechange", checkState);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", checkState);
  });
}

async function startInferenceWebRtc(jobId) {
  if (!window.RTCPeerConnection) {
    throw new Error("WebRTC is not supported by this browser.");
  }
  stopInferenceWebRtc();
  state.inferenceWebRtcStartingJobId = jobId;
  const peer = new RTCPeerConnection();
  state.inferenceWebRtcPeer = peer;
  const video = $("inference-live-preview-video");
  peer.addTransceiver("video", { direction: "recvonly" });
  peer.addEventListener("track", (event) => {
    if (event.track.kind !== "video") {
      return;
    }
    video.srcObject = event.streams[0] || new MediaStream([event.track]);
    video.hidden = false;
    $("inference-live-preview-image").hidden = true;
    video.play().catch(() => {});
  });
  peer.addEventListener("connectionstatechange", () => {
    if (["failed", "disconnected", "closed"].includes(peer.connectionState)) {
      if (state.inferenceWebRtcPeer === peer) {
        state.inferenceWebRtcPeer = null;
        state.inferenceWebRtcJobId = "";
      }
    }
  });

  const offer = await peer.createOffer();
  await peer.setLocalDescription(offer);
  await waitForIceGatheringComplete(peer);
  const answer = await apiJson(`/api/inference/webrtc/${encodeURIComponent(jobId)}/offer`, {
    method: "POST",
    body: JSON.stringify({
      sdp: peer.localDescription.sdp,
      type: peer.localDescription.type,
    }),
  });
  await peer.setRemoteDescription(answer);
  state.inferenceWebRtcJobId = jobId;
  state.inferenceWebRtcStartingJobId = "";
}

function renderInferenceLivePreview(job) {
  if (!$("inference-live-preview-enabled").checked) {
    $("inference-live-preview-shell").hidden = true;
    stopInferenceWebRtc();
    $("inference-live-preview-image").removeAttribute("src");
    $("inference-live-preview-image").hidden = false;
    state.inferencePreviewStreamJobId = "";
    return;
  }
  if (["complete", "failed", "stopped"].includes(job.status)) {
    return;
  }
  if (!job.preview_available || !job.job_id) {
    return;
  }
  $("inference-live-preview-shell").hidden = false;
  const canStartWebRtc = state.inferenceWebRtcFailedJobId !== job.job_id
    && state.inferenceWebRtcStartingJobId !== job.job_id
    && state.inferenceWebRtcJobId !== job.job_id
    && !state.inferenceWebRtcPeer;
  if (canStartWebRtc) {
    startInferenceWebRtc(job.job_id).catch(() => {
      stopInferenceWebRtc();
      state.inferenceWebRtcFailedJobId = job.job_id;
      $("inference-live-preview-image").hidden = false;
      if (state.inferencePreviewStreamJobId !== job.job_id) {
        $("inference-live-preview-image").src = `/api/inference/stream/${encodeURIComponent(job.job_id)}?t=${Date.now()}`;
        state.inferencePreviewStreamJobId = job.job_id;
      }
    });
  } else if (!state.inferenceWebRtcPeer && state.inferencePreviewStreamJobId !== job.job_id) {
    $("inference-live-preview-image").hidden = false;
    $("inference-live-preview-image").src = `/api/inference/stream/${encodeURIComponent(job.job_id)}?t=${Date.now()}`;
    state.inferencePreviewStreamJobId = job.job_id;
  }
  const total = Number(job.total_frames) || 0;
  const frames = Number(job.frames) || 0;
  const fps = Number(job.preview_fps) || 0;
  const fpsText = fps > 0 ? ` Live preview: ${fps.toFixed(1)} FPS.` : " Live preview: measuring FPS.";
  $("inference-live-preview-summary").textContent = total
    ? `Latest annotated frame while processing ${frames} of ${total}.${fpsText}`
    : `Latest annotated frame while processing ${frames} frames.${fpsText}`;
}

function updateInferenceJobProgress(job) {
  const stage = inferenceStageLabel(job.stage);
  const percent = Number(job.percent) || 0;
  const total = Number(job.total_frames) || 0;
  const frames = Number(job.frames) || 0;
  const frameText = total ? ` ${frames} / ${total} frames.` : frames ? ` ${frames} frames.` : "";
  setInferenceUploadProgress(
    true,
    percent,
    `${stage}: ${job.detail || "Inference is running."}${frameText}`,
  );
  renderInferenceLivePreview(job);
}

function pollInferenceJob(jobId, revision) {
  const poll = async () => {
    if (revision !== state.inferencePollRevision) {
      return;
    }
    try {
      const job = await apiJson(`/api/inference/status/${encodeURIComponent(jobId)}`);
      if (revision !== state.inferencePollRevision) {
        return;
      }
      updateInferenceJobProgress(job);
      state.inferenceStoppable = Boolean(job.stoppable);
      syncInferenceControls();
      if (job.status === "complete") {
        state.inferenceRunning = false;
        state.inferenceStopping = false;
        state.inferenceStoppable = false;
        stopInferenceWebRtc();
        $("run-inference").textContent = "Run Inference";
        setInferenceMessage("Inference complete.");
        renderInferenceResult(job.result || {});
        syncInferenceControls();
        return;
      }
      if (job.status === "failed") {
        state.inferenceRunning = false;
        state.inferenceStopping = false;
        state.inferenceStoppable = false;
        stopInferenceWebRtc();
        $("run-inference").textContent = "Run Inference";
        setInferenceMessage(job.error || job.detail || "Inference failed.", true);
        syncInferenceControls();
        return;
      }
      if (job.status === "stopped") {
        state.inferenceRunning = false;
        state.inferenceStopping = false;
        state.inferenceStoppable = false;
        stopInferenceWebRtc();
        $("run-inference").textContent = "Run Inference";
        setInferenceMessage(job.detail || "Inference stopped.");
        updateInferenceJobProgress(job);
        syncInferenceControls();
        return;
      }
      state.inferencePollTimer = window.setTimeout(poll, 750);
    } catch (error) {
      if (revision !== state.inferencePollRevision) {
        return;
      }
      if ((error.message || "").includes("404")) {
        state.inferenceRunning = false;
        $("run-inference").textContent = "Run Inference";
        setInferenceMessage("Inference job was not found. Restart the web app and run inference again.", true);
        syncInferenceControls();
        return;
      }
      state.inferencePollTimer = window.setTimeout(poll, 1200);
    }
  };
  poll();
}

function renderInferenceResult(result) {
  if (!result || !result.result_url) {
    setInferenceMessage("Inference completed, but the result payload was missing.", true);
    return;
  }
  const cacheBust = `t=${Date.now()}`;
  const resultUrl = `${result.result_url}?${cacheBust}`;
  $("inference-results").hidden = false;
  $("download-inference-result").href = result.download_url || result.result_url;
  const task = inferenceResultTask(result);
  const taskLabel = inferenceTaskLabel(task);
  const frames = Number(result.frames) || 0;
  const videoDetails = result.media_type === "video"
    ? ` Video stride ${result.video_stride || 1}; browser MP4 ${result.browser_video ? "ready" : "fallback"}.`
    : "";
  const predictionCount = task === "classify"
    ? Number(result.classifications) || frames
    : Number(result.detections) || 0;
  const predictionLabel = task === "classify"
    ? `classification${predictionCount === 1 ? "" : "s"}`
    : task === "segment"
      ? `instance${predictionCount === 1 ? "" : "s"}`
      : `detection${predictionCount === 1 ? "" : "s"}`;
  const summary = `${taskLabel} inference processed ${frames} frame${frames === 1 ? "" : "s"} with ${predictionCount} ${predictionLabel} in ${metricText(result.elapsed_ms)} ms.${videoDetails}`;
  $("inference-result-summary").textContent = summary;

  const image = $("inference-result-image");
  const video = $("inference-result-video");
  const imageShell = $("inference-result-image-shell");
  const videoShell = $("inference-result-video-shell");
  if (result.media_type === "video") {
    imageShell.hidden = true;
    image.hidden = true;
    image.removeAttribute("src");
    videoShell.hidden = false;
    video.hidden = false;
    video.src = resultUrl;
    video.load();
  } else {
    videoShell.hidden = true;
    video.hidden = true;
    video.pause();
    video.removeAttribute("src");
    imageShell.hidden = false;
    image.hidden = false;
    image.src = resultUrl;
  }
  renderInferencePredictions(result);
}

function uploadInferenceRequest(form) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/inference/run");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) {
        return;
      }
      const percent = (event.loaded / event.total) * 100;
      setInferenceUploadProgress(true, percent, `Uploading media: ${formatBytes(event.loaded)} of ${formatBytes(event.total)}.`);
    });
    xhr.addEventListener("load", () => {
      let payload = {};
      try {
        payload = xhr.response && typeof xhr.response === "object"
          ? xhr.response
          : JSON.parse(xhr.responseText || "{}");
      } catch (error) {
        payload = {};
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
      } else {
        reject(new Error(payload.detail || `Inference failed: ${xhr.status}`));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("Inference upload failed.")));
    xhr.addEventListener("abort", () => reject(new Error("Inference upload was aborted.")));
    xhr.send(form);
  });
}

async function runInference() {
  const button = $("run-inference");
  const source = selectedInferenceWeightSource();
  const media = $("inference-media-file").files[0];
  if (!media) {
    setInferenceMessage("Choose an image or video file for inference.", true);
    return;
  }
  if (source === "selected" && !$("inference-weight-select").value) {
    setInferenceMessage("Choose weights from runs or upload a .pt/.onnx file.", true);
    return;
  }
  if (source === "upload" && !$("inference-weight-file").files[0]) {
    setInferenceMessage("Choose a .pt or .onnx weights file to upload.", true);
    return;
  }
  if (source === "upload") {
    const suffix = inferenceWeightSuffix($("inference-weight-file").files[0]);
    const uploadFamily = $("inference-upload-family").value;
    const isDetrUpload = uploadFamily === "rfdetr" || uploadFamily === "dfine";
    const supportedSuffixes = isDetrUpload ? [".pt"] : [".pt", ".onnx"];
    if (!supportedSuffixes.includes(suffix)) {
      setInferenceMessage(isDetrUpload ? "Uploaded DETR-family weights must be a .pt file." : "Uploaded weights must be a .pt or .onnx file.", true);
      return;
    }
  }

  const convertToOnnx = source === "upload"
    && inferenceWeightSuffix($("inference-weight-file").files[0]) === ".pt"
    && $("inference-upload-family").value !== "rfdetr"
    && $("inference-upload-family").value !== "dfine"
    && $("inference-convert-onnx").checked;

  const form = new FormData();
  form.append("weight_source", source);
  form.append("weight_path", $("inference-weight-select").value);
  form.append("weight_family", source === "upload" ? $("inference-upload-family").value : "auto");
  form.append("convert_to_onnx", convertToOnnx ? "true" : "false");
  form.append("imgsz", $("inference-imgsz").value || "512");
  form.append("conf", $("inference-conf").value || "0.25");
  form.append("iou", $("inference-iou").value || "0.45");
  form.append("vid_stride", $("inference-vid-stride").value || "1");
  form.append("show_masks", $("inference-show-masks").checked ? "true" : "false");
  form.append("show_boxes", $("inference-show-boxes").checked ? "true" : "false");
  form.append("show_labels", $("inference-show-labels").checked ? "true" : "false");
  form.append("show_conf", $("inference-show-conf").checked ? "true" : "false");
  form.append("media_file", media);
  if (source === "upload") {
    form.append("weight_file", $("inference-weight-file").files[0]);
  }

  stopInferencePolling();
  resetInferenceResultForRun();
  state.inferenceRunning = true;
  state.inferenceStoppable = false;
  syncInferenceControls();
  button.textContent = "Running...";
  setInferenceMessage("Uploading media and starting inference job.");
  setInferenceUploadProgress(true, 0, `Preparing upload for ${media.name}.`);
  try {
    const payload = await uploadInferenceRequest(form);
    state.inferenceJobId = payload.job_id || "";
    if (!state.inferenceJobId) {
      throw new Error("Inference job did not return a job ID.");
    }
    updateInferenceJobProgress(payload);
    setInferenceMessage("Inference job started.");
    const revision = state.inferencePollRevision;
    pollInferenceJob(state.inferenceJobId, revision);
  } catch (error) {
    state.inferenceRunning = false;
    state.inferenceStoppable = false;
    button.textContent = "Run Inference";
    setInferenceMessage(error.message, true);
    syncInferenceControls();
  }
}

async function stopInference() {
  if (!state.inferenceRunning || state.inferenceStopping || !state.inferenceJobId) {
    return;
  }
  state.inferenceStopping = true;
  syncInferenceControls();
  setInferenceMessage("Stopping inference...");
  try {
    const job = await apiJson(`/api/inference/stop/${encodeURIComponent(state.inferenceJobId)}`, {
      method: "POST",
      body: "{}",
    });
    updateInferenceJobProgress(job);
    if (job.status === "stopped") {
      state.inferenceRunning = false;
      state.inferenceStoppable = false;
      stopInferencePolling();
      setInferenceMessage(job.detail || "Inference stopped.");
    }
  } catch (error) {
    setInferenceMessage(error.message, true);
  } finally {
    state.inferenceStopping = false;
    $("run-inference").textContent = "Run Inference";
    syncInferenceControls();
  }
}

async function clearInferenceOutput() {
  if (state.inferenceRunning || state.inferenceClearing) {
    return;
  }
  state.inferenceClearing = true;
  syncInferenceControls();
  try {
    const storage = await apiJson("/api/inference/storage");
    const jobCount = Number(storage.job_count) || 0;
    const jobsSize = Number(storage.jobs_size) || 0;
    if (!jobCount || jobsSize <= 0) {
      setInferenceMessage("No inference output jobs to clear.");
      return;
    }
    const confirmed = window.confirm(
      `Delete ${formatBytes(jobsSize)} from ${jobCount} inference job${jobCount === 1 ? "" : "s"}? This removes old generated results and cannot be undone.`,
    );
    if (!confirmed) {
      setInferenceMessage("Inference output cleanup cancelled.");
      return;
    }
    const result = await apiJson("/api/inference/outputs", { method: "DELETE" });
    resetInferenceResultForRun();
    stopInferencePolling();
    state.inferenceJobId = "";
    state.inferenceStoppable = false;
    setInferenceUploadProgress(false);
    const successMessage = `Cleared ${formatBytes(result.freed_bytes || 0)} from ${result.removed_jobs || 0} inference job${result.removed_jobs === 1 ? "" : "s"}.`;
    setInferenceMessage(successMessage);
    setMessage(successMessage);
  } catch (error) {
    setInferenceMessage(error.message, true);
  } finally {
    state.inferenceClearing = false;
    syncInferenceControls();
  }
}

async function clearStorageTarget(key, options = {}) {
  if (state.storageCleanup.has(key)) {
    return;
  }
  const {
    label = "storage",
    message = setMessage,
    systemMessage = setMessage,
    sync = syncActionStates,
  } = options;
  state.storageCleanup.add(key);
  sync();
  try {
    const storage = await apiJson(`/api/storage/${encodeURIComponent(key)}`);
    const itemCount = Number(storage.item_count) || 0;
    const size = Number(storage.size) || 0;
    if (!itemCount || size <= 0) {
      message(`No ${label} to clear.`);
      return;
    }
    const confirmed = window.confirm(
      `Delete ${formatBytes(size)} from ${itemCount} ${label} item${itemCount === 1 ? "" : "s"}? This cannot be undone.`,
    );
    if (!confirmed) {
      message(`${label} cleanup cancelled.`);
      return;
    }
    const result = await apiJson(`/api/storage/${encodeURIComponent(key)}`, { method: "DELETE" });
    const successMessage = `Cleared ${formatBytes(result.freed_bytes || 0)} from ${result.removed_items || 0} ${label} item${result.removed_items === 1 ? "" : "s"}.`;
    message(successMessage);
    if (systemMessage && systemMessage !== message) {
      systemMessage(successMessage);
    }
  } catch (error) {
    message(error.message, true);
  } finally {
    state.storageCleanup.delete(key);
    sync();
  }
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
    resetAnnotationQaForDataset();
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

async function downloadAnnotatedDataset() {
  if (!state.datasetYaml || state.downloads.has("annotated_dataset")) {
    return;
  }

  state.downloads.add("annotated_dataset");
  syncActionStates();
  setMessage("Preparing annotated dataset ZIP. Large datasets may take several minutes...");
  try {
    const result = await apiJson("/api/dataset/download/annotated/prepare", {
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
    state.downloads.delete("annotated_dataset");
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
  setMessage(`Starting training... ${augmentationConfigurationSummary()}.`);
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
        exist_ok: $("exist-ok").checked,
        seed: numberValue("seed"),
        project: $("project").value,
        name: $("run-name").value,
        resume,
        augmentation_enabled: $("augmentation-enabled").checked,
        disable_ultralytics_albumentations: $("disable-ultralytics-albumentations").checked,
        mosaic: numberValue("mosaic"),
        close_mosaic: numberValue("close-mosaic"),
        hsv_h: numberValue("hsv-h"),
        hsv_s: numberValue("hsv-s"),
        hsv_v: numberValue("hsv-v"),
        degrees: numberValue("degrees"),
        translate: numberValue("translate"),
        scale: numberValue("scale"),
        shear: numberValue("shear"),
        perspective: numberValue("perspective"),
        flipud: numberValue("flipud"),
        fliplr: numberValue("fliplr"),
        bgr: numberValue("bgr"),
        mixup: numberValue("mixup"),
        cutmix: numberValue("cutmix"),
        copy_paste: numberValue("copy-paste"),
        auto_augment: optionalTextValue("auto-augment"),
        erasing: numberValue("erasing"),
      }),
    });
    state.running = true;
    state.trainingStarted = true;
    state.trainingCompleted = false;
    setPanelExpanded("logs", true);
    setPanelExpanded("results", true);
    state.lastDataRefresh = 0;
    setMessage(`${result.message}\nPID: ${result.pid}`);
    if (result.training_run?.requested_project) {
      $("project").value = result.training_run.requested_project;
    }
    if (result.training_run?.name) {
      $("run-name").value = result.training_run.name;
    }
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
      state.latestMetrics = null;
      applyMetricLabels();
      $("training-results-panel").classList.remove("has-results");
      $("metric-precision").textContent = "-";
      $("metric-recall").textContent = "-";
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
      setMagicAdjustedState();
      renderConfusionMatrices(metrics.artifacts || {}, target, metrics);
      renderRocAuc({}, metrics.artifacts || {}, target);
      $("metrics-status").textContent = "No results.csv found for this run yet.";
      return;
    }

    state.metricsAvailable = true;
    state.latestMetrics = metrics;
    applyMetricLabels(metrics.metric_labels, metrics.chart_title || "Detection Performance by Epoch");
    $("training-results-panel").classList.add("has-results");
    $("metric-precision").textContent = metricText(metrics.precision);
    $("metric-recall").textContent = metricText(metrics.recall);
    $("metric-macro-f1").textContent = metricText(metrics.macro_f1);
    $("metric-weighted-f1").textContent = metricText(metrics.weighted_f1);
    $("metric-train-loss").textContent = metricText(metrics.training_loss);
    $("metric-test-loss").textContent = metricText(metrics.testing_loss);
    $("metric-map50").textContent = metricText(metrics.map50);
    $("metric-map").textContent = metricText(metrics.map50_95);
    renderBestMetrics(metrics.best, metrics.history, metrics.metric_labels);
    renderClassMetrics(metrics.per_class);
    renderMetricCharts(metrics.history, metrics.metric_labels);
    setArtifactButtons(metrics.artifacts || true);
    setMagicAdjustedState(metrics);
    renderConfusionMatrices(metrics.artifacts || {}, target, metrics);
    renderRocAuc(metrics.roc_auc || {}, metrics.artifacts || {}, target);
    $("metrics-status").textContent = `Epoch ${metrics.epoch}. ${metrics.note}`;
  } catch (error) {
    if (revision !== state.targetRevision) {
      return;
    }
    state.latestMetrics = null;
    $("training-results-panel").classList.remove("has-results");
    $("metric-precision").textContent = "-";
    $("metric-recall").textContent = "-";
    resetCharts();
    setArtifactButtons(false);
    setMagicAdjustedState();
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
    setMessage(`Preparing ${weight}.pt...`);
    const response = await fetch(weightDownloadUrl(weight));
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Download failed: ${response.status}`);
    }

    const filename = responseDownloadFilename(response, `${weight}.pt`);
    const blob = await response.blob();
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
    const filename = responseDownloadFilename(response, "training_report.pdf");
    saveBlobWithBrowserDownload(await response.blob(), filename);
    setMessage(`Downloading ${filename}.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    state.downloads.delete("training_report");
    button.textContent = originalText;
    await refreshMetrics();
  }
}

function magicCurrentValue(metrics, choice, classChoice) {
  const source = choice.scope === "per_class" ? classChoice : metrics;
  const fallbackKeys = {
    map50_95: ["ap50_95", "AP50-95"],
    map50: ["ap50", "AP50"],
  };
  let value = source ? source[choice.key] : null;
  if ((value === null || value === undefined) && source) {
    for (const key of fallbackKeys[choice.key] || []) {
      if (source[key] !== null && source[key] !== undefined) {
        value = source[key];
        break;
      }
    }
  }
  return value === null || value === undefined ? "" : String(value);
}

function magicClasses(metrics) {
  return Array.isArray(metrics?.per_class) ? metrics.per_class : [];
}

function selectedMagicClass(metrics) {
  const index = Number.parseInt($("magic-class-select").value, 10);
  const classes = magicClasses(metrics);
  return Number.isInteger(index) && classes[index] ? classes[index] : null;
}

function magicChoiceLabel(choice, classRow = null) {
  if (!choice) {
    return "";
  }
  if (choice.scope === "per_class") {
    return `${classRow?.class_name || "Class"} ${choice.label}`;
  }
  return choice.label;
}

function setMagicStatus(text, isError = false) {
  $("magic-status").textContent = text;
  $("magic-status").classList.toggle("error", isError);
}

function magicAdjustmentId(adjustment) {
  return [adjustment.scope, adjustment.class_name || "", adjustment.metric_key].join("::");
}

function magicExistingAdjustments(metrics) {
  if (Array.isArray(metrics?.magic_adjustments) && metrics.magic_adjustments.length) {
    return metrics.magic_adjustments.map((adjustment) => ({
      scope: adjustment.scope,
      metric_key: adjustment.metric_key,
      target: Number(adjustment.target),
      class_name: adjustment.class_name || "",
    })).filter((adjustment) => Number.isFinite(adjustment.target));
  }
  if (metrics?.magic_adjusted && metrics.magic_metric_key) {
    return [{
      scope: metrics.magic_scope || "overall",
      metric_key: metrics.magic_metric_key,
      target: Number(metrics.magic_target),
      class_name: metrics.magic_class_name || "",
    }].filter((adjustment) => Number.isFinite(adjustment.target));
  }
  return [];
}

function magicOptionForAdjustment(adjustment) {
  return [...MAGIC_OVERALL_OPTIONS, ...MAGIC_PER_CLASS_OPTIONS]
    .find((option) => option.scope === adjustment.scope && option.key === adjustment.metric_key);
}

function renderMagicAdjustments() {
  const list = $("magic-adjustment-list");
  const adjustments = state.magicAdjustments;
  $("magic-selection-count").textContent = `${adjustments.length} selected`;
  $("magic-empty").hidden = adjustments.length > 0;
  $("magic-clear").disabled = adjustments.length === 0;
  list.innerHTML = adjustments.map((adjustment) => {
    const option = magicOptionForAdjustment(adjustment);
    const label = adjustment.scope === "per_class"
      ? `${adjustment.class_name || "Class"} ${option?.label || adjustment.metric_key}`
      : option?.label || adjustment.metric_key;
    return `
      <div class="magic-adjustment-row" data-magic-adjustment-id="${escapeHtml(magicAdjustmentId(adjustment))}">
        <span class="magic-adjustment-copy">
          <strong>${escapeHtml(label)}</strong>
          <small>${adjustment.scope === "per_class" ? "Per-class" : "Overall"} target: ${Number(adjustment.target).toFixed(4)}</small>
        </span>
        <span class="magic-adjustment-controls">
          <button class="secondary compact" type="button" data-magic-edit="${escapeHtml(magicAdjustmentId(adjustment))}">Edit</button>
          <button class="secondary danger compact magic-remove-adjustment" type="button" data-magic-remove="${escapeHtml(magicAdjustmentId(adjustment))}">Remove</button>
        </span>
      </div>`;
  }).join("");
  $("magic-apply").disabled = adjustments.length === 0;
}

function addMagicAdjustment() {
  const metrics = state.latestMetrics;
  const choice = state.magicChoice;
  if (!metrics?.available || !choice) {
    setMagicStatus("Metrics are not available for this run.", true);
    return;
  }
  const classRow = choice.scope === "per_class" ? selectedMagicClass(metrics) : null;
  if (choice.scope === "per_class" && !classRow) {
    setMagicStatus("Choose a class row.", true);
    return;
  }
  const target = Number($("magic-target").value);
  if (!Number.isFinite(target) || target < 0 || target > 1) {
    setMagicStatus("Enter a target score between 0 and 1.", true);
    return;
  }
  const adjustment = {
    scope: choice.scope,
    metric_key: choice.key,
    target,
    class_name: classRow?.class_name || "",
  };
  const identity = magicAdjustmentId(adjustment);
  const existingIndex = state.magicAdjustments.findIndex((item) => magicAdjustmentId(item) === identity);
  if (existingIndex >= 0) {
    state.magicAdjustments[existingIndex] = adjustment;
    setMagicStatus("Updated the selected score target. Apply all to save the complete set.");
  } else {
    state.magicAdjustments.push(adjustment);
    setMagicStatus("Added the score. Apply all to save the complete set.");
  }
  renderMagicAdjustments();
}

function renderMagicOptionGroup(containerId, options) {
  const container = $(containerId);
  container.innerHTML = options.map((option) => `
    <button
      class="secondary magic-option"
      type="button"
      data-magic-scope="${option.scope}"
      data-magic-key="${option.key}"
    >${escapeHtml(option.label)}</button>
  `).join("");
}

function populateMagicClassSelect(metrics) {
  const select = $("magic-class-select");
  select.innerHTML = "";
  const classes = magicClasses(metrics);
  if (!classes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No per-class metrics available";
    select.appendChild(option);
    return;
  }
  classes.forEach((row, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = row.class_name || "Unnamed class";
    select.appendChild(option);
  });
}

function setMagicChoice(scope, key) {
  state.magicChoice = [...MAGIC_OVERALL_OPTIONS, ...MAGIC_PER_CLASS_OPTIONS]
    .find((option) => option.scope === scope && option.key === key) || MAGIC_OVERALL_OPTIONS[0];
  document.querySelectorAll("[data-magic-scope]").forEach((button) => {
    button.classList.toggle(
      "active-control",
      button.dataset.magicScope === state.magicChoice.scope && button.dataset.magicKey === state.magicChoice.key,
    );
  });
  updateMagicModalFields();
}

function updateMagicModalFields() {
  const metrics = state.latestMetrics;
  const choice = state.magicChoice || MAGIC_OVERALL_OPTIONS[0];
  const isPerClass = choice.scope === "per_class";
  const classField = $("magic-class-field");
  const classSelect = $("magic-class-select");
  const classes = magicClasses(metrics);
  const hasClasses = classes.length > 0;
  const previousIndex = Number.parseInt(classSelect.value, 10);
  classSelect.disabled = !isPerClass || !hasClasses;
  classField.classList.toggle("magic-field-disabled", classSelect.disabled);
  if (!isPerClass) {
    classSelect.innerHTML = '<option value="">Select a per-class metric first</option>';
  } else if (!hasClasses) {
    classSelect.innerHTML = '<option value="">No per-class metrics available</option>';
  } else if (
    classSelect.options.length !== classes.length
    || (classSelect.options.length && classSelect.options[0].value === "")
  ) {
    populateMagicClassSelect(metrics);
    if (Number.isInteger(previousIndex) && classes[previousIndex]) {
      classSelect.value = String(previousIndex);
    }
  }
  const classRow = isPerClass ? selectedMagicClass(metrics) : null;
  $("magic-add").disabled = isPerClass && !classRow;

  const currentValue = magicCurrentValue(metrics, choice, classRow);
  $("magic-target").value = currentValue;
  $("magic-current").textContent = currentValue === ""
    ? `${magicChoiceLabel(choice, classRow)} current score: N/A`
    : `${magicChoiceLabel(choice, classRow)} current score: ${currentValue}`;
  setMagicStatus("Raw logs, results.csv, and weights stay unchanged.");
}

function openMagicMetricsModal() {
  const metrics = state.latestMetrics;
  if (!metrics?.available) {
    setMessage("Refresh metrics before using the Magic Button.", true);
    return;
  }

  renderMagicOptionGroup("magic-overall-options", MAGIC_OVERALL_OPTIONS);
  renderMagicOptionGroup("magic-per-class-options", MAGIC_PER_CLASS_OPTIONS);
  populateMagicClassSelect(metrics);
  state.magicAdjustments = magicExistingAdjustments(metrics);
  state.magicChoice = state.magicChoice || MAGIC_OVERALL_OPTIONS[0];
  setMagicChoice(state.magicChoice.scope, state.magicChoice.key);
  renderMagicAdjustments();
  $("magic-reset").disabled = !metrics.magic_adjusted;
  $("magic-modal").hidden = false;
  $("magic-modal").querySelector(".magic-dialog")?.focus();
}

function closeMagicMetricsModal() {
  $("magic-modal").hidden = true;
  $("magic-apply").textContent = "Apply all";
  state.downloads.delete("magic_metrics");
}

async function applyMagicMetrics() {
  const metrics = state.latestMetrics;
  const adjustments = state.magicAdjustments;
  if (!metrics?.available) {
    setMagicStatus("Metrics are not available for this run.", true);
    return;
  }
  if (!adjustments.length) {
    setMagicStatus("Add at least one score before applying.", true);
    return;
  }

  const button = $("magic-apply");
  state.downloads.add("magic_metrics");
  button.disabled = true;
  button.textContent = "Tweaking...";
  try {
    const response = await fetch("/api/train/metrics/magic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...weightTarget(),
        adjustments,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(errorDetailText(payload, `Metric adjustment failed: ${response.status}`));
    }
    closeMagicMetricsModal();
    setMessage(`Magic Button applied ${adjustments.length} adjusted ${adjustments.length === 1 ? "score" : "scores"} for reports.`);
    await refreshMetrics();
  } catch (error) {
    setMagicStatus(error.message, true);
  } finally {
    state.downloads.delete("magic_metrics");
    button.disabled = false;
    button.textContent = "Apply all";
  }
}

async function resetMagicMetrics() {
  const button = $("magic-reset");
  state.downloads.add("magic_metrics");
  button.disabled = true;
  button.textContent = "Resetting...";
  try {
    const response = await fetch("/api/train/metrics/magic/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(weightTarget()),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = response.status === 404
        ? "Reset service is unavailable. Restart the training web service and try again."
        : errorDetailText(payload, `Metric reset failed: ${response.status}`);
      throw new Error(detail);
    }
    state.magicAdjustments = [];
    closeMagicMetricsModal();
    setMessage("Magic Button adjustments cleared. Raw validation metrics are active.");
    await refreshMetrics();
  } catch (error) {
    setMagicStatus(error.message, true);
  } finally {
    state.downloads.delete("magic_metrics");
    button.textContent = "Reset to raw";
    button.disabled = !state.latestMetrics?.magic_adjusted;
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
      renderTestTiming({});
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
    renderTestTiming(results.timing);
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
    const filename = responseDownloadFilename(response, "training_and_test_report.pdf");
    saveBlobWithBrowserDownload(await response.blob(), filename);
    setMessage(`Downloading ${filename}.`);
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
document.querySelectorAll("[data-app-tab]").forEach((button) => {
  button.addEventListener("click", () => setAppTab(button.dataset.appTab));
});

$("prepare-dataset").addEventListener("click", prepareDataset);
$("download-dataset").addEventListener("click", downloadPreparedDataset);
$("download-annotated-dataset").addEventListener("click", downloadAnnotatedDataset);
$("run-annotation-qa").addEventListener("click", runAnnotationQa);
$("stop-annotation-qa").addEventListener("click", stopAnnotationQa);
$("annotation-qa-preset").addEventListener("change", () => {
  applyAnnotationQaDifferencePreset();
  updateAnnotationQaConfigurationSummary();
});
$("annotation-qa-scope").addEventListener("change", updateAnnotationQaConfigurationSummary);
$("annotation-qa-auto-mode").addEventListener("change", updateAnnotationQaConfigurationSummary);
$("download-annotation-qa-csv").addEventListener("click", () => downloadAnnotationQaReport("csv"));
$("download-annotation-qa-json").addEventListener("click", () => downloadAnnotationQaReport("json"));
$("apply-annotation-qa-fixes").addEventListener("click", applyAnnotationQaFixes);
$("download-corrected-dataset").addEventListener("click", downloadCorrectedDataset);
document.querySelectorAll("[data-qa-queue]").forEach((button) => {
  button.addEventListener("click", () => {
    state.annotationQaQueue = button.dataset.qaQueue;
    state.annotationQaPage = 1;
    renderAnnotationQaIssues();
  });
});
[
  ["annotation-qa-filter-severity", "severity"],
  ["annotation-qa-filter-split", "split"],
  ["annotation-qa-filter-class", "className"],
  ["annotation-qa-filter-type", "issueType"],
].forEach(([id, key]) => {
  $(id).addEventListener("change", () => {
    state.annotationQaFilters[key] = $(id).value;
    state.annotationQaPage = 1;
    renderAnnotationQaIssues();
  });
});
$("annotation-qa-search").addEventListener("input", () => {
  state.annotationQaFilters.search = $("annotation-qa-search").value.trim().toLowerCase();
  state.annotationQaPage = 1;
  renderAnnotationQaIssues();
});
$("annotation-qa-clear-filters").addEventListener("click", () => {
  state.annotationQaFilters = { search: "", severity: "", split: "", className: "", issueType: "" };
  $("annotation-qa-search").value = "";
  ["annotation-qa-filter-severity", "annotation-qa-filter-split", "annotation-qa-filter-class", "annotation-qa-filter-type"].forEach((id) => {
    $(id).value = "";
  });
  state.annotationQaPage = 1;
  renderAnnotationQaIssues();
});
$("annotation-qa-page-prev").addEventListener("click", () => {
  state.annotationQaPage -= 1;
  renderAnnotationQaIssues();
});
$("annotation-qa-page-next").addEventListener("click", () => {
  state.annotationQaPage += 1;
  renderAnnotationQaIssues();
});
$("annotation-qa-select-page").addEventListener("change", () => {
  document.querySelectorAll("#annotation-qa-issues [data-qa-select]").forEach((checkbox) => {
    checkbox.checked = $("annotation-qa-select-page").checked;
    if (checkbox.checked) {
      state.annotationQaSelected.add(checkbox.dataset.qaSelect);
    } else {
      state.annotationQaSelected.delete(checkbox.dataset.qaSelect);
    }
  });
  renderAnnotationQaBulkToolbar(annotationQaFilteredIssues().slice(
    (state.annotationQaPage - 1) * state.annotationQaPageSize,
    state.annotationQaPage * state.annotationQaPageSize,
  ));
});
$("annotation-qa-bulk-accept-sam").addEventListener("click", bulkAcceptAnnotationQaSamBoxes);
$("dataset-cleanup-toggle").addEventListener("click", toggleDatasetCleanupMenu);
$("clear-dataset-uploads").addEventListener("click", () => {
  closeDatasetCleanupMenu();
  clearStorageTarget("dataset_uploads", { label: "uploaded dataset ZIP" });
});
$("clear-dataset-extracted").addEventListener("click", () => {
  closeDatasetCleanupMenu();
  clearStorageTarget("dataset_extracted", { label: "extracted dataset" });
});
$("clear-dataset-prepared").addEventListener("click", () => {
  closeDatasetCleanupMenu();
  clearStorageTarget("dataset_prepared", { label: "prepared dataset" });
});
$("detect-classes").addEventListener("click", detectClasses);
$("model-selector-toggle").addEventListener("click", () => {
  if ($("model-selector-menu").hidden) {
    openModelSelector();
  } else {
    closeModelSelector();
  }
});
$("model-search").addEventListener("input", filterModelOptions);
$("model-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    const firstOption = $("model-options").querySelector(".model-option");
    if (firstOption) {
      event.preventDefault();
      chooseModelOption(firstOption.dataset.value);
    }
  }
});
["model-task", "model-family"].forEach((id) => {
  $(id).addEventListener("change", chooseGuidedModel);
});
$("model-size").addEventListener("change", () => {
  applyModelFamilyDefaults();
  syncModelFamilyControls();
  syncModelSelectorDisplay();
  syncGuidedControlsFromModel();
  filterModelOptions();
  syncProjectWithModelTask();
});
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
$("magic-metrics").addEventListener("click", openMagicMetricsModal);
$("magic-add").addEventListener("click", addMagicAdjustment);
$("magic-apply").addEventListener("click", applyMagicMetrics);
$("magic-reset").addEventListener("click", resetMagicMetrics);
$("magic-clear").addEventListener("click", () => {
  state.magicAdjustments = [];
  renderMagicAdjustments();
  setMagicStatus("Cleared the pending adjustment set. Saved adjustments remain active until you apply or reset.");
});
$("magic-cancel").addEventListener("click", closeMagicMetricsModal);
$("magic-close").addEventListener("click", closeMagicMetricsModal);
document.querySelectorAll("[data-magic-close]").forEach((element) => {
  element.addEventListener("click", closeMagicMetricsModal);
});
$("magic-class-select").addEventListener("change", updateMagicModalFields);
$("magic-target").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addMagicAdjustment();
  }
});
$("magic-adjustment-list").addEventListener("click", (event) => {
  const editButton = event.target.closest("[data-magic-edit]");
  if (editButton) {
    const adjustment = state.magicAdjustments
      .find((item) => magicAdjustmentId(item) === editButton.dataset.magicEdit);
    if (!adjustment) {
      return;
    }
    setMagicChoice(adjustment.scope, adjustment.metric_key);
    if (adjustment.scope === "per_class") {
      const classIndex = magicClasses(state.latestMetrics)
        .findIndex((row) => row.class_name === adjustment.class_name);
      if (classIndex >= 0) {
        $("magic-class-select").value = String(classIndex);
        updateMagicModalFields();
      }
    }
    $("magic-target").value = String(adjustment.target);
    $("magic-target").focus();
    setMagicStatus("Editing the selected score. Choose Add score to update it in the pending set.");
    return;
  }
  const button = event.target.closest("[data-magic-remove]");
  if (!button) {
    return;
  }
  state.magicAdjustments = state.magicAdjustments
    .filter((adjustment) => magicAdjustmentId(adjustment) !== button.dataset.magicRemove);
  renderMagicAdjustments();
  setMagicStatus("Removed the score from the pending adjustment set.");
});
["magic-overall-options", "magic-per-class-options"].forEach((id) => {
  $(id).addEventListener("click", (event) => {
    const button = event.target.closest("[data-magic-scope]");
    if (button) {
      setMagicChoice(button.dataset.magicScope, button.dataset.magicKey);
    }
  });
});
$("download-run-log").addEventListener("click", () => downloadLog("/api/train/logs/download"));
$("start-test").addEventListener("click", startTest);
$("stop-test").addEventListener("click", stopTest);
$("refresh-test-results").addEventListener("click", refreshTestResults);
$("download-test-log").addEventListener("click", downloadTestLog);
$("download-test-metrics-json").addEventListener("click", () => downloadTestArtifact("metrics_json", "test_metrics.json"));
$("download-test-roc-auc-graph").addEventListener("click", () => downloadTestArtifact("roc_auc_curve", "test_roc_auc_curve.png"));
$("download-combined-report").addEventListener("click", downloadCombinedReport);
$("refresh-training-sessions").addEventListener("click", () => (
  loadTrainingSessions().catch((error) => setTrainingSessionStatus(trainingSessionErrorMessage(error), true))
));
$("load-training-session").addEventListener("click", loadSelectedTrainingSession);
$("training-session-select").addEventListener("change", () => {
  if ($("training-session-select").value) {
    loadSelectedTrainingSession();
  }
});
$("refresh-inference-weights").addEventListener("click", () => loadInferenceWeights().catch((error) => setInferenceMessage(error.message, true)));
$("clear-inference-output").addEventListener("click", clearInferenceOutput);
$("clear-inference-uploads").addEventListener("click", () => clearStorageTarget("inference_uploads", {
  label: "uploaded inference weight",
  message: setInferenceMessage,
  sync: syncInferenceControls,
}));
$("run-inference").addEventListener("click", runInference);
$("stop-inference").addEventListener("click", stopInference);
$("project").addEventListener("input", scheduleTargetRefresh);
$("run-name").addEventListener("input", scheduleTargetRefresh);
$("upload-file").addEventListener("change", updateFileSelection);
$("folder-files").addEventListener("change", updateFileSelection);
$("test-weight-file").addEventListener("change", updateFileSelection);
$("test-dataset-zip").addEventListener("change", updateFileSelection);
$("test-dataset-folder").addEventListener("change", updateFileSelection);
$("inference-weight-file").addEventListener("change", updateFileSelection);
$("inference-media-file").addEventListener("change", updateFileSelection);
$("inference-weight-select").addEventListener("change", syncInferenceControls);
$("inference-upload-family").addEventListener("change", syncInferenceControls);
$("inference-convert-onnx").addEventListener("change", syncInferenceControls);
["upload-force-split", "folder-force-split", "roboflow-force-split"].forEach((id) => {
  $(id).addEventListener("change", syncDatasetSourceControls);
});
document.querySelectorAll('input[name="test-weight-source"]').forEach((input) => {
  input.addEventListener("change", syncTestSourceControls);
});
document.querySelectorAll('input[name="test-dataset-source"]').forEach((input) => {
  input.addEventListener("change", syncTestSourceControls);
});
document.querySelectorAll('input[name="inference-weight-source"]').forEach((input) => {
  input.addEventListener("change", syncInferenceControls);
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
document.addEventListener("click", (event) => {
  if (!$("dataset-cleanup-menu").contains(event.target)) {
    closeDatasetCleanupMenu();
  }
  if (!$("model-selector-menu").contains(event.target) && !$("model-selector-toggle").contains(event.target)) {
    closeModelSelector();
  }
});
document.addEventListener("keydown", (event) => {
  handleAnnotationQaReviewKeydown(event);
  if (event.key === "Escape") {
    if ($("magic-modal") && !$("magic-modal").hidden) {
      closeMagicMetricsModal();
    }
    if ($("qa-review-modal") && !$("qa-review-modal").hidden) {
      closeAnnotationQaReview();
    }
    closeDatasetCleanupMenu();
    closeModelSelector();
  }
});
if ($("qa-review-modal")) {
  $("qa-review-close").addEventListener("click", closeAnnotationQaReview);
  document.querySelectorAll("[data-qa-review-close]").forEach((element) => {
    element.addEventListener("click", closeAnnotationQaReview);
  });
  $("qa-review-status").addEventListener("change", () => {
    setActiveAnnotationQaReviewStatus($("qa-review-status").value);
  });
  $("qa-review-accept-sam").addEventListener("click", () => {
    if (state.annotationQaActiveIssueId) {
      decideAnnotationQaSamBox();
    }
  });
  $("qa-review-keep-yolo").addEventListener("click", () => decideAnnotationQaStatus("accepted", "Original YOLO box kept."));
  $("qa-review-accept-class").addEventListener("click", () => {
    if (state.annotationQaActiveIssueId) {
      decideAnnotationQaClassChange();
    }
  });
  $("qa-review-class-fix").addEventListener("change", () => {
    $("qa-review-accept-class").disabled = !Number.isInteger(Number($("qa-review-class-fix").value));
    $("qa-review-accept-class").textContent = "Use Selected Class";
  });
  $("qa-review-needs-fix").addEventListener("click", () => decideAnnotationQaStatus("needs_fix", "Annotation flagged for manual correction."));
  $("qa-review-false-positive").addEventListener("click", () => decideAnnotationQaStatus("false_positive", "Annotation marked as a suspected false positive."));
  $("qa-review-ignore").addEventListener("click", () => decideAnnotationQaStatus("ignored", "Issue ignored."));
  $("qa-review-prev").addEventListener("click", () => stepAnnotationQaReview(-1));
  $("qa-review-next").addEventListener("click", () => stepAnnotationQaReview(1));
  $("qa-review-zoom-out").addEventListener("click", () => setAnnotationQaZoom(state.annotationQaZoom - 0.25));
  $("qa-review-zoom-reset").addEventListener("click", () => setAnnotationQaZoom(1));
  $("qa-review-zoom-in").addEventListener("click", () => setAnnotationQaZoom(state.annotationQaZoom + 0.25));
  ["qa-overlay-yolo", "qa-overlay-mask", "qa-overlay-sam"].forEach((id) => {
    $(id).addEventListener("change", () => {
      const issue = annotationQaIssueById(state.annotationQaActiveIssueId);
      if (issue) {
        renderAnnotationQaCanvas(issue);
      }
    });
  });
  $("qa-undo-button").addEventListener("click", undoAnnotationQaDecision);
}
$("reset-advanced").addEventListener("click", () => {
  applyControlValues(CONTROL_DEFAULTS);
  setActivePreset(null);
  setMessage("Reset training controls to defaults.");
});
$("augmentation-enabled").addEventListener("change", () => {
  syncAugmentationControls();
  setMessage($("augmentation-enabled").checked
    ? "Augmentation is On. The configured values will be applied during training."
    : "Augmentation is Off. No training augmentation will be applied; configured values are preserved.");
});
const presetControlIds = new Set([
  ...Object.keys(CONTROL_DEFAULTS),
  ...Object.values(TRAINING_PRESETS).flatMap((preset) => Object.keys(preset)),
]);
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

syncGuidedControlsFromModel();
syncModelSelectorDisplay();
filterModelOptions();
syncModelFamilyControls();
loadConfig().catch((error) => setMessage(error.message, true));
initializeTooltips();
initializeChartTooltips();
initializeCollapsiblePanels();
filterModelOptions();
updateFileSelection();
updateSplitTotal();
syncDatasetSourceControls();
syncInferenceControls();
resetAnnotationQaForDataset();
setActivePreset(null);
updateCurrentRunDisplay();
renderEpochProgress();
syncActionStates();
loadTrainingSessions().catch((error) => setTrainingSessionStatus(trainingSessionErrorMessage(error), true));
loadInferenceWeights().catch((error) => setInferenceMessage(error.message, true));
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
