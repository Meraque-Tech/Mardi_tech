const state = {
  source: "local",
  datasetYaml: "",
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function setMessage(text, isError = false) {
  const message = $("message");
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function numberValue(id) {
  return Number($(id).value);
}

function classNames() {
  return $("classes").value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitConfig() {
  return {
    train: numberValue("split-train"),
    val: numberValue("split-val"),
    test: numberValue("split-test"),
  };
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

async function prepareLocalDataset() {
  return apiJson("/api/dataset/local", {
    method: "POST",
    body: JSON.stringify({
      path: $("local-path").value,
      classes: classNames(),
      split: splitConfig(),
      name: $("dataset-name").value,
      force_split: $("local-force-split").checked,
    }),
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
    let result;
    if (state.source === "local") {
      result = await prepareLocalDataset();
    } else if (state.source === "upload") {
      result = await prepareUploadedDataset();
    } else {
      result = await prepareRoboflowDataset();
    }

    state.datasetYaml = result.dataset_yaml;
    $("dataset-yaml").value = result.dataset_yaml;
    setMessage(result.message);
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

async function pollStatus() {
  try {
    const status = await apiJson("/api/train/status");
    $("status-pill").textContent = status.running ? "Training" : "Idle";
    $("logs").textContent = status.log_tail || "";
  } catch (error) {
    $("status-pill").textContent = "Error";
    setMessage(error.message, true);
  }
}

async function refreshLogs() {
  const response = await fetch("/api/train/logs");
  $("logs").textContent = await response.text();
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});

$("prepare-dataset").addEventListener("click", prepareDataset);
$("start-training").addEventListener("click", startTraining);
$("stop-training").addEventListener("click", stopTraining);
$("refresh-logs").addEventListener("click", refreshLogs);

loadConfig().catch((error) => setMessage(error.message, true));
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
