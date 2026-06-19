const state = {
  source: "folder",
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

function setClassNames(classes) {
  if (Array.isArray(classes) && classes.length) {
    $("classes").value = classes.join("\n");
  }
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
      $("weights-status").textContent = `Available: ${sizes.join(", ")}`;
    } else {
      $("weights-status").textContent = "No trained weights found for this run yet.";
    }
  } catch (error) {
    $("download-best").disabled = true;
    $("download-last").disabled = true;
    $("weights-status").textContent = error.message;
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

async function pollStatus() {
  try {
    const status = await apiJson("/api/train/status");
    $("status-pill").textContent = status.running ? "Training" : "Idle";
    $("logs").textContent = status.log_tail || "";
    refreshWeightsStatus();
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
$("detect-classes").addEventListener("click", detectClasses);
$("start-training").addEventListener("click", startTraining);
$("stop-training").addEventListener("click", stopTraining);
$("refresh-logs").addEventListener("click", refreshLogs);
$("download-best").addEventListener("click", () => downloadWeight("best"));
$("download-last").addEventListener("click", () => downloadWeight("last"));
$("project").addEventListener("input", refreshWeightsStatus);
$("run-name").addEventListener("input", refreshWeightsStatus);

loadConfig().catch((error) => setMessage(error.message, true));
state.pollTimer = window.setInterval(pollStatus, 2500);
pollStatus();
