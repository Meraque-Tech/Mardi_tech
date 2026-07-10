// REST calls for the pretrained YOLO object-detection workflow -- a separate
// backend path (nn_graph/detection.py) from the classification graph builder.

async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export function listModels() {
  return getJSON("/api/detect/models");
}
export function listDatasets() {
  return getJSON("/api/detect/datasets");
}
export function loadModel(modelId) {
  return postJSON("/api/detect/load", { model_id: modelId });
}
export function trainDetect(cfg) {
  return postJSON("/api/detect/train", cfg);
}
export function stopDetect() {
  return postJSON("/api/detect/stop", {});
}
export function detectStatus() {
  return getJSON("/api/detect/status");
}

export async function predict(file, conf) {
  const form = new FormData();
  form.append("image", file);
  form.append("conf", conf);
  const res = await fetch("/api/detect/predict", { method: "POST", body: form });
  return res.json();
}

export async function downloadWeights() {
  const res = await fetch("/api/detect/download");
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return { ok: false, error: err.error || "Download failed" };
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const match = /filename="?([^";]+)"?/.exec(cd);
  const filename = match ? match[1] : "model_finetuned.pt";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  return { ok: true };
}
