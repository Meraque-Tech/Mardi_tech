// REST calls against the backend for validate/build/export/save/load/list/delete.

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export function validateGraph(graph) {
  return postJSON("/api/graph/validate", { graph });
}

export function buildGraph(graph) {
  return postJSON("/api/graph/build", { graph });
}

export function exportGraph(graph, trainConfig) {
  return postJSON("/api/graph/export", { graph, train_config: trainConfig });
}

export async function listGraphs() {
  const res = await fetch("/api/graphs");
  return res.json();
}

export function saveGraph(name, graph) {
  return postJSON(`/api/graphs/${encodeURIComponent(name)}`, { graph });
}

export async function loadGraph(name) {
  const res = await fetch(`/api/graphs/${encodeURIComponent(name)}`);
  return res.json();
}

export async function deleteGraph(name) {
  const res = await fetch(`/api/graphs/${encodeURIComponent(name)}`, { method: "DELETE" });
  return res.json();
}

export function trainStart(graph) {
  return postJSON("/api/train/start", { graph });
}
export function trainStop() {
  return postJSON("/api/train/stop", {});
}
export function trainPause() {
  return postJSON("/api/train/pause", {});
}
export function trainResume() {
  return postJSON("/api/train/resume", {});
}
export function trainStep() {
  return postJSON("/api/train/step", {});
}
export function trainEvaluate() {
  return postJSON("/api/train/evaluate", {});
}
export async function trainStatus() {
  const res = await fetch("/api/train/status");
  return res.json();
}

export async function downloadCheckpoint() {
  const res = await fetch("/api/train/download");
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return { ok: false, error: err.error || "Download failed" };
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const match = /filename="?([^";]+)"?/.exec(cd);
  const filename = match ? match[1] : "model_state_dict.pt";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  return { ok: true };
}
