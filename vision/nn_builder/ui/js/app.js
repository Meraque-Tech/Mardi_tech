import "./ws/ws_client.js";
import { loadCatalog, families, getSpec } from "./nodes/node_defs.js";
import { GraphModel } from "./graph/graphModel.js";
import { NodeCanvasEditor } from "./canvas/interaction.js";
import { Inspector } from "./panel/inspector.js";
import { TrainingPanel } from "./training/trainingPanel.js";
import * as api from "./graph/graphIO.js";

function starterGraph(graph) {
  const input = graph.addNode("input", { x: 40, y: 160 });
  const l1 = graph.addNode("linear", { x: 260, y: 160 });
  const relu = graph.addNode("relu", { x: 460, y: 160 });
  const l2 = graph.addNode("linear", { x: 660, y: 160 });
  const output = graph.addNode("output", { x: 860, y: 160 });
  graph.addEdge(input, l1);
  graph.addEdge(l1, relu);
  graph.addEdge(relu, l2);
  graph.addEdge(l2, output);

  graph.updateParams(l2, { out_features: 2 });

  graph.addNode("dataset", { x: 40, y: 400 });
  graph.addNode("optimizer", { x: 260, y: 400 });
  graph.addNode("loss", { x: 460, y: 400 });
  graph.addNode("train_config", { x: 660, y: 400 });
  graph.meta.name = "starter_mlp";
}

function buildPalette(root, editor) {
  root.innerHTML = "";
  const fams = families();
  const order = ["struct", "cnn", "ann", "rnn", "transformer", "activation", "config"];
  for (const fam of order) {
    const types = fams[fam];
    if (!types) continue;
    const details = document.createElement("details");
    details.className = `palette-group fam-${fam}`;
    details.open = fam === "ann" || fam === "struct";
    const summary = document.createElement("summary");
    summary.textContent = fam;
    details.appendChild(summary);
    for (const type of types) {
      const spec = getSpec(type);
      const item = document.createElement("div");
      item.className = "palette-item";
      item.textContent = spec.label;
      item.draggable = true;
      item.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", type);
      });
      item.addEventListener("dblclick", () => {
        editor.addNodeFromPalette(type, { x: 300, y: 200 });
      });
      details.appendChild(item);
    }
    root.appendChild(details);
  }
}

function wireTabs() {
  document.querySelectorAll(".nav-item[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item[data-tab]").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-page").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

function errorNodeIds(result) {
  return new Set((result.errors || []).map((e) => e.node_id).filter(Boolean));
}

async function main() {
  await loadCatalog();

  const graph = new GraphModel();
  starterGraph(graph);

  const canvas = document.getElementById("node-canvas");
  const editor = new NodeCanvasEditor(canvas, graph);
  const inspector = new Inspector(document.getElementById("inspector"), graph, () => {});
  inspector.showEmpty();
  editor.onSelect = (id) => (id ? inspector.show(id) : inspector.showEmpty());

  buildPalette(document.getElementById("palette"), editor);

  const canvasWrap = document.getElementById("canvas-wrap");
  canvasWrap.addEventListener("dragover", (e) => e.preventDefault());
  canvasWrap.addEventListener("drop", (e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData("text/plain");
    if (!type) return;
    const rect = canvas.getBoundingClientRect();
    editor.addNodeFromPalette(type, { x: e.clientX - rect.left, y: e.clientY - rect.top });
  });

  document.getElementById("btn-zoom-reset").addEventListener("click", () => {
    editor.viewport.offsetX = 0;
    editor.viewport.offsetY = 0;
    editor.viewport.scale = 1;
    editor.render();
  });

  wireTabs();
  new TrainingPanel(document.getElementById("training-tab"), graph);

  document.getElementById("btn-new").addEventListener("click", () => {
    if (!confirm("Clear the current graph?")) return;
    graph.clear();
    starterGraph(graph);
    inspector.showEmpty();
    editor.setValidation({}, new Set());
  });

  document.getElementById("btn-validate").addEventListener("click", async () => {
    const res = await api.validateGraph(graph.toJSON());
    editor.setValidation(res.shapes, errorNodeIds(res));
    toast(res.ok ? "Graph is valid" : (res.errors || []).map((e) => e.message).join("; "));
  });

  document.getElementById("btn-build").addEventListener("click", async () => {
    const res = await api.buildGraph(graph.toJSON());
    if (res.ok) toast(`Built OK — ${res.param_count} trainable parameters`);
    else {
      editor.setValidation(res.shapes || {}, errorNodeIds(res));
      toast((res.errors || []).map((e) => e.message).join("; "));
    }
  });

  document.getElementById("btn-export").addEventListener("click", async () => {
    const res = await api.exportGraph(graph.toJSON());
    if (!res.ok) return toast((res.errors || []).map((e) => e.message).join("; "));
    const blob = new Blob([res.code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = res.filename;
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("btn-save").addEventListener("click", async () => {
    const name = document.getElementById("graph-name").value.trim();
    if (!name) return toast("Enter a graph name first");
    graph.meta.name = name;
    const res = await api.saveGraph(name, graph.toJSON());
    if (res.ok) {
      toast(`Saved as "${name}"`);
      refreshSavedList();
    }
  });

  document.getElementById("btn-delete").addEventListener("click", async () => {
    const select = document.getElementById("saved-graphs");
    const name = select.value;
    if (!name) return toast("Choose a saved graph to delete");
    await api.deleteGraph(name);
    toast(`Deleted "${name}"`);
    refreshSavedList();
  });

  document.getElementById("saved-graphs").addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) return;
    const d = await api.loadGraph(name);
    graph.fromJSON(d);
    document.getElementById("graph-name").value = name;
    inspector.showEmpty();
    editor.setValidation({}, new Set());
  });

  async function refreshSavedList() {
    const list = await api.listGraphs();
    const select = document.getElementById("saved-graphs");
    select.innerHTML = '<option value="">Load saved…</option>' + list.map((g) => `<option value="${g.name}">${g.name}</option>`).join("");
  }
  refreshSavedList();
}

function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.style.display = "block";
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => (el.style.display = "none"), 3000);
}

main();
