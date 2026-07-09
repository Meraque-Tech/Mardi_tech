// Fetches the node catalog from the backend once and caches it.
// The palette, inspector, and graph model all read from this single object —
// no param schema is ever hand-duplicated in JS.

let _catalog = null;

// Minimal local fallback so the editor is usable even before the first
// successful /api/catalog fetch (or if opened as a static file during dev).
const FALLBACK_CATALOG = {
  nodes: {
    input: { family: "struct", label: "Input", inputs: 0, outputs: 1,
      params: [{ name: "shape", type: "list_int", default: [2] },
               { name: "input_kind", type: "enum", default: "tensor2d", options: ["tensor2d", "image", "sequence"] }] },
    output: { family: "struct", label: "Output", inputs: 1, outputs: 0,
      params: [{ name: "task", type: "enum", default: "classification", options: ["classification", "regression"] }] },
    linear: { family: "ann", label: "Linear", inputs: 1, outputs: 1,
      params: [{ name: "out_features", type: "int", default: 8, min: 1, max: 4096 },
               { name: "bias", type: "bool", default: true }] },
    relu: { family: "activation", label: "ReLU", inputs: 1, outputs: 1, params: [] },
  },
  families: { struct: ["input", "output"], ann: ["linear"], activation: ["relu"] },
};

export async function loadCatalog() {
  if (_catalog) return _catalog;
  try {
    const res = await fetch("/api/catalog");
    _catalog = await res.json();
  } catch (e) {
    console.warn("Falling back to local catalog stub:", e);
    _catalog = FALLBACK_CATALOG;
  }
  return _catalog;
}

export function getCatalog() {
  return _catalog || FALLBACK_CATALOG;
}

export function getSpec(type) {
  return getCatalog().nodes[type];
}

export function isMultiInput(type) {
  const spec = getSpec(type);
  return !!spec && spec.inputs === -1;
}

export function defaultParams(type) {
  const spec = getSpec(type);
  if (!spec) return {};
  const out = {};
  for (const p of spec.params) out[p.name] = p.default;
  return out;
}

export function families() {
  return getCatalog().families || {};
}
