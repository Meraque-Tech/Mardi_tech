import { getSpec } from "../nodes/node_defs.js";

export class Inspector {
  constructor(container, graphModel, onChange) {
    this.container = container;
    this.graph = graphModel;
    this.onChange = onChange || (() => {});
    this.currentId = null;
  }

  showEmpty() {
    this.currentId = null;
    this.container.innerHTML = '<div class="empty-hint">Select a node to edit its parameters.</div>';
  }

  show(nodeId) {
    this.currentId = nodeId;
    const node = this.graph.nodes.get(nodeId);
    if (!node) return this.showEmpty();
    const spec = getSpec(node.type);
    if (!spec) return this.showEmpty();

    const rows = spec.params
      .map((p) => this._renderParam(node, p))
      .join("");

    this.container.innerHTML = `
      <h3>${spec.label}</h3>
      <div class="node-id-hint" style="color:#475569;font-size:0.7rem;margin-bottom:0.6rem;">${node.id}</div>
      ${rows || '<div class="empty-hint">No parameters.</div>'}
    `;

    for (const p of spec.params) {
      const el = this.container.querySelector(`[data-param="${p.name}"]`);
      if (!el) continue;
      el.addEventListener("change", () => this._commit(node, p, el));
      el.addEventListener("input", () => this._commit(node, p, el));
    }
  }

  _renderParam(node, p) {
    const value = node.params[p.name];
    const id = `param-${p.name}`;
    if (p.type === "bool") {
      return `<div class="param-row"><label><input type="checkbox" data-param="${p.name}" id="${id}" ${value ? "checked" : ""}/> ${p.name}</label></div>`;
    }
    if (p.type === "enum") {
      const opts = (p.options || []).map((o) => `<option value="${o}" ${o === value ? "selected" : ""}>${o}</option>`).join("");
      return `<div class="param-row"><label for="${id}">${p.name}</label><select data-param="${p.name}" id="${id}">${opts}</select></div>`;
    }
    if (p.type === "list_int") {
      return `<div class="param-row"><label for="${id}">${p.name}</label><input type="text" data-param="${p.name}" id="${id}" value="${(value || []).join(",")}"/></div>`;
    }
    const step = p.type === "float" ? "any" : "1";
    return `<div class="param-row"><label for="${id}">${p.name}</label><input type="number" step="${step}" min="${p.min ?? ""}" max="${p.max ?? ""}" data-param="${p.name}" id="${id}" value="${value}"/></div>`;
  }

  _commit(node, p, el) {
    let value;
    if (p.type === "bool") value = el.checked;
    else if (p.type === "enum") value = el.value;
    else if (p.type === "list_int") value = el.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n));
    else if (p.type === "float") value = parseFloat(el.value);
    else value = parseInt(el.value, 10);
    this.graph.updateParams(node.id, { [p.name]: value });
    this.onChange();
  }
}
