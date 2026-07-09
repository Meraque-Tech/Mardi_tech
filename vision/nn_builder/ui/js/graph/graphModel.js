import { isMultiInput, defaultParams } from "../nodes/node_defs.js";

export class GraphModel {
  constructor() {
    this.nodes = new Map(); // id -> {id, type, params, position}
    this.edges = new Map(); // id -> {id, source:{node,port}, target:{node,port}}
    this._nodeCounter = 0;
    this._edgeCounter = 0;
    this._listeners = [];
    this.meta = { name: "untitled" };
  }

  onChange(cb) {
    this._listeners.push(cb);
  }

  _notify() {
    for (const cb of this._listeners) cb();
  }

  addNode(type, position) {
    const id = `n${++this._nodeCounter}`;
    this.nodes.set(id, { id, type, params: defaultParams(type), position: { ...position } });
    this._notify();
    return id;
  }

  removeNode(id) {
    this.nodes.delete(id);
    for (const [eid, e] of this.edges) {
      if (e.source.node === id || e.target.node === id) this.edges.delete(eid);
    }
    this._notify();
  }

  moveNode(id, position) {
    const n = this.nodes.get(id);
    if (!n) return;
    n.position = position;
    this._notify();
  }

  updateParams(id, params) {
    const n = this.nodes.get(id);
    if (!n) return;
    n.params = { ...n.params, ...params };
    this._notify();
  }

  incomingEdges(nodeId) {
    return [...this.edges.values()].filter((e) => e.target.node === nodeId);
  }

  outgoingEdges(nodeId) {
    return [...this.edges.values()].filter((e) => e.source.node === nodeId);
  }

  canConnect(sourceId, targetId) {
    if (sourceId === targetId) return false;
    const target = this.nodes.get(targetId);
    if (!target) return false;
    if (!isMultiInput(target.type) && this.incomingEdges(targetId).length >= 1) return false;
    return true;
  }

  addEdge(sourceId, targetId) {
    if (!this.canConnect(sourceId, targetId)) return null;
    const target = this.nodes.get(targetId);
    const multi = isMultiInput(target.type);
    const targetPort = multi ? `in${this.incomingEdges(targetId).length}` : "in";
    const id = `e${++this._edgeCounter}`;
    this.edges.set(id, {
      id,
      source: { node: sourceId, port: "out" },
      target: { node: targetId, port: targetPort },
    });
    this._notify();
    return id;
  }

  removeEdge(id) {
    this.edges.delete(id);
    this._notify();
  }

  clear() {
    this.nodes.clear();
    this.edges.clear();
    this._nodeCounter = 0;
    this._edgeCounter = 0;
    this._notify();
  }

  toJSON() {
    return {
      id: this.meta.id || `graph_${Date.now()}`,
      meta: { ...this.meta },
      nodes: [...this.nodes.values()].map((n) => ({ ...n, params: { ...n.params } })),
      edges: [...this.edges.values()].map((e) => ({ ...e })),
    };
  }

  fromJSON(d) {
    this.clear();
    this.meta = d.meta || { name: "untitled" };
    this.meta.id = d.id;
    let maxNode = 0;
    let maxEdge = 0;
    for (const n of d.nodes || []) {
      this.nodes.set(n.id, { ...n });
      const m = /^n(\d+)$/.exec(n.id);
      if (m) maxNode = Math.max(maxNode, parseInt(m[1], 10));
    }
    for (const e of d.edges || []) {
      this.edges.set(e.id, { ...e });
      const m = /^e(\d+)$/.exec(e.id);
      if (m) maxEdge = Math.max(maxEdge, parseInt(m[1], 10));
    }
    this._nodeCounter = maxNode;
    this._edgeCounter = maxEdge;
    this._notify();
  }
}
