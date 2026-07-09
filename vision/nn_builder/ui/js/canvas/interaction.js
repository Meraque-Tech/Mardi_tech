// Pointer drag-mode state machine (pan / node-drag / edge-drag), wheel zoom-to-cursor,
// modeled on agv_dashboard's js/map_edit/edit_main.js dragMode pattern and
// js/mapping/mapping.js's wheel-zoom-to-cursor math.

import { Viewport } from "./coordinateConverter.js";
import {
  render, hitTestNode, hitTestOutputPort, hitTestInputPort, outputPortScreenPos,
} from "./canvasRenderer.js";

export class NodeCanvasEditor {
  constructor(canvas, graphModel) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.graph = graphModel;
    this.viewport = new Viewport();
    this.selectedId = null;
    this.dragMode = "none"; // none | pan | node | edge
    this.dragNodeId = null;
    this.dragNodeOffset = { x: 0, y: 0 };
    this.dragEdgeSourceId = null;
    this.dragPointer = { x: 0, y: 0 };
    this.lastPointer = { x: 0, y: 0 };
    this.validationShapes = {};
    this.errorNodeIds = new Set();
    this.onSelect = () => {};

    this._resize();
    window.addEventListener("resize", () => this._resize());

    canvas.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this._onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this._onPointerUp(e));
    canvas.addEventListener("wheel", (e) => this._onWheel(e), { passive: false });
    window.addEventListener("keydown", (e) => this._onKeyDown(e));

    graphModel.onChange(() => this.render());
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  _screenPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  addNodeFromPalette(type, screenPos) {
    const world = this.viewport.screenToWorld(screenPos.x, screenPos.y);
    const id = this.graph.addNode(type, world);
    this.selectedId = id;
    this.onSelect(id);
    this.render();
    return id;
  }

  _onPointerDown(e) {
    const pos = this._screenPos(e);
    this.lastPointer = pos;

    const outPortNode = hitTestOutputPort(this.graph, this.viewport, pos.x, pos.y);
    if (outPortNode) {
      this.dragMode = "edge";
      this.dragEdgeSourceId = outPortNode.id;
      this.dragPointer = pos;
      return;
    }

    const node = hitTestNode(this.graph, this.viewport, pos.x, pos.y);
    if (node) {
      this.selectedId = node.id;
      this.onSelect(node.id);
      this.dragMode = "node";
      this.dragNodeId = node.id;
      const world = this.viewport.screenToWorld(pos.x, pos.y);
      this.dragNodeOffset = { x: world.x - node.position.x, y: world.y - node.position.y };
      this.render();
      return;
    }

    this.selectedId = null;
    this.onSelect(null);
    this.dragMode = "pan";
    this.canvas.classList.add("panning");
    this.render();
  }

  _onPointerMove(e) {
    const pos = this._screenPos(e);
    if (this.dragMode === "pan") {
      this.viewport.pan(pos.x - this.lastPointer.x, pos.y - this.lastPointer.y);
      this.lastPointer = pos;
      this.render();
    } else if (this.dragMode === "node" && this.dragNodeId) {
      const world = this.viewport.screenToWorld(pos.x, pos.y);
      this.graph.moveNode(this.dragNodeId, {
        x: world.x - this.dragNodeOffset.x,
        y: world.y - this.dragNodeOffset.y,
      });
    } else if (this.dragMode === "edge") {
      this.dragPointer = pos;
      this.render();
    }
  }

  _onPointerUp(e) {
    const pos = this._screenPos(e);
    if (this.dragMode === "edge" && this.dragEdgeSourceId) {
      const targetNode = hitTestInputPort(this.graph, this.viewport, pos.x, pos.y);
      if (targetNode) {
        const edgeId = this.graph.addEdge(this.dragEdgeSourceId, targetNode.id);
        if (!edgeId) this._toast("Cannot connect: input already occupied");
      }
    }
    this.dragMode = "none";
    this.dragNodeId = null;
    this.dragEdgeSourceId = null;
    this.canvas.classList.remove("panning");
    this.render();
  }

  _onWheel(e) {
    e.preventDefault();
    const pos = this._screenPos(e);
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    this.viewport.zoomAt(pos.x, pos.y, factor);
    this.render();
  }

  _onKeyDown(e) {
    if ((e.key === "Delete" || e.key === "Backspace") && this.selectedId) {
      if (document.activeElement && ["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
      this.graph.removeNode(this.selectedId);
      this.selectedId = null;
      this.onSelect(null);
    }
  }

  _toast(message) {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    el.style.display = "block";
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => (el.style.display = "none"), 2500);
  }

  setValidation(shapes, errorNodeIds) {
    this.validationShapes = shapes || {};
    this.errorNodeIds = errorNodeIds || new Set();
    this.render();
  }

  render() {
    let dragEdge = null;
    if (this.dragMode === "edge" && this.dragEdgeSourceId) {
      const srcNode = this.graph.nodes.get(this.dragEdgeSourceId);
      if (srcNode) {
        const p1 = outputPortScreenPos(this.viewport, srcNode);
        dragEdge = { x1: p1.x, y1: p1.y, x2: this.dragPointer.x, y2: this.dragPointer.y };
      }
    }
    render(this.ctx, this.canvas, this.viewport, this.graph, {
      selectedId: this.selectedId,
      dragEdge,
      shapes: this.validationShapes,
      errorNodeIds: this.errorNodeIds,
    });
  }
}
