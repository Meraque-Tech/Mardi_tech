import { getSpec } from "../nodes/node_defs.js";

export const NODE_WIDTH = 160;
export const NODE_HEIGHT = 48;
export const PORT_RADIUS = 6;

const FAMILY_COLORS = {
  struct: "#94a3b8",
  cnn: "#38bdf8",
  ann: "#a78bfa",
  rnn: "#34d399",
  transformer: "#f59e0b",
  activation: "#f472b6",
  config: "#ef4444",
};

function nodePortCount(node) {
  const spec = getSpec(node.type);
  return spec && spec.inputs === -1 ? Math.max(2, portArity(node)) : 1;
}

// count of *connected* input ports (for multi-input nodes we still draw at least min 2 slots)
function portArity(node) {
  return Math.max(2, (node._incomingCount || 0));
}

export function inputPortScreenPos(viewport, node, portIndex, totalPorts) {
  const h = NODE_HEIGHT;
  const step = h / (totalPorts + 1);
  const wx = node.position.x;
  const wy = node.position.y + step * (portIndex + 1);
  return viewport.worldToScreen(wx, wy);
}

export function outputPortScreenPos(viewport, node) {
  const wx = node.position.x + NODE_WIDTH;
  const wy = node.position.y + NODE_HEIGHT / 2;
  return viewport.worldToScreen(wx, wy);
}

export function nodeScreenRect(viewport, node) {
  const topLeft = viewport.worldToScreen(node.position.x, node.position.y);
  return { x: topLeft.x, y: topLeft.y, w: NODE_WIDTH * viewport.scale, h: NODE_HEIGHT * viewport.scale };
}

export function hitTestNode(graphModel, viewport, screenX, screenY) {
  for (const node of [...graphModel.nodes.values()].reverse()) {
    const r = nodeScreenRect(viewport, node);
    if (screenX >= r.x && screenX <= r.x + r.w && screenY >= r.y && screenY <= r.y + r.h) return node;
  }
  return null;
}

export function hitTestOutputPort(graphModel, viewport, screenX, screenY) {
  for (const node of graphModel.nodes.values()) {
    const p = outputPortScreenPos(viewport, node);
    if (Math.hypot(p.x - screenX, p.y - screenY) <= PORT_RADIUS + 3) return node;
  }
  return null;
}

export function hitTestInputPort(graphModel, viewport, screenX, screenY) {
  for (const node of graphModel.nodes.values()) {
    const incoming = graphModel.incomingEdges(node.id).length;
    const total = Math.max(1, nodePortCount({ ...node, _incomingCount: incoming }));
    for (let i = 0; i < total; i++) {
      const p = inputPortScreenPos(viewport, node, i, total);
      if (Math.hypot(p.x - screenX, p.y - screenY) <= PORT_RADIUS + 3) return node;
    }
  }
  return null;
}

function drawBezier(ctx, x1, y1, x2, y2, color) {
  const dx = Math.max(40, Math.abs(x2 - x1) / 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(x1 + dx, y1, x2 - dx, y2, x2, y2);
  ctx.stroke();
}

export function render(ctx, canvas, viewport, graphModel, opts = {}) {
  const { selectedId = null, dragEdge = null, shapes = {}, errorNodeIds = new Set() } = opts;
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // edges
  for (const e of graphModel.edges.values()) {
    const srcNode = graphModel.nodes.get(e.source.node);
    const tgtNode = graphModel.nodes.get(e.target.node);
    if (!srcNode || !tgtNode) continue;
    const p1 = outputPortScreenPos(viewport, srcNode);
    const incoming = graphModel.incomingEdges(tgtNode.id);
    const idx = incoming.findIndex((x) => x.id === e.id);
    const total = Math.max(1, nodePortCount({ ...tgtNode, _incomingCount: incoming.length }));
    const p2 = inputPortScreenPos(viewport, tgtNode, idx, total);
    drawBezier(ctx, p1.x, p1.y, p2.x, p2.y, "rgba(56,189,248,0.55)");
  }

  if (dragEdge) {
    drawBezier(ctx, dragEdge.x1, dragEdge.y1, dragEdge.x2, dragEdge.y2, "rgba(255,255,255,0.5)");
  }

  // nodes
  for (const node of graphModel.nodes.values()) {
    const spec = getSpec(node.type) || {};
    const color = FAMILY_COLORS[spec.family] || "#94a3b8";
    const r = nodeScreenRect(viewport, node);

    ctx.fillStyle = "#151933";
    ctx.strokeStyle = node.id === selectedId ? "#00d4ff" : "rgba(255,255,255,0.12)";
    ctx.lineWidth = node.id === selectedId ? 2 : 1;
    roundRect(ctx, r.x, r.y, r.w, r.h, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    roundRect(ctx, r.x, r.y, r.w, 5 * viewport.scale, 4);
    ctx.fill();

    ctx.fillStyle = "#e0e0e0";
    ctx.font = `${12 * viewport.scale}px -apple-system, Segoe UI, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText(spec.label || node.type, r.x + 10 * viewport.scale, r.y + r.h / 2, r.w - 20 * viewport.scale);

    if (errorNodeIds.has(node.id)) {
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(r.x + r.w - 8 * viewport.scale, r.y + 8 * viewport.scale, 4 * viewport.scale, 0, Math.PI * 2);
      ctx.fill();
    }

    const shape = shapes[node.id];
    if (shape) {
      ctx.fillStyle = "#64748b";
      ctx.font = `${10 * viewport.scale}px monospace`;
      ctx.fillText(`[${shape.join(",")}]`, r.x, r.y + r.h + 12 * viewport.scale);
    }

    // ports
    const incoming = graphModel.incomingEdges(node.id).length;
    const total = Math.max(1, nodePortCount({ ...node, _incomingCount: incoming }));
    if ((spec.inputs || 0) !== 0) {
      for (let i = 0; i < total; i++) {
        const p = inputPortScreenPos(viewport, node, i, total);
        drawPort(ctx, p, color);
      }
    }
    if ((spec.outputs || 0) !== 0) {
      drawPort(ctx, outputPortScreenPos(viewport, node), color);
    }
  }

  ctx.restore();
}

function drawPort(ctx, p, color) {
  ctx.fillStyle = "#0a0e27";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, PORT_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
