import { getSpec } from "../nodes/node_defs.js";

export const NODE_WIDTH = 160;
export const NODE_HEIGHT = 48;
export const PORT_RADIUS = 6;

// Reads the validated categorical palette straight from CSS custom properties
// (defined once in css/style.css) so colors have a single source of truth.
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

let _familyColors = null;
function familyColors() {
  if (_familyColors) return _familyColors;
  _familyColors = {
    struct: cssVar("--fam-struct", "#3987e5"),
    cnn: cssVar("--fam-cnn", "#199e70"),
    ann: cssVar("--fam-ann", "#c98500"),
    rnn: cssVar("--fam-rnn", "#4caf3f"),
    transformer: cssVar("--fam-transformer", "#9085e9"),
    activation: cssVar("--fam-activation", "#d55181"),
    config: cssVar("--fam-config", "#d95926"),
  };
  return _familyColors;
}

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

  const colors = familyColors();

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
    drawBezier(ctx, p1.x, p1.y, p2.x, p2.y, "rgba(90,160,239,0.45)");
  }

  if (dragEdge) {
    drawBezier(ctx, dragEdge.x1, dragEdge.y1, dragEdge.x2, dragEdge.y2, "rgba(255,255,255,0.45)");
  }

  // nodes
  for (const node of graphModel.nodes.values()) {
    const spec = getSpec(node.type) || {};
    const color = colors[spec.family] || colors.struct;
    const r = nodeScreenRect(viewport, node);
    const selected = node.id === selectedId;

    if (selected) {
      ctx.save();
      ctx.shadowColor = "rgba(90,160,239,0.55)";
      ctx.shadowBlur = 14 * viewport.scale;
    }

    ctx.fillStyle = "#1c1e22";
    ctx.strokeStyle = selected ? "#5aa0ef" : "rgba(255,255,255,0.12)";
    ctx.lineWidth = selected ? 1.5 : 1;
    roundRect(ctx, r.x, r.y, r.w, r.h, 9 * viewport.scale);
    ctx.fill();
    if (selected) ctx.restore();
    ctx.stroke();

    // family accent bar (rounded top only)
    ctx.save();
    roundRectClip(ctx, r.x, r.y, r.w, r.h, 9 * viewport.scale);
    ctx.fillStyle = color;
    ctx.fillRect(r.x, r.y, r.w, 4 * viewport.scale);
    ctx.restore();

    if (errorNodeIds.has(node.id)) {
      ctx.strokeStyle = "#e35b5b";
      ctx.lineWidth = 1.5;
      roundRect(ctx, r.x, r.y, r.w, r.h, 9 * viewport.scale);
      ctx.stroke();
    }

    ctx.fillStyle = "#ffffff";
    ctx.font = `600 ${12 * viewport.scale}px -apple-system, Segoe UI, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText(spec.label || node.type, r.x + 11 * viewport.scale, r.y + r.h / 2 + 2 * viewport.scale, r.w - 22 * viewport.scale);

    if (errorNodeIds.has(node.id)) {
      ctx.fillStyle = "#e35b5b";
      ctx.beginPath();
      ctx.arc(r.x + r.w - 9 * viewport.scale, r.y + 9 * viewport.scale, 3.5 * viewport.scale, 0, Math.PI * 2);
      ctx.fill();
    }

    const shape = shapes[node.id];
    if (shape) {
      ctx.fillStyle = "#898781";
      ctx.font = `${9.5 * viewport.scale}px "SF Mono", Consolas, monospace`;
      ctx.fillText(`[${shape.join(", ")}]`, r.x, r.y + r.h + 13 * viewport.scale);
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

function roundRectClip(ctx, x, y, w, h, r) {
  roundRect(ctx, x, y, w, h, r);
  ctx.clip();
}

function drawPort(ctx, p, color) {
  ctx.fillStyle = "#16171a";
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
