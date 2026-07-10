// Single-axis line chart with gridlines, baseline, current-value label, and a
// hover crosshair+tooltip. One measure per chart — never dual-axis (see
// dataviz skill: "two measures of different scale -> two charts").

export class MiniLineChart {
  constructor(canvas, { color, label, format = (v) => v.toFixed(3), maxPoints = 500 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.color = color || "#3987e5";
    this.label = label || "";
    this.format = format;
    this.maxPoints = maxPoints;
    this.points = []; // {x, y}
    this.hoverIdx = null;

    this._resize();
    window.addEventListener("resize", () => this._resize());
    canvas.addEventListener("mousemove", (e) => this._onHover(e));
    canvas.addEventListener("mouseleave", () => { this.hoverIdx = null; this.draw(); });
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this._w = rect.width;
    this._h = 120;
    this.canvas.width = this._w * dpr;
    this.canvas.height = this._h * dpr;
    this.canvas.style.height = `${this._h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  reset() {
    this.points = [];
    this.hoverIdx = null;
    this.draw();
  }

  push(x, y) {
    if (y === undefined || y === null) return;
    this.points.push({ x, y });
    if (this.points.length > this.maxPoints) this.points.shift();
    this.draw();
  }

  _onHover(e) {
    if (this.points.length < 2) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const idx = Math.round((px / this._w) * (this.points.length - 1));
    this.hoverIdx = Math.max(0, Math.min(this.points.length - 1, idx));
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const w = this._w;
    const h = this._h;
    const padL = 4, padR = 4, padT = 8, padB = 16;
    ctx.clearRect(0, 0, w, h);

    // baseline + gridlines (recessive)
    ctx.strokeStyle = "#2c2c2a";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 2; i++) {
      const gy = padT + ((h - padT - padB) / 2) * i;
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(w - padR, gy);
      ctx.stroke();
    }

    if (this.points.length < 2) {
      ctx.fillStyle = "#898781";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.fillText("waiting for training data…", padL, h / 2);
      return;
    }

    const ys = this.points.map((p) => p.y);
    const maxY = Math.max(...ys);
    const minY = Math.min(...ys, 0);
    const range = maxY - minY || 1;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const xAt = (i) => padL + (i / (this.points.length - 1)) * plotW;
    const yAt = (v) => padT + plotH - ((v - minY) / range) * plotH;

    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    this.points.forEach((p, i) => {
      const x = xAt(i), y = yAt(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // filled area under the line (subtle)
    ctx.lineTo(xAt(this.points.length - 1), padT + plotH);
    ctx.lineTo(xAt(0), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = this.color + "14";
    ctx.fill();

    // current value label (top-right)
    const last = this.points[this.points.length - 1];
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 11px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(this.format(last.y), w - padR, padT + 2);
    ctx.textAlign = "left";

    // hover crosshair + tooltip
    if (this.hoverIdx !== null && this.points[this.hoverIdx]) {
      const p = this.points[this.hoverIdx];
      const hx = xAt(this.hoverIdx), hy = yAt(p.y);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, padT);
      ctx.lineTo(hx, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(hx, hy, 3, 0, Math.PI * 2);
      ctx.fill();

      const text = `${this.label ? this.label + " " : ""}${this.format(p.y)}  ·  step ${p.x}`;
      ctx.font = "11px -apple-system, sans-serif";
      const tw = ctx.measureText(text).width + 12;
      let tx = hx + 8;
      if (tx + tw > w) tx = hx - tw - 8;
      ctx.fillStyle = "#1c1e22";
      ctx.strokeStyle = "rgba(255,255,255,0.15)";
      roundRect(ctx, tx, padT, tw, 20, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, tx + 6, padT + 14);
    }
  }
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
