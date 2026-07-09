// Canvas line chart: loss (+ optional accuracy) vs step, ring-buffered.

export class LossChart {
  constructor(canvas, maxPoints = 500) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.maxPoints = maxPoints;
    this.points = []; // {step, loss, accuracy}
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = 160 * dpr;
    this.canvas.style.height = "160px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  reset() {
    this.points = [];
    this.draw();
  }

  push(point) {
    this.points.push(point);
    if (this.points.length > this.maxPoints) this.points.shift();
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = 160;
    ctx.clearRect(0, 0, w, h);
    if (this.points.length < 2) return;

    const losses = this.points.map((p) => p.loss);
    const maxLoss = Math.max(...losses, 1e-6);
    const minLoss = Math.min(...losses, 0);

    const plot = (getter, color, range) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      this.points.forEach((p, i) => {
        const v = getter(p);
        if (v === undefined || v === null) return;
        const x = (i / (this.points.length - 1)) * w;
        const norm = (v - range[0]) / (range[1] - range[0] || 1);
        const y = h - norm * (h - 10) - 5;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    plot((p) => p.loss, "#38bdf8", [minLoss, maxLoss]);
    plot((p) => p.accuracy, "#34d399", [0, 1]);
  }
}
