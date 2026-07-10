// ROC curve: FPR (x) vs TPR (y), both 0..1, plus the diagonal random-chance
// baseline. Identity encoding (one line per class) -> fixed categorical hues,
// legend always present for >=2 curves (dataviz skill: color-formula.md).

const SERIES_COLORS = ["#3987e5", "#199e70", "#c98500", "#4caf3f", "#9085e9", "#d55181"];

export class RocCurve {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.min(this.canvas.parentElement.clientWidth, 280);
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._size = size;
    this.draw(this.data);
  }

  draw(data) {
    this.data = data;
    const ctx = this.ctx;
    const size = this._size;
    const pad = 22;
    ctx.clearRect(0, 0, size, size);

    if (!data || !data.curves || !data.curves.length) {
      ctx.fillStyle = "#898781";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.fillText("waiting for evaluation…", 4, size / 2);
      return;
    }

    const plot = size - pad * 2;
    const toX = (fpr) => pad + fpr * plot;
    const toY = (tpr) => pad + (1 - tpr) * plot;

    // axes
    ctx.strokeStyle = "#2c2c2a";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, plot, plot);

    // diagonal random-chance baseline
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#52514e";
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(0));
    ctx.lineTo(toX(1), toY(1));
    ctx.stroke();
    ctx.setLineDash([]);

    // axis labels
    ctx.fillStyle = "#64645f";
    ctx.font = "9px -apple-system, sans-serif";
    ctx.fillText("FPR", size - pad - 18, size - 6);
    ctx.save();
    ctx.translate(8, pad + 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("TPR", 0, 0);
    ctx.restore();

    // curves
    data.curves.forEach((curve, i) => {
      const color = SERIES_COLORS[i % SERIES_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.fpr.forEach((fpr, j) => {
        const x = toX(fpr), y = toY(curve.tpr[j]);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // legend (always present for >=2 series; single series is self-explanatory)
    if (data.curves.length > 1) {
      let ly = pad + 4;
      data.curves.forEach((curve, i) => {
        const color = SERIES_COLORS[i % SERIES_COLORS.length];
        ctx.fillStyle = color;
        ctx.fillRect(size - pad - 8, ly, 8, 8);
        ctx.fillStyle = "#c3c2b7";
        ctx.font = "9px -apple-system, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(curve.label, size - pad - 12, ly + 8);
        ctx.textAlign = "left";
        ly += 12;
      });
    }
  }
}
