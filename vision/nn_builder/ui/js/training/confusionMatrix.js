// Confusion-matrix heatmap: magnitude encoding, so single hue light->dark
// (sequential), never a rainbow — see dataviz skill color-formula.md.

const HUE = [57, 135, 229]; // accent blue

export class ConfusionMatrix {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.matrix = null;
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
    this.draw(this.matrix);
  }

  draw(matrix) {
    this.matrix = matrix;
    const ctx = this.ctx;
    const size = this._size;
    ctx.clearRect(0, 0, size, size);
    if (!matrix || !matrix.length) {
      ctx.fillStyle = "#898781";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.fillText("waiting for evaluation…", 4, size / 2);
      return;
    }

    const n = matrix.length;
    const labelW = n > 12 ? 16 : 22;
    const gridSize = size - labelW;
    const cell = gridSize / n;
    const maxV = Math.max(1, ...matrix.flat());

    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const v = matrix[row][col];
        const t = v / maxV;
        ctx.fillStyle = `rgba(${HUE[0]}, ${HUE[1]}, ${HUE[2]}, ${0.08 + t * 0.87})`;
        ctx.fillRect(labelW + col * cell, labelW + row * cell, cell - 1, cell - 1);

        if (n <= 12 && cell > 14) {
          ctx.fillStyle = t > 0.5 ? "#0d0d0d" : "#c3c2b7";
          ctx.font = `${Math.min(10, cell * 0.35)}px -apple-system, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(v), labelW + col * cell + cell / 2, labelW + row * cell + cell / 2);
        }
      }
    }

    if (n <= 12) {
      ctx.fillStyle = "#898781";
      ctx.font = "9px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let col = 0; col < n; col++) {
        ctx.fillText(String(col), labelW + col * cell + cell / 2, labelW / 2);
      }
      ctx.textAlign = "right";
      for (let row = 0; row < n; row++) {
        ctx.fillText(String(row), labelW - 4, labelW + row * cell + cell / 2);
      }
    }

    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }
}
