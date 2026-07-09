// TF-Playground-style decision-boundary heatmap: blue<->orange diverging grid
// plus the actual sampled training points overplotted, colored by true label.

export class DecisionBoundary {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.points = []; // [{x,y,label}] in data-space (-6..6)
    this.range = [-6, 6];
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.min(this.canvas.parentElement.clientWidth, 320);
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._size = size;
  }

  setSamplePoints(points, range) {
    this.points = points;
    if (range) this.range = range;
  }

  draw(grid) {
    const ctx = this.ctx;
    const size = this._size;
    ctx.clearRect(0, 0, size, size);
    if (grid && grid.length) {
      const gridSize = grid.length;
      const cell = size / gridSize;
      for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
          const v = grid[row][col]; // 0..1 probability of class 1
          ctx.fillStyle = colorFor(v);
          // grid row 0 = smallest y (bottom in data space) -> flip vertically for screen
          ctx.fillRect(col * cell, size - (row + 1) * cell, cell + 1, cell + 1);
        }
      }
    }

    const [lo, hi] = this.range;
    for (const p of this.points) {
      const sx = ((p.x - lo) / (hi - lo)) * size;
      const sy = size - ((p.y - lo) / (hi - lo)) * size;
      ctx.beginPath();
      ctx.fillStyle = p.label === 1 ? "#38bdf8" : "#f59e0b";
      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function colorFor(v) {
  // v in [0,1]: 0 -> orange, 1 -> blue, matching TF Playground's diverging scale
  const orange = [245, 158, 11];
  const blue = [56, 189, 248];
  const bg = [10, 14, 39];
  const t = Math.abs(v - 0.5) * 2; // 0 at boundary -> full bg dark, 1 at extremes -> full color
  const base = v > 0.5 ? blue : orange;
  const r = bg[0] + (base[0] - bg[0]) * t;
  const g = bg[1] + (base[1] - bg[1]) * t;
  const b = bg[2] + (base[2] - bg[2]) * t;
  return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
}
