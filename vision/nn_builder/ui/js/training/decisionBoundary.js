// TF-Playground-style decision-boundary heatmap. Uses the validated diverging
// pair (blue <-> red, neutral gray midpoint) rather than an arbitrary rainbow —
// see dataviz skill: "diverging = two hues + a neutral gray midpoint".

const POLE_A = [230, 91, 91]; // red  — class 0
const POLE_B = [57, 135, 229]; // blue — class 1
const MIDPOINT = [26, 27, 30]; // neutral dark-surface gray

export class DecisionBoundary {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.points = []; // [{x,y,label}] in data-space
    this.range = [-6, 6];
    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.min(this.canvas.parentElement.clientWidth, 300);
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._size = size;
    this.draw(this._lastGrid);
  }

  setSamplePoints(points, range) {
    this.points = points;
    if (range) this.range = range;
  }

  draw(grid) {
    this._lastGrid = grid;
    const ctx = this.ctx;
    const size = this._size;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, size, size);

    if (grid && grid.length) {
      const gridSize = grid.length;
      const cell = size / gridSize;
      for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
          ctx.fillStyle = colorFor(grid[row][col]);
          ctx.fillRect(col * cell, size - (row + 1) * cell, cell + 1, cell + 1);
        }
      }
    }

    const [lo, hi] = this.range;
    for (const p of this.points) {
      const sx = ((p.x - lo) / (hi - lo)) * size;
      const sy = size - ((p.y - lo) / (hi - lo)) * size;
      ctx.beginPath();
      ctx.fillStyle = p.label === 1 ? "rgb(57,135,229)" : "rgb(230,91,91)";
      ctx.strokeStyle = "rgba(13,13,13,0.6)";
      ctx.lineWidth = 1;
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function colorFor(v) {
  // v in [0,1]: 0 -> red pole, 0.5 -> neutral midpoint, 1 -> blue pole
  const t = Math.min(1, Math.abs(v - 0.5) * 2);
  const pole = v >= 0.5 ? POLE_B : POLE_A;
  const r = MIDPOINT[0] + (pole[0] - MIDPOINT[0]) * t;
  const g = MIDPOINT[1] + (pole[1] - MIDPOINT[1]) * t;
  const b = MIDPOINT[2] + (pole[2] - MIDPOINT[2]) * t;
  return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
}
