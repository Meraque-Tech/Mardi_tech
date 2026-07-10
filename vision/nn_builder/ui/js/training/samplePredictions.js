// Renders a small grid of held-out images with true vs predicted label --
// the "let me actually see the classified digits" view, not just numbers.

export class SamplePredictions {
  constructor(container) {
    this.container = container;
  }

  draw(samples) {
    this.container.innerHTML = "";
    if (!samples || !samples.length) {
      this.container.innerHTML = '<div class="empty-hint">waiting for evaluation…</div>';
      return;
    }
    for (const s of samples) {
      const correct = s.true === s.pred;
      const cell = document.createElement("div");
      cell.className = "sample-cell";
      cell.style.borderColor = correct ? "var(--status-good)" : "var(--status-critical)";

      const canvas = document.createElement("canvas");
      const h = s.pixels.length, w = s.pixels[0].length;
      canvas.width = w;
      canvas.height = h;
      canvas.className = "sample-canvas";
      const ctx = canvas.getContext("2d");
      const imgData = ctx.createImageData(w, h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const v = s.pixels[y][x];
          const idx = (y * w + x) * 4;
          imgData.data[idx] = v;
          imgData.data[idx + 1] = v;
          imgData.data[idx + 2] = v;
          imgData.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);

      const label = document.createElement("div");
      label.className = "sample-label";
      label.style.color = correct ? "var(--status-good)" : "var(--status-critical)";
      label.textContent = correct ? `${s.pred}` : `${s.pred} (≠${s.true})`;

      cell.appendChild(canvas);
      cell.appendChild(label);
      this.container.appendChild(cell);
    }
  }
}
