// Screen<->world pan/zoom transform, modeled on agv_dashboard's
// js/map_edit/coordinateConverter.js and js/mapping/mapping.js MapViewer math.

export class Viewport {
  constructor() {
    this.offsetX = 0;
    this.offsetY = 0;
    this.scale = 1;
  }

  worldToScreen(x, y) {
    return { x: x * this.scale + this.offsetX, y: y * this.scale + this.offsetY };
  }

  screenToWorld(x, y) {
    return { x: (x - this.offsetX) / this.scale, y: (y - this.offsetY) / this.scale };
  }

  pan(dx, dy) {
    this.offsetX += dx;
    this.offsetY += dy;
  }

  zoomAt(screenX, screenY, factor) {
    const newScale = Math.min(2.5, Math.max(0.25, this.scale * factor));
    const actualFactor = newScale / this.scale;
    this.offsetX = screenX - (screenX - this.offsetX) * actualFactor;
    this.offsetY = screenY - (screenY - this.offsetY) * actualFactor;
    this.scale = newScale;
  }
}
