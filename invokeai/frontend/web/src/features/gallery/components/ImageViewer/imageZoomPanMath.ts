/**
 * Pure math for the image viewer's free zoom + pan.
 *
 * All coordinates are relative to the container's top-left corner, and the
 * transformed element uses `transform-origin: 0 0`. This keeps the focal-point
 * zoom math simple: screen = translate + scale * contentPoint.
 */

export type Transform = {
  scale: number;
  x: number;
  y: number;
};

export const MIN_SCALE = 1;
export const MAX_SCALE = 20;
export const IDENTITY: Transform = { scale: 1, x: 0, y: 0 };

export const clampScale = (scale: number, min = MIN_SCALE, max = MAX_SCALE): number =>
  Math.min(max, Math.max(min, scale));

/**
 * Zoom to `newScale` while keeping the content point under (pointerX, pointerY)
 * fixed on screen. Pointer coords are relative to the container's top-left.
 */
export const getFocalZoom = (current: Transform, newScale: number, pointerX: number, pointerY: number): Transform => {
  const scale = clampScale(newScale);
  const ratio = scale / current.scale;
  return {
    scale,
    x: pointerX - (pointerX - current.x) * ratio,
    y: pointerY - (pointerY - current.y) * ratio,
  };
};

/**
 * Constrain the translation so the scaled content always covers the container
 * (no empty gutters), and recenters exactly when scale === 1.
 */
export const constrainTranslate = (t: Transform, width: number, height: number): Transform => {
  const minX = width * (1 - t.scale);
  const minY = height * (1 - t.scale);
  return {
    scale: t.scale,
    x: Math.min(0, Math.max(minX, t.x)),
    y: Math.min(0, Math.max(minY, t.y)),
  };
};

/**
 * Convert a wheel `deltaY` into a multiplicative zoom factor. Trackpad pinch is
 * delivered as wheel events with `ctrlKey` set; smaller deltas => finer zoom.
 */
export const wheelDeltaToScaleFactor = (deltaY: number, sensitivity = 0.01): number => Math.exp(-deltaY * sensitivity);
