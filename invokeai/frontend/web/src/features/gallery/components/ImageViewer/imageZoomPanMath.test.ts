import { describe, expect, it } from 'vitest';

import type { Transform } from './imageZoomPanMath';
import {
  clampScale,
  constrainTranslate,
  getFocalZoom,
  IDENTITY,
  MAX_SCALE,
  MIN_SCALE,
  wheelDeltaToScaleFactor,
} from './imageZoomPanMath';

describe('clampScale', () => {
  it('clamps to the [MIN_SCALE, MAX_SCALE] range', () => {
    expect(clampScale(0.1)).toBe(MIN_SCALE);
    expect(clampScale(1000)).toBe(MAX_SCALE);
    expect(clampScale(3)).toBe(3);
  });
});

describe('getFocalZoom', () => {
  it('keeps the content point under the cursor fixed', () => {
    const pointerX = 100;
    const pointerY = 50;
    // Content point under the cursor before zooming (identity => 1:1).
    const contentBefore = {
      x: (pointerX - IDENTITY.x) / IDENTITY.scale,
      y: (pointerY - IDENTITY.y) / IDENTITY.scale,
    };

    const next = getFocalZoom(IDENTITY, 2, pointerX, pointerY);

    // Where that same content point lands after zooming.
    const screenAfter = {
      x: next.x + next.scale * contentBefore.x,
      y: next.y + next.scale * contentBefore.y,
    };
    expect(next.scale).toBe(2);
    expect(screenAfter.x).toBeCloseTo(pointerX);
    expect(screenAfter.y).toBeCloseTo(pointerY);
  });

  it('clamps the resulting scale', () => {
    expect(getFocalZoom(IDENTITY, 9999, 0, 0).scale).toBe(MAX_SCALE);
    expect(getFocalZoom({ scale: 2, x: -10, y: -10 }, 0.001, 0, 0).scale).toBe(MIN_SCALE);
  });
});

describe('constrainTranslate', () => {
  const width = 200;
  const height = 100;

  it('recenters when not zoomed', () => {
    const result = constrainTranslate({ scale: 1, x: 73, y: -42 }, width, height);
    expect(result).toEqual({ scale: 1, x: 0, y: 0 });
  });

  it('keeps the scaled content covering the container', () => {
    const t: Transform = { scale: 2, x: 50, y: 50 };
    const result = constrainTranslate(t, width, height);
    // Upper bound is 0, lower bound is width/height * (1 - scale).
    expect(result.x).toBe(0);
    expect(result.y).toBe(0);

    const tooFar = constrainTranslate({ scale: 2, x: -9999, y: -9999 }, width, height);
    expect(tooFar.x).toBe(width * (1 - 2));
    expect(tooFar.y).toBe(height * (1 - 2));
  });
});

describe('wheelDeltaToScaleFactor', () => {
  it('zooms in on negative delta and out on positive delta', () => {
    expect(wheelDeltaToScaleFactor(-10)).toBeGreaterThan(1);
    expect(wheelDeltaToScaleFactor(10)).toBeLessThan(1);
    expect(wheelDeltaToScaleFactor(0)).toBe(1);
  });
});
