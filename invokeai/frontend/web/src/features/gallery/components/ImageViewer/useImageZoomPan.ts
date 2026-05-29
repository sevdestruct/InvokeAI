import type { CSSProperties, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { Transform } from './imageZoomPanMath';
import {
  clampScale,
  constrainTranslate,
  getFocalZoom,
  IDENTITY,
  MIN_SCALE,
  wheelDeltaToScaleFactor,
} from './imageZoomPanMath';

const DOUBLE_CLICK_SCALE = 2;
const ANIMATED_TRANSITION = 'transform 0.2s ease-out';

/**
 * Free, smooth zoom + pan for the pop-out image viewer.
 *
 * - Trackpad pinch (wheel + ctrlKey) zooms toward the cursor.
 * - Two-finger drag (plain wheel) pans when zoomed in.
 * - Mouse drag pans when zoomed in.
 * - Double-click toggles between fit and 2x (or resets when zoomed).
 * - Resets automatically when `resetKey` changes (i.e. a new image is shown).
 *
 * Attach `containerRef` to the (overflow-hidden) viewport element and spread
 * `contentStyle` onto the inner wrapper that holds the image.
 */
export const useImageZoomPan = (resetKey?: string) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState<Transform>(IDENTITY);
  const [isAnimating, setIsAnimating] = useState(false);

  // Mirror state in a ref so native (non-passive) listeners read fresh values.
  const transformRef = useRef<Transform>(IDENTITY);
  transformRef.current = transform;

  const apply = useCallback((next: Transform, animate: boolean) => {
    const rect = containerRef.current?.getBoundingClientRect();
    const constrained = rect ? constrainTranslate(next, rect.width, rect.height) : next;
    setIsAnimating(animate);
    setTransform(constrained);
  }, []);

  const reset = useCallback(() => {
    setIsAnimating(true);
    setTransform(IDENTITY);
  }, []);

  // Reset instantly when the displayed image changes.
  useEffect(() => {
    setIsAnimating(false);
    setTransform(IDENTITY);
  }, [resetKey]);

  // Native, non-passive wheel listener so we can preventDefault the browser's
  // page-level pinch zoom. React's onWheel is passive and cannot.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }
    const onWheel = (e: WheelEvent) => {
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const t = transformRef.current;

      if (e.ctrlKey) {
        // Trackpad pinch (or ctrl + wheel) => zoom toward the cursor.
        e.preventDefault();
        const factor = wheelDeltaToScaleFactor(e.deltaY);
        apply(getFocalZoom(t, t.scale * factor, px, py), false);
        return;
      }

      if (t.scale > MIN_SCALE) {
        // Two-finger pan while zoomed in.
        e.preventDefault();
        apply({ scale: t.scale, x: t.x - e.deltaX, y: t.y - e.deltaY }, false);
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [apply]);

  // Mouse / pointer drag-to-pan, active only when zoomed in.
  const dragState = useRef<{ pointerId: number; x: number; y: number } | null>(null);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    if (transformRef.current.scale <= MIN_SCALE || e.button !== 0) {
      return;
    }
    dragState.current = { pointerId: e.pointerId, x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const drag = dragState.current;
      if (!drag || drag.pointerId !== e.pointerId) {
        return;
      }
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      const t = transformRef.current;
      apply({ scale: t.scale, x: t.x + dx, y: t.y + dy }, false);
    },
    [apply]
  );

  const onPointerUp = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const drag = dragState.current;
    if (drag && drag.pointerId === e.pointerId) {
      dragState.current = null;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    }
  }, []);

  const onDoubleClick = useCallback(
    (e: ReactMouseEvent<HTMLElement>) => {
      const t = transformRef.current;
      if (t.scale > MIN_SCALE) {
        reset();
        return;
      }
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      apply(getFocalZoom(t, DOUBLE_CLICK_SCALE, e.clientX - rect.left, e.clientY - rect.top), true);
    },
    [apply, reset]
  );

  const isZoomed = transform.scale > MIN_SCALE;

  const containerStyle = useMemo<CSSProperties>(
    () => ({
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
      touchAction: 'none',
      outline: 'none',
      cursor: isZoomed ? 'grab' : 'default',
    }),
    [isZoomed]
  );

  const contentStyle = useMemo<CSSProperties>(
    () => ({
      transform: `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${clampScale(transform.scale)})`,
      transformOrigin: '0 0',
      transition: isAnimating ? ANIMATED_TRANSITION : 'none',
      willChange: 'transform',
      width: '100%',
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }),
    [transform.x, transform.y, transform.scale, isAnimating]
  );

  return {
    containerRef,
    containerStyle,
    contentStyle,
    isZoomed,
    scale: transform.scale,
    reset,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onDoubleClick,
  };
};
