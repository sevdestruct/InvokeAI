#!/usr/bin/env python3
"""Image-quality metrics for the performance arbiter (scripts/perf_arbiter.py).

Pure functions, no InvokeAI deps. Implements SSIM via scipy Gaussian windows (so we don't
need scikit-image), plus PSNR / mean-abs-diff, a Laplacian-sharpness delta (blur/artifact
proxy), face-region SSIM via the OpenCV Haar cascade (to localize where anatomy/faces
degrade), and an abs-diff heatmap for visual inspection.

Run directly for a self-test:  python scripts/arbiter_metrics.py
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

_LUMA = np.array([0.299, 0.587, 0.114])


def _gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64)[..., :3] @ _LUMA


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-pixel difference on the 0-255 scale."""
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    if mse <= 1e-9:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5, L: float = 255.0) -> tuple[float, np.ndarray]:
    """Gaussian-windowed SSIM on luma. Returns (mean_ssim, per-pixel ssim map)."""
    ga, gb = _gray(a), _gray(b)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a = gaussian_filter(ga, sigma)
    mu_b = gaussian_filter(gb, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = gaussian_filter(ga * ga, sigma) - mu_a2
    var_b = gaussian_filter(gb * gb, sigma) - mu_b2
    cov = gaussian_filter(ga * gb, sigma) - mu_ab
    smap = ((2 * mu_ab + c1) * (2 * cov + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
    return float(smap.mean()), smap


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian -- a standard focus/sharpness measure. Lower => blurrier."""
    g = _gray(img).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


_FACE_CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))


def detect_faces(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = _gray(img).astype(np.uint8)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [tuple(int(v) for v in f) for f in faces]


def face_region_ssim(a: np.ndarray, b: np.ndarray, smap: np.ndarray) -> float | None:
    """Mean SSIM inside faces detected in the reference image -- localizes facial/anatomy drift.
    Returns None when no face is detected."""
    faces = detect_faces(a)
    if not faces:
        return None
    vals = []
    for (x, y, w, h) in faces:
        region = smap[y : y + h, x : x + w]
        if region.size:
            vals.append(float(region.mean()))
    return float(np.mean(vals)) if vals else None


def save_diff_heatmap(a: np.ndarray, b: np.ndarray, path: str) -> None:
    """Write a JET-colormapped absolute-difference heatmap (bright = larger deviation)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    norm = np.clip(diff / max(diff.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(path, heat)


@dataclass
class QualityReport:
    mean_abs_diff_pct: float  # 0-100
    psnr_db: float
    ssim: float
    face_region_ssim: float | None
    n_faces: int
    sharpness_ref: float
    sharpness_var: float
    sharpness_ratio: float  # variant/ref; <1 => softer/blurrier than reference
    heatmap_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def compare(ref: np.ndarray, var: np.ndarray, heatmap_path: str | None = None) -> QualityReport:
    """Full quality comparison of a variant image against the reference (acceleration-off) image."""
    if ref.shape != var.shape:
        var = cv2.resize(var, (ref.shape[1], ref.shape[0]))
    mad = mean_abs_diff(ref, var)
    s, smap = ssim(ref, var)
    faces = detect_faces(ref)
    fr = face_region_ssim(ref, var, smap)
    sharp_ref = laplacian_sharpness(ref)
    sharp_var = laplacian_sharpness(var)
    if heatmap_path:
        save_diff_heatmap(ref, var, heatmap_path)
    return QualityReport(
        mean_abs_diff_pct=round(mad / 255.0 * 100.0, 3),
        psnr_db=round(psnr(ref, var), 2),
        ssim=round(s, 4),
        face_region_ssim=round(fr, 4) if fr is not None else None,
        n_faces=len(faces),
        sharpness_ref=round(sharp_ref, 1),
        sharpness_var=round(sharp_var, 1),
        sharpness_ratio=round(sharp_var / sharp_ref, 3) if sharp_ref > 1e-6 else 1.0,
        heatmap_path=heatmap_path,
    )


def _self_test() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    # identical -> ssim 1, psnr high, mad 0
    r = compare(img, img.copy())
    assert r.ssim > 0.999 and r.psnr_db > 50 and r.mean_abs_diff_pct < 0.01, r
    # add noise -> degraded
    noisy = np.clip(img.astype(np.int16) + rng.integers(-40, 40, img.shape), 0, 255).astype(np.uint8)
    r2 = compare(img, noisy)
    assert r2.ssim < r.ssim and r2.mean_abs_diff_pct > 1.0, r2
    # blur -> sharpness ratio < 1
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    r3 = compare(img, blurred)
    assert r3.sharpness_ratio < 1.0, r3
    print("arbiter_metrics self-test OK:")
    print("  identical:", r.ssim, r.psnr_db, r.mean_abs_diff_pct)
    print("  noisy    :", r2.ssim, r2.psnr_db, r2.mean_abs_diff_pct)
    print("  blurred  : sharpness_ratio", r3.sharpness_ratio)


if __name__ == "__main__":
    _self_test()
