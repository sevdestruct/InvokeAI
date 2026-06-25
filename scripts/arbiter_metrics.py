#!/usr/bin/env python3
"""Image-quality metrics for the performance arbiter (scripts/perf_arbiter.py).

Pure functions, no InvokeAI deps. Implements SSIM via scipy Gaussian windows (so we don't
need scikit-image), plus PSNR / mean-abs-diff, a Laplacian-sharpness delta (blur/artifact
proxy), face-region SSIM via the OpenCV Haar cascade (to localize where anatomy/faces
degrade), and an abs-diff heatmap for visual inspection.

Run directly for a self-test:  python scripts/arbiter_metrics.py
"""

from __future__ import annotations

import functools
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


def gmsd(a: np.ndarray, b: np.ndarray, c: float = 170.0) -> tuple[float, float]:
    """Gradient Magnitude Similarity (mean) and Deviation. GMSD is a well-validated full-reference
    quality metric (Xue et al. 2014) that is far more sensitive than SSIM to TEXTURE/EDGE/fine-detail
    loss -- exactly the skin-texture/realism differences a perceptual eyeball glosses over.

    Returns (gmsm, gmsd): gmsm (mean gradient-similarity, higher=closer) and gmsd (std, the headline
    distortion score, higher=more texture distortion). Computed on 2x-average-pooled luma with Prewitt
    gradients, per the original paper.
    """
    ga, gb = _gray(a).astype(np.float32), _gray(b).astype(np.float32)
    ga = cv2.resize(ga, (ga.shape[1] // 2, ga.shape[0] // 2), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(gb, (gb.shape[1] // 2, gb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    hx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3.0
    hy = hx.T
    ma = np.sqrt(cv2.filter2D(ga, -1, hx) ** 2 + cv2.filter2D(ga, -1, hy) ** 2)
    mb = np.sqrt(cv2.filter2D(gb, -1, hx) ** 2 + cv2.filter2D(gb, -1, hy) ** 2)
    gms = (2 * ma * mb + c) / (ma**2 + mb**2 + c)
    return float(gms.mean()), float(gms.std())


def tiled_deviation(a: np.ndarray, b: np.ndarray, grid: int = 8) -> np.ndarray:
    """Per-tile mean-abs-diff (%) over a grid x grid layout -- shows WHERE deviation concentrates
    (face vs. skin vs. hair vs. background) rather than a single global number."""
    h, w = a.shape[:2]
    th, tw = h // grid, w // grid
    out = np.zeros((grid, grid), dtype=np.float32)
    af, bf = a.astype(np.float32), b.astype(np.float32)
    for i in range(grid):
        for j in range(grid):
            ta = af[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            tb = bf[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            out[i, j] = np.abs(ta - tb).mean() / 255.0 * 100.0
    return out


def save_tilemap(dev: np.ndarray, shape_hw: tuple[int, int], path: str) -> None:
    """Render the tiled-deviation grid as a labeled heatmap (per-tile MAD% printed)."""
    grid = dev.shape[0]
    h, w = shape_hw
    norm = np.clip(dev / max(dev.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
    th, tw = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            cv2.putText(heat, f"{dev[i, j]:.0f}", (j * tw + 4, i * th + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(tw / 320.0, 0.3), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, heat)


def save_triptych(ref: np.ndarray, var: np.ndarray, heatmap_path: str, out_path: str, labels=("off", "variant", "diff")) -> None:
    """Side-by-side [reference | variant | diff-heatmap] with labels -- the at-a-glance A/B/C view."""
    heat = cv2.imread(heatmap_path) if os.path.exists(heatmap_path) else np.zeros_like(ref[..., ::-1])
    # ref/var are RGB (from PIL); convert to BGR for cv2 hconcat/write
    panels = [cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), cv2.cvtColor(var, cv2.COLOR_RGB2BGR), heat]
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    strip = cv2.hconcat(panels)
    x = 0
    for p, lab in zip(panels, labels, strict=False):
        cv2.putText(strip, lab, (x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(strip, lab, (x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        x += p.shape[1]
    cv2.imwrite(out_path, strip)


@functools.lru_cache(maxsize=1)
def _vgg_features():
    """Lazily build a frozen VGG16 feature extractor (downloads ~528MB ImageNet weights on first use).
    Used for an LPIPS-like perceptual distance without adding the lpips dependency."""
    import torch
    from torchvision.models import VGG16_Weights, vgg16

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dev


def perceptual_vgg_distance(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS-like perceptual distance: normalized L2 between VGG16 feature maps at several ReLU stages,
    averaged. Higher = more perceptually different (captures texture/realism, not just pixels).
    Opt-in (downloads weights, slower); call only when requested."""
    import torch

    model, dev = _vgg_features()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(img):
        t = torch.from_numpy(img[..., :3].astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        if t.shape[-1] > 768:
            t = torch.nn.functional.interpolate(t, scale_factor=768 / t.shape[-1], mode="bilinear", align_corners=False)
        return (t - mean) / std

    taps = {3, 8, 15, 22}  # after a few ReLU blocks
    with torch.no_grad():
        xa, xb = prep(a), prep(b)
        dist = 0.0
        for i, layer in enumerate(model):
            xa, xb = layer(xa), layer(xb)
            if i in taps:
                fa = torch.nn.functional.normalize(xa, dim=1)
                fb = torch.nn.functional.normalize(xb, dim=1)
                dist += float(((fa - fb) ** 2).mean())
    return round(dist / len(taps), 5)


@dataclass
class QualityReport:
    mean_abs_diff_pct: float  # 0-100
    psnr_db: float
    ssim: float
    gmsd: float  # gradient-magnitude similarity deviation; higher => more texture/edge distortion
    gmsm: float  # gradient-magnitude similarity mean; higher => closer
    tiled_max_dev_pct: float  # worst grid-tile MAD% (localizes the biggest regional change)
    tiled_mean_dev_pct: float
    face_region_ssim: float | None
    n_faces: int
    sharpness_ref: float
    sharpness_var: float
    sharpness_ratio: float  # variant/ref; <1 => softer/blurrier than reference
    perceptual_vgg: float | None  # LPIPS-like VGG distance (opt-in)
    heatmap_path: str | None
    tilemap_path: str | None
    triptych_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def compare(
    ref: np.ndarray,
    var: np.ndarray,
    heatmap_path: str | None = None,
    tilemap_path: str | None = None,
    triptych_path: str | None = None,
    grid: int = 8,
    perceptual: bool = False,
) -> QualityReport:
    """Full quality comparison of a variant image against the reference (acceleration-off) image.

    Combines pixel (MAD/PSNR), structural (SSIM), TEXTURE/EDGE (GMSD -- catches skin-detail/realism
    loss), localized (tiled MAD), face-region, sharpness, and optional perceptual (VGG) metrics, and
    writes a diff heatmap, a labeled tile map, and a [ref|var|diff] triptych for at-a-glance review.
    """
    if ref.shape != var.shape:
        var = cv2.resize(var, (ref.shape[1], ref.shape[0]))
    mad = mean_abs_diff(ref, var)
    s, smap = ssim(ref, var)
    gmsm, gmsd_val = gmsd(ref, var)
    tiles = tiled_deviation(ref, var, grid=grid)
    faces = detect_faces(ref)
    fr = face_region_ssim(ref, var, smap)
    sharp_ref = laplacian_sharpness(ref)
    sharp_var = laplacian_sharpness(var)
    if heatmap_path:
        save_diff_heatmap(ref, var, heatmap_path)
    if tilemap_path:
        save_tilemap(tiles, (ref.shape[0], ref.shape[1]), tilemap_path)
    if triptych_path and heatmap_path:
        save_triptych(ref, var, heatmap_path, triptych_path)
    return QualityReport(
        mean_abs_diff_pct=round(mad / 255.0 * 100.0, 3),
        psnr_db=round(psnr(ref, var), 2),
        ssim=round(s, 4),
        gmsd=round(gmsd_val, 4),
        gmsm=round(gmsm, 4),
        tiled_max_dev_pct=round(float(tiles.max()), 3),
        tiled_mean_dev_pct=round(float(tiles.mean()), 3),
        face_region_ssim=round(fr, 4) if fr is not None else None,
        n_faces=len(faces),
        sharpness_ref=round(sharp_ref, 1),
        sharpness_var=round(sharp_var, 1),
        sharpness_ratio=round(sharp_var / sharp_ref, 3) if sharp_ref > 1e-6 else 1.0,
        perceptual_vgg=perceptual_vgg_distance(ref, var) if perceptual else None,
        heatmap_path=heatmap_path,
        tilemap_path=tilemap_path,
        triptych_path=triptych_path,
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
    # blur -> sharpness ratio < 1 AND higher GMSD (texture distortion) than identical
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    r3 = compare(img, blurred)
    assert r3.sharpness_ratio < 1.0, r3
    # identical -> gmsd ~0; degraded -> gmsd grows; tiled max >= mean
    assert r.gmsd < 1e-3 and r2.gmsd > r.gmsd and r3.gmsd > r.gmsd, (r.gmsd, r2.gmsd, r3.gmsd)
    assert r2.tiled_max_dev_pct >= r2.tiled_mean_dev_pct
    print("arbiter_metrics self-test OK:")
    print("  identical: ssim", r.ssim, "gmsd", r.gmsd, "psnr", r.psnr_db)
    print("  noisy    : ssim", r2.ssim, "gmsd", r2.gmsd, "tiled max/mean", r2.tiled_max_dev_pct, r2.tiled_mean_dev_pct)
    print("  blurred  : sharpness_ratio", r3.sharpness_ratio, "gmsd", r3.gmsd)


if __name__ == "__main__":
    _self_test()
