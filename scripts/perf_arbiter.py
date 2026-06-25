#!/usr/bin/env python3
"""Performance arbiter: an automated, verbose, feedback-loop benchmark harness for InvokeAI on
Apple Silicon. It drives the REAL in-app pipeline over the HTTP API (so it measures exactly what
ships, not raw diffusers) across a matrix of {model architecture x prompt x optimization
experiment}, then judges each experiment against the acceleration-OFF reference.

For each run it records: wall-clock, per-stage timing + cache skip-count + peak cache memory
(parsed from the server log), and -- versus the reference image -- MAD / PSNR / SSIM, face-region
SSIM (where faces are detected, to localize anatomy drift), a sharpness ratio, and an abs-diff
heatmap. It writes machine-readable JSONL, a human-readable ledger with a PASS/WARN/FAIL verdict
per experiment, and a markdown gallery.

  conda run -n invokeai python scripts/perf_arbiter.py --arch sdxl --prompts portrait,scene
  conda run -n invokeai python scripts/perf_arbiter.py --arch sd-1,sdxl,flux --fast

The OFF experiment is the upstream-equivalent denoise (caches disabled); the balanced/max
experiments are this branch's MPS acceleration. (A fully separate upstream checkout would also
capture the lossless startup/sync/memory changes, which by construction do not alter output.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time

import arbiter_metrics as M
import numpy as np
import requests
from PIL import Image

API = os.environ.get("INVOKE_API", "http://127.0.0.1:9090")
SERVER_LOG = os.environ.get("INVOKE_LOG", "/tmp/invokeai_server.log")
OUT_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, ".perf-logs", "arbiter")

PROMPTS = {
    "portrait": (
        "close-up portrait photo of a woman, freckles, detailed eyes, natural skin texture, "
        "soft window light, 85mm",
        "blurry, low quality, deformed, extra fingers",
    ),
    "fullbody": (
        "full body photo of a person standing, both hands visible, fingers spread, detailed face, "
        "realistic anatomy, sharp focus",
        "blurry, low quality, deformed hands, extra fingers, bad anatomy",
    ),
    "scene": (
        "a bustling medieval marketplace at golden hour, many people, stalls, intricate detail, wide shot",
        "blurry, low quality",
    ),
    "landscape": (
        "a scenic mountain landscape at sunset, highly detailed, sharp focus, dramatic clouds",
        "blurry, low quality",
    ),
}

# Per-architecture defaults. `experiments` map a label -> override value for the denoise node field.
ARCH = {
    "sd-1": {
        "model": "animics_v30",
        "steps": 28, "cfg": 6.0, "w": 512, "h": 768, "scheduler": "dpmpp_2m_k",
        "field": "deepcache_interval", "experiments": [("off", 1), ("balanced", 2), ("max", 3)],
    },
    "sdxl": {
        "model": "cyberrealisticPony_v160",
        "steps": 28, "cfg": 6.0, "w": 832, "h": 1216, "scheduler": "dpmpp_2m_k",
        "field": "deepcache_interval", "experiments": [("off", 1), ("balanced", 2), ("max", 3)],
    },
    "flux": {
        "model": "flux_schnell_flux1-schnell",
        "steps": 20, "cfg": 1.0, "guidance": 3.5, "w": 1024, "h": 1024,
        "field": "first_block_cache_threshold", "experiments": [("off", 0.0), ("balanced", 0.12), ("max", 0.2)],
    },
}

SEED = 7777


def ident(m: dict) -> dict:
    return {"key": m["key"], "hash": m["hash"], "name": m["name"], "base": m["base"], "type": m["type"]}


def get_models() -> list[dict]:
    return requests.get(f"{API}/api/v2/models/", timeout=20).json()["models"]


def pick_model(models: list[dict], base: str, name: str) -> dict:
    for m in models:
        if m["base"] == base and m["type"] == "main" and m["name"] == name:
            return m
    # fall back to any main of that base
    for m in models:
        if m["base"] == base and m["type"] == "main":
            return m
    raise SystemExit(f"no main model for base={base} (wanted {name})")


def _edges(pairs):
    return [{"source": {"node_id": s, "field": sf}, "destination": {"node_id": d, "field": df}} for (s, sf, d, df) in pairs]


def build_graph(arch: str, models: list[dict], cfg: dict, pos: str, neg: str, field_val) -> dict:
    field = cfg["field"]
    if arch == "flux":
        m = pick_model(models, "flux", cfg["model"])
        t5 = next(x for x in models if x["type"] == "t5_encoder")
        clip = next(x for x in models if x["type"] == "clip_embed")
        vae = next(x for x in models if x["base"] == "flux" and x["type"] == "vae")
        nodes = {
            "loader": {"id": "loader", "type": "flux_model_loader", "model": ident(m),
                       "t5_encoder_model": ident(t5), "clip_embed_model": ident(clip), "vae_model": ident(vae)},
            "pos": {"id": "pos", "type": "flux_text_encoder", "prompt": pos},
            "denoise": {"id": "denoise", "type": "flux_denoise", "num_steps": cfg["steps"],
                        "guidance": cfg["guidance"], "cfg_scale": cfg["cfg"], "width": cfg["w"], "height": cfg["h"],
                        "seed": SEED, field: field_val},
            "decode": {"id": "decode", "type": "flux_vae_decode", "is_intermediate": False},
        }
        edges = _edges([
            ("loader", "transformer", "denoise", "transformer"),
            ("loader", "clip", "pos", "clip"), ("loader", "t5_encoder", "pos", "t5_encoder"),
            ("loader", "max_seq_len", "pos", "t5_max_seq_len"),
            ("pos", "conditioning", "denoise", "positive_text_conditioning"),
            ("loader", "vae", "decode", "vae"), ("denoise", "latents", "decode", "latents"),
        ])
        return {"id": "arb_flux", "nodes": nodes, "edges": edges}

    # SD1.5 / SDXL share denoise_latents + noise + l2i; differ in loader + compel.
    base = "sdxl" if arch == "sdxl" else "sd-1"
    m = pick_model(models, base, cfg["model"])
    common_denoise = {"id": "denoise", "type": "denoise_latents", "steps": cfg["steps"], "cfg_scale": cfg["cfg"],
                      "scheduler": cfg["scheduler"], "denoising_start": 0.0, "denoising_end": 1.0, field: field_val}
    noise = {"id": "noise", "type": "noise", "seed": SEED, "width": cfg["w"], "height": cfg["h"]}
    l2i = {"id": "l2i", "type": "l2i", "fp32": False, "is_intermediate": False}
    if arch == "sdxl":
        nodes = {
            "loader": {"id": "loader", "type": "sdxl_model_loader", "model": ident(m)},
            "pos": {"id": "pos", "type": "sdxl_compel_prompt", "prompt": pos, "style": pos},
            "neg": {"id": "neg", "type": "sdxl_compel_prompt", "prompt": neg, "style": neg},
            "noise": noise, "denoise": common_denoise, "l2i": l2i,
        }
        edges = _edges([
            ("loader", "unet", "denoise", "unet"),
            ("loader", "clip", "pos", "clip"), ("loader", "clip2", "pos", "clip2"),
            ("loader", "clip", "neg", "clip"), ("loader", "clip2", "neg", "clip2"),
            ("pos", "conditioning", "denoise", "positive_conditioning"),
            ("neg", "conditioning", "denoise", "negative_conditioning"),
            ("noise", "noise", "denoise", "noise"),
            ("denoise", "latents", "l2i", "latents"), ("loader", "vae", "l2i", "vae"),
        ])
        return {"id": "arb_sdxl", "nodes": nodes, "edges": edges}
    # sd-1
    nodes = {
        "loader": {"id": "loader", "type": "main_model_loader", "model": ident(m)},
        "pos": {"id": "pos", "type": "compel", "prompt": pos},
        "neg": {"id": "neg", "type": "compel", "prompt": neg},
        "noise": noise, "denoise": common_denoise, "l2i": l2i,
    }
    edges = _edges([
        ("loader", "unet", "denoise", "unet"),
        ("loader", "clip", "pos", "clip"), ("loader", "clip", "neg", "clip"),
        ("pos", "conditioning", "denoise", "positive_conditioning"),
        ("neg", "conditioning", "denoise", "negative_conditioning"),
        ("noise", "noise", "denoise", "noise"),
        ("denoise", "latents", "l2i", "latents"), ("loader", "vae", "l2i", "vae"),
    ])
    return {"id": "arb_sd1", "nodes": nodes, "edges": edges}


_NODE_TIME = re.compile(r"^\s*(flux_denoise|denoise_latents|flux_vae_decode|l2i|flux_text_encoder|compel|sdxl_compel_prompt)\s+\d+\s+([\d.]+)s")
_SKIP = re.compile(r"(FirstBlockCache|DeepCache).*?(\d+) step|DeepCache enabled \(cache_interval=(\d+)")
_HWM = re.compile(r"high water mark:\s*([\d.]+)/")


def parse_log_tail(text: str) -> dict:
    stages: dict[str, float] = {}
    for line in text.splitlines():
        mt = _NODE_TIME.match(line)
        if mt:
            stages[mt.group(1)] = float(mt.group(2))
    skips = 0
    sk = re.search(r"skipped the block stack on (\d+) step|forecast.*?on (\d+) step", text)
    if sk:
        skips = int(next(g for g in sk.groups() if g))
    hwm = None
    hm = _HWM.search(text)
    if hm:
        hwm = float(hm.group(1))
    return {"stages": stages, "skips": skips, "peak_cache_gb": hwm}


def run_one(arch: str, models: list[dict], cfg: dict, pos: str, neg: str, field_val, out_dir: str, tag: str) -> dict:
    graph = build_graph(arch, models, cfg, pos, neg, field_val)
    # Disable InvokeAI's node-output cache so every run actually recomputes -- otherwise repeat runs
    # with the same seed/graph return stale cached results (instant, identical) and corrupt the A/B.
    for node in graph["nodes"].values():
        node["use_cache"] = False
    log_off = os.path.getsize(SERVER_LOG) if os.path.exists(SERVER_LOG) else 0
    r = requests.post(f"{API}/api/v1/queue/default/enqueue_batch",
                      json={"prepend": False, "batch": {"graph": graph, "runs": 1}}, timeout=30)
    if r.status_code not in (200, 201):
        return {"error": f"enqueue {r.status_code}: {r.text[:300]}"}
    bid = r.json()["batch"]["batch_id"]
    t0 = time.time()
    while True:
        st = requests.get(f"{API}/api/v1/queue/default/b/{bid}/status", timeout=10).json()
        if st["completed"] >= 1:
            break
        if st["failed"] or st["canceled"]:
            return {"error": f"generation failed: {st}"}
        if time.time() - t0 > 900:
            return {"error": "timeout"}
        time.sleep(1)
    wall = time.time() - t0
    # newly-appended log text for this run
    with open(SERVER_LOG, "r", errors="ignore") as f:
        f.seek(log_off)
        logtext = f.read()
    parsed = parse_log_tail(logtext)
    img_meta = requests.get(f"{API}/api/v1/images/?limit=1&is_intermediate=false", timeout=10).json()["items"][0]
    blob = requests.get(f"{API}/api/v1/images/i/{img_meta['image_name']}/full", timeout=60).content
    img_path = os.path.join(out_dir, f"{tag}.png")
    with open(img_path, "wb") as f:
        f.write(blob)
    return {"wall": round(wall, 2), "img_path": img_path, **parsed}


def verdict(speedup: float, q: M.QualityReport | None) -> str:
    """Judge an experiment by ARTIFACTS (degradation), not by deviation from the reference.

    SSIM/MAD measure how much the optimization changed the image versus acceleration-off; an
    approximation like DeepCache legitimately changes composition while staying high quality, so
    deviation alone is not a failure. We fail/warn on actual degradation signals: a degenerate
    (flat/blank) image, loss of high-frequency detail (blur), or faces degrading much more than
    the frame as a whole (anatomy weirdness).
    """
    if q is None:
        return "BASELINE"
    if q.sharpness_var < 5.0:
        return "FAIL (degenerate/blank)"
    if speedup < 1.05:
        return "FAIL (no speedup)"
    flags = []
    if q.sharpness_ratio < 0.55:
        flags.append("blur")
    if q.face_region_ssim is not None and (q.ssim - q.face_region_ssim) > 0.06:
        flags.append("faces")
    return "PASS" if not flags else "WARN (" + ",".join(flags) + ")"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sdxl", help="comma list of sd-1,sdxl,flux")
    ap.add_argument("--prompts", default="portrait,scene", help="comma list of prompt keys")
    ap.add_argument("--fast", action="store_true", help="fewer steps for a quick smoke run")
    ap.add_argument("--perceptual", action="store_true", help="also compute the VGG perceptual distance (downloads ~528MB weights on first use)")
    ap.add_argument("--grid", type=int, default=8, help="tiled-deviation grid size")
    args = ap.parse_args()

    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    prompt_keys = [p.strip() for p in args.prompts.split(",") if p.strip()]
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(os.path.join(OUT_ROOT, ts))
    os.makedirs(out_dir, exist_ok=True)
    jsonl = open(os.path.join(out_dir, "runs.jsonl"), "w")
    models = get_models()
    rows = []

    def emit(rec: dict) -> None:
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        rows.append(rec)

    for arch in archs:
        cfg = dict(ARCH[arch])
        if args.fast:
            cfg["steps"] = max(6, cfg["steps"] // 3)
        for pkey in prompt_keys:
            pos, neg = PROMPTS[pkey]
            ref_img = None
            ref_time = None
            for label, val in cfg["experiments"]:
                tag = f"{arch}_{pkey}_{label}"
                print(f"[run] {tag} ({cfg['field']}={val}) ...", flush=True)
                res = run_one(arch, models, cfg, pos, neg, val, out_dir, tag)
                rec = {"arch": arch, "prompt": pkey, "experiment": label, "field": cfg["field"], "value": val, **res}
                if "error" in res:
                    rec["verdict"] = "ERROR"
                    print(f"    ERROR: {res['error']}")
                    emit(rec)
                    continue
                arr = np.asarray(Image.open(res["img_path"]).convert("RGB"))
                if label == "off":
                    ref_img, ref_time = arr, res["wall"]
                    rec["speedup"] = 1.0
                    rec["verdict"] = verdict(1.0, None)
                else:
                    q = (
                        M.compare(
                            ref_img, arr,
                            heatmap_path=os.path.join(out_dir, f"{tag}_diff.png"),
                            tilemap_path=os.path.join(out_dir, f"{tag}_tiles.png"),
                            triptych_path=os.path.join(out_dir, f"{tag}_triptych.png"),
                            grid=args.grid,
                            perceptual=args.perceptual,
                        )
                        if ref_img is not None
                        else None
                    )
                    speedup = (ref_time / res["wall"]) if ref_time else 1.0
                    rec["speedup"] = round(speedup, 2)
                    rec["quality"] = q.to_dict() if q else None
                    rec["verdict"] = verdict(speedup, q)
                den = res["stages"].get("flux_denoise") or res["stages"].get("denoise_latents")
                rec["denoise_s"] = den
                q = rec.get("quality") or {}
                print(f"    wall={res['wall']}s denoise={den}s skips={res.get('skips')} "
                      f"speedup={rec.get('speedup')} ssim={q.get('ssim')} gmsd={q.get('gmsd')} "
                      f"tiledMax%={q.get('tiled_max_dev_pct')} -> {rec['verdict']}")
                emit(rec)
    jsonl.close()
    write_ledger(out_dir, rows)
    write_html(out_dir, rows)
    print(f"\n=== arbiter complete -> {out_dir} ===")
    print(f"ledger: {os.path.join(out_dir, 'ledger.md')}")
    print(f"gallery (open in browser): {os.path.join(out_dir, 'report.html')}")


def write_ledger(out_dir: str, rows: list[dict]) -> None:
    lines = ["# Performance Arbiter Ledger", "", f"Runs: {len(rows)}  |  dir: `{out_dir}`", "",
             "Quality columns are **deviation from the off-reference**, not absolute quality. "
             "GMSD = texture/edge distortion (higher = more skin/detail change, the most sensitive "
             "to realism loss); MAD% = mean pixel diff; tiledMax% = worst regional change; "
             "perceptual = VGG/LPIPS-like distance (when --perceptual).", ""]
    lines += ["| arch | prompt | exp | field=val | wall(s) | denoise(s) | speedup | SSIM | GMSD | tiledMax% | MAD% | sharpR | faceSSIM | percep | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r.get("quality") or {}
        lines.append(
            f"| {r['arch']} | {r['prompt']} | {r['experiment']} | {r['field']}={r['value']} | "
            f"{r.get('wall','')} | {r.get('denoise_s','')} | {r.get('speedup','')} | "
            f"{q.get('ssim','')} | {q.get('gmsd','')} | {q.get('tiled_max_dev_pct','')} | {q.get('mean_abs_diff_pct','')} | "
            f"{q.get('sharpness_ratio','')} | {q.get('face_region_ssim','')} | {q.get('perceptual_vgg','')} | {r.get('verdict','')} |"
        )
    # gallery
    lines += ["", "## Gallery (reference | variant | diff heatmap)", ""]
    by = {}
    for r in rows:
        by.setdefault((r["arch"], r["prompt"]), []).append(r)
    for (arch, pk), rs in by.items():
        lines.append(f"### {arch} — {pk}")
        for r in rs:
            if "img_path" not in r:
                continue
            base = os.path.basename(r["img_path"])
            diff = base.replace(".png", "_diff.png")
            extra = f" — speedup {r.get('speedup')}×, SSIM {(r.get('quality') or {}).get('ssim','-')}, **{r.get('verdict')}**"
            lines.append(f"- **{r['experiment']}** ({r['field']}={r['value']}){extra}")
            lines.append(f"  ![{base}]({base})" + (f" ![{diff}]({diff})" if os.path.exists(os.path.join(out_dir, diff)) else ""))
        lines.append("")
    with open(os.path.join(out_dir, "ledger.md"), "w") as f:
        f.write("\n".join(lines))


def write_html(out_dir: str, rows: list[dict]) -> None:
    """Self-contained HTML gallery: per (arch,prompt), each variant's [ref|var|diff] triptych + tile
    map + a metrics row. Open report.html in a browser to A/B/C by eye AND by numbers in one place."""
    css = (
        "body{font:14px -apple-system,system-ui,sans-serif;background:#111;color:#eee;margin:24px}"
        "h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:32px}"
        "img{max-width:100%;border:1px solid #333;border-radius:6px;display:block;margin:6px 0}"
        ".v{margin:18px 0;padding:12px;background:#1b1b1b;border-radius:8px}"
        "table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #333;padding:3px 8px;font-size:12px}"
        ".pass{color:#5cd65c}.warn{color:#ffcc44}.fail{color:#ff5c5c}.base{color:#88aaff}"
        ".note{color:#aaa;font-size:12px}"
    )
    h = [f"<!doctype html><meta charset=utf8><title>Perf Arbiter</title><style>{css}</style>",
         "<h1>Performance Arbiter — visual + empirical A/B</h1>",
         "<p class=note>Each row: <b>left</b>=acceleration off (reference), <b>middle</b>=variant, "
         "<b>right</b>=abs-diff heatmap (bright=more change). The tile map shows per-region change %. "
         "Metrics are <b>deviation</b> from off, not absolute quality — GMSD is the texture/realism "
         "signal; sharpR&lt;1 = softer than reference.</p>"]
    by: dict = {}
    for r in rows:
        by.setdefault((r["arch"], r["prompt"]), []).append(r)
    for (arch, pk), rs in by.items():
        h.append(f"<h2>{arch} — {pk}</h2>")
        for r in rs:
            if "img_path" not in r:
                continue
            tag = os.path.basename(r["img_path"]).replace(".png", "")
            q = r.get("quality") or {}
            vcls = "base" if r["experiment"] == "off" else ("pass" if r["verdict"].startswith("PASS") else ("fail" if "FAIL" in r["verdict"] else "warn"))
            h.append("<div class=v>")
            h.append(f"<b>{r['experiment']}</b> ({r['field']}={r['value']}) — "
                     f"speedup <b>{r.get('speedup','-')}×</b>, <span class={vcls}>{r['verdict']}</span>")
            if r["experiment"] == "off":
                h.append(f"<img src='{tag}.png' style='max-width:360px'>")
            else:
                h.append(f"<table><tr><th>speedup<th>SSIM<th>GMSD<th>tiledMax%<th>MAD%<th>sharpR<th>faceSSIM<th>percep</tr>"
                         f"<tr><td>{r.get('speedup','')}<td>{q.get('ssim','')}<td>{q.get('gmsd','')}"
                         f"<td>{q.get('tiled_max_dev_pct','')}<td>{q.get('mean_abs_diff_pct','')}"
                         f"<td>{q.get('sharpness_ratio','')}<td>{q.get('face_region_ssim','')}"
                         f"<td>{q.get('perceptual_vgg','')}</tr></table>")
                h.append(f"<img src='{tag}_triptych.png'>")
                h.append(f"<img src='{tag}_tiles.png' style='max-width:360px' title='per-region change %'>")
            h.append("</div>")
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write("\n".join(h))


if __name__ == "__main__":
    main()
