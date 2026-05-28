#!/usr/bin/env python3
"""Headless text-to-image benchmark for InvokeAI on Apple Silicon (MPS).

Drives generations against a *running* InvokeAI server over HTTP and records:
  - end-to-end wall time (client-side, robust), and
  - per-node timings parsed from the server log (denoise_latents, l2i/VAE, model load),
    plus the model-cache high-water mark.

This is the measurement foundation for the MPS optimization work: capture a baseline,
then re-run after each change and diff. It deliberately does not import torch or load
models itself -- it exercises the real generation path the app uses.

Example:
    conda run -n invokeai python scripts/benchmark_mps.py \
        --server http://127.0.0.1:9091 --server-log /tmp/invokeai_server.log \
        --models sd1,sdxl --steps 20 --width 512 --height 512 \
        --runs 3 --warmup 1 --out .perf-logs/baseline.json --label baseline
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NODE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s+(\d+)\s+([\d.]+)s\s+([+-][\d.]+)G\s*$")
EXEC_RE = re.compile(r"TOTAL GRAPH EXECUTION TIME:\s+([\d.]+)s")
HWM_RE = re.compile(r"Cache high water mark:\s+([\d.]+)/([\d.]+)G")

DEFAULT_PROMPT = "a photograph of an astronaut riding a horse, detailed, sharp focus"
DEFAULT_NEG = "blurry, low quality"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_models(server: str) -> list[dict[str, Any]]:
    r = requests.get(f"{server}/api/v2/models/", timeout=30)
    r.raise_for_status()
    return r.json()["models"]


def pick_model(models: list[dict[str, Any]], base: str, name: Optional[str] = None) -> Optional[dict[str, Any]]:
    candidates = [m for m in models if m.get("type") == "main" and m.get("base") == base]
    if name:
        for m in candidates:
            if m.get("name") == name:
                return m
    return candidates[0] if candidates else None


def model_field(m: dict[str, Any]) -> dict[str, Any]:
    return {"key": m["key"], "hash": m["hash"], "name": m["name"], "base": m["base"], "type": m["type"]}


def edge(sn: str, sf: str, dn: str, df: str) -> dict[str, Any]:
    return {"source": {"node_id": sn, "field": sf}, "destination": {"node_id": dn, "field": df}}


def build_sd1_graph(m, prompt, neg, seed, w, h, steps, cfg, scheduler) -> dict[str, Any]:
    i = {k: new_id(k) for k in ["loader", "pos", "neg", "noise", "denoise", "l2i"]}
    nodes = {
        i["loader"]: {"id": i["loader"], "type": "main_model_loader", "model": model_field(m), "is_intermediate": True},
        i["pos"]: {"id": i["pos"], "type": "compel", "prompt": prompt, "is_intermediate": True},
        i["neg"]: {"id": i["neg"], "type": "compel", "prompt": neg, "is_intermediate": True},
        i["noise"]: {"id": i["noise"], "type": "noise", "seed": seed, "width": w, "height": h, "is_intermediate": True},
        i["denoise"]: {
            "id": i["denoise"], "type": "denoise_latents", "steps": steps, "cfg_scale": cfg,
            "denoising_start": 0.0, "denoising_end": 1.0, "scheduler": scheduler, "is_intermediate": True,
        },
        i["l2i"]: {"id": i["l2i"], "type": "l2i", "fp32": False, "is_intermediate": False},
    }
    edges = [
        edge(i["loader"], "unet", i["denoise"], "unet"),
        edge(i["loader"], "clip", i["pos"], "clip"),
        edge(i["loader"], "clip", i["neg"], "clip"),
        edge(i["loader"], "vae", i["l2i"], "vae"),
        edge(i["pos"], "conditioning", i["denoise"], "positive_conditioning"),
        edge(i["neg"], "conditioning", i["denoise"], "negative_conditioning"),
        edge(i["noise"], "noise", i["denoise"], "noise"),
        edge(i["denoise"], "latents", i["l2i"], "latents"),
    ]
    return {"id": new_id("g"), "nodes": nodes, "edges": edges}


def build_sdxl_graph(m, prompt, neg, seed, w, h, steps, cfg, scheduler) -> dict[str, Any]:
    i = {k: new_id(k) for k in ["loader", "pos", "neg", "noise", "denoise", "l2i"]}

    def compel(nid: str, p: str) -> dict[str, Any]:
        return {
            "id": nid, "type": "sdxl_compel_prompt", "prompt": p, "style": p,
            "original_width": w, "original_height": h, "target_width": w, "target_height": h,
            "crop_top": 0, "crop_left": 0, "is_intermediate": True,
        }

    nodes = {
        i["loader"]: {"id": i["loader"], "type": "sdxl_model_loader", "model": model_field(m), "is_intermediate": True},
        i["pos"]: compel(i["pos"], prompt),
        i["neg"]: compel(i["neg"], neg),
        i["noise"]: {"id": i["noise"], "type": "noise", "seed": seed, "width": w, "height": h, "is_intermediate": True},
        i["denoise"]: {
            "id": i["denoise"], "type": "denoise_latents", "steps": steps, "cfg_scale": cfg,
            "denoising_start": 0.0, "denoising_end": 1.0, "scheduler": scheduler, "is_intermediate": True,
        },
        i["l2i"]: {"id": i["l2i"], "type": "l2i", "fp32": False, "is_intermediate": False},
    }
    edges = [
        edge(i["loader"], "unet", i["denoise"], "unet"),
        edge(i["loader"], "clip", i["pos"], "clip"),
        edge(i["loader"], "clip2", i["pos"], "clip2"),
        edge(i["loader"], "clip", i["neg"], "clip"),
        edge(i["loader"], "clip2", i["neg"], "clip2"),
        edge(i["loader"], "vae", i["l2i"], "vae"),
        edge(i["pos"], "conditioning", i["denoise"], "positive_conditioning"),
        edge(i["neg"], "conditioning", i["denoise"], "negative_conditioning"),
        edge(i["noise"], "noise", i["denoise"], "noise"),
        edge(i["denoise"], "latents", i["l2i"], "latents"),
    ]
    return {"id": new_id("g"), "nodes": nodes, "edges": edges}


BUILDERS = {"sd1": build_sd1_graph, "sdxl": build_sdxl_graph}
BASE_FOR = {"sd1": "sd-1", "sdxl": "sdxl"}


def enqueue(server: str, queue_id: str, graph: dict[str, Any]) -> int:
    body = {"batch": {"graph": graph, "runs": 1}, "prepend": False}
    r = requests.post(f"{server}/api/v1/queue/{queue_id}/enqueue_batch", json=body, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(f"enqueue failed {r.status_code}: {r.text[:500]}")
    return r.json()["item_ids"][0]


def wait_for_item(server: str, queue_id: str, item_id: int, timeout: float) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{server}/api/v1/queue/{queue_id}/i/{item_id}", timeout=30)
        r.raise_for_status()
        status = r.json()["status"]
        if status in ("completed", "failed", "canceled"):
            return status
        time.sleep(0.2)
    return "timeout"


def parse_last_stats(log_path: Path) -> dict[str, Any]:
    """Parse the most recent 'Graph stats' block from the server log."""
    if not log_path or not log_path.exists():
        return {}
    lines = ANSI_RE.sub("", log_path.read_text(errors="ignore")).splitlines()
    start = None
    for idx in range(len(lines) - 1, -1, -1):
        if "Graph stats:" in lines[idx]:
            start = idx
            break
    if start is None:
        return {}
    nodes: dict[str, float] = {}
    total_exec = None
    hwm = None
    for line in lines[start:]:
        m = NODE_RE.match(line)
        if m:
            nodes[m.group(1)] = float(m.group(3))
        m2 = EXEC_RE.search(line)
        if m2:
            total_exec = float(m2.group(1))
        m3 = HWM_RE.search(line)
        if m3:
            hwm = {"used_gb": float(m3.group(1)), "cache_gb": float(m3.group(2))}
    return {"nodes": nodes, "total_execution_s": total_exec, "cache_high_water_mark": hwm}


def run_config(args, models, which: str) -> dict[str, Any]:
    base = BASE_FOR[which]
    m = pick_model(models, base, args.model_name)
    if not m:
        return {"model_type": which, "error": f"no installed main model with base {base}"}
    builder = BUILDERS[which]
    log_path = Path(args.server_log) if args.server_log else None
    seed_counter = {"n": 0}

    def one_run() -> dict[str, Any]:
        # Use a unique seed per run and disable the node cache so every run actually computes
        # (identical inputs would otherwise be served from InvokeAI's invocation cache).
        seed_counter["n"] += 1
        seed = args.seed + seed_counter["n"]
        graph = builder(m, args.prompt, args.neg, seed, args.width, args.height, args.steps, args.cfg, args.scheduler)
        for node in graph["nodes"].values():
            node["use_cache"] = False
        t0 = time.time()
        item_id = enqueue(args.server, args.queue_id, graph)
        status = wait_for_item(args.server, args.queue_id, item_id, args.timeout)
        wall = time.time() - t0
        stats = parse_last_stats(log_path) if status == "completed" else {}
        return {"status": status, "wall_s": wall, "stats": stats}

    print(f"[{which}] model={m['name']!r} ({m['base']}) warmup={args.warmup} runs={args.runs} "
          f"{args.width}x{args.height} steps={args.steps}")
    for _ in range(args.warmup):
        w = one_run()
        print(f"  warmup: {w['status']} {w['wall_s']:.2f}s")
        if w["status"] != "completed":
            return {"model_type": which, "model": m["name"], "error": f"warmup {w['status']}", "detail": w}

    runs = []
    for r_i in range(args.runs):
        res = one_run()
        denoise = res["stats"].get("nodes", {}).get("denoise_latents")
        l2i = res["stats"].get("nodes", {}).get("l2i")
        its = (args.steps / denoise) if denoise else None
        print(f"  run {r_i + 1}/{args.runs}: {res['status']} wall={res['wall_s']:.2f}s "
              f"denoise={denoise!s}s l2i={l2i!s}s it/s={its:.2f}" if its else
              f"  run {r_i + 1}/{args.runs}: {res['status']} wall={res['wall_s']:.2f}s")
        runs.append(res)

    completed = [r for r in runs if r["status"] == "completed"]
    walls = [r["wall_s"] for r in completed]
    denoises = [r["stats"]["nodes"]["denoise_latents"] for r in completed if r["stats"].get("nodes", {}).get("denoise_latents")]
    l2is = [r["stats"]["nodes"]["l2i"] for r in completed if r["stats"].get("nodes", {}).get("l2i")]

    def med(xs):
        return round(statistics.median(xs), 4) if xs else None

    return {
        "model_type": which,
        "model": m["name"],
        "base": m["base"],
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "scheduler": args.scheduler,
        "runs_completed": len(completed),
        "median_wall_s": med(walls),
        "median_denoise_s": med(denoises),
        "median_l2i_s": med(l2is),
        "median_denoise_it_s": round(args.steps / med(denoises), 3) if med(denoises) else None,
        "cache_high_water_mark": completed[-1]["stats"].get("cache_high_water_mark") if completed else None,
        "raw_runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="InvokeAI MPS generation benchmark")
    p.add_argument("--server", default="http://127.0.0.1:9091")
    p.add_argument("--server-log", default="/tmp/invokeai_server.log", help="Path to the server log for per-node stats")
    p.add_argument("--queue-id", default="default")
    p.add_argument("--models", default="sd1,sdxl", help="Comma list of: sd1, sdxl")
    p.add_argument("--model-name", default=None, help="Force a specific model name (else first of base)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--scheduler", default="euler")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--neg", default=DEFAULT_NEG)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    models = get_models(args.server)
    results = [run_config(args, models, w.strip()) for w in args.models.split(",") if w.strip()]

    out = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server": args.server,
        "results": results,
    }
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nWrote {args.out}")
    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("error"):
            print(f"  {r['model_type']}: ERROR {r['error']}")
        else:
            print(f"  {r['model_type']} ({r['model']}): wall={r['median_wall_s']}s "
                  f"denoise={r['median_denoise_s']}s ({r['median_denoise_it_s']} it/s) l2i={r['median_l2i_s']}s")


if __name__ == "__main__":
    main()
