#!/usr/bin/env python3
"""Quantify the unified-memory model-cache win.

Cycles through all installed SDXL models across N rounds at low step count (so model
load/reload time dominates over denoise). Round 1 loads everything cold; round 2 shows
the payoff: with a big enough cache all models stay resident (0 reloads), with a small
cache the least-recently-used ones get evicted and must reload.

Run it twice against servers started with different caps to A/B the change:
  INVOKEAI_MAX_CACHE_RAM_GB=32  -> old behavior
  INVOKEAI_MAX_CACHE_RAM_GB=62  -> new MPS unified-memory behavior

    conda run -n invokeai python scripts/cache_bench.py --label cap32
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from benchmark_mps import build_sdxl_graph, enqueue, get_models, wait_for_item

ANSI = re.compile(r"\x1b\[[0-9;]*m")
MISS = re.compile(r"Model cache misses:\s*(\d+)")
HITS = re.compile(r"Model cache hits:\s*(\d+)")


def last_cache_stats(log_path: str):
    p = Path(log_path)
    if not p.exists():
        return None, None
    lines = ANSI.sub("", p.read_text(errors="ignore")).splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if "Graph stats:" in lines[i]:
            start = i
            break
    if start is None:
        return None, None
    misses = hits = None
    for line in lines[start:]:
        m = MISS.search(line)
        if m:
            misses = int(m.group(1))
        h = HITS.search(line)
        if h:
            hits = int(h.group(1))
    return misses, hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:9090")
    p.add_argument("--server-log", default="/tmp/invokeai_server.log")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--label", default="cache")
    args = p.parse_args()

    models = [m for m in get_models(args.server) if m.get("type") == "main" and m.get("base") == "sdxl"]
    print(f"[{args.label}] Cycling {len(models)} SDXL models x {args.rounds} rounds @ {args.width}px steps={args.steps}")
    for m in models:
        print(f"  - {m['name']}")

    seed = 1000
    round_summary = []
    for r in range(args.rounds):
        rt = 0.0
        rmiss = 0
        for m in models:
            seed += 1
            g = build_sdxl_graph(m, "a scenic landscape", "blurry", seed, args.width, args.height, args.steps, 5.0, "euler")
            for n in g["nodes"].values():
                n["use_cache"] = False
            t0 = time.time()
            item = enqueue(args.server, "default", g)
            status = wait_for_item(args.server, "default", item, 900)
            wall = time.time() - t0
            misses, _hits = last_cache_stats(args.server_log)
            rt += wall
            rmiss += misses or 0
            print(f"  round{r + 1} {m['name'][:30]:30s} {status:9s} {wall:6.2f}s  misses={misses}")
        round_summary.append((rt, rmiss))
        print(f"  -> round {r + 1} TOTAL: {rt:.1f}s, {rmiss} cache misses\n")

    print("=== SUMMARY ===")
    for i, (rt, rm) in enumerate(round_summary):
        print(f"  round {i + 1}: {rt:.1f}s wall, {rm} model-cache misses")
    if len(round_summary) >= 2:
        print("  (round 2 is the steady-state: lower time + fewer misses = bigger effective cache)")


if __name__ == "__main__":
    main()
