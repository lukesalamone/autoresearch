#!/usr/bin/env python3
"""
plot.py — visualise autoresearch progress from results.txt.

Usage:
    python3 plot.py                      # saves progress.png
    python3 plot.py --out myplot.png
    python3 plot.py --show               # open interactive window instead
"""

import argparse
import re
import json
from pathlib import Path

def parse_results(path="results.txt"):
    text = Path(path).read_text()
    blocks = re.split(r"\n={40,}\n", text)
    experiments = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        def field(name):
            m = re.search(rf"^{name}:\s*(.+)$", block, re.MULTILINE)
            return m.group(1).strip() if m else None

        exp_id  = field("exp_id")
        mrr     = field(r"mrr@10")
        notes   = field("notes") or ""
        hparams = field("hyperparams")

        if not exp_id or not mrr:
            continue

        size_mb = field("model_size_mb_fp16")

        try:
            hp = json.loads(hparams) if hparams else {}
        except json.JSONDecodeError:
            hp = {}

        experiments.append({
            "exp_id":   exp_id,
            "mrr":      float(mrr),
            "notes":    notes,
            "size_mb":  float(size_mb) if size_mb else None,
            "hparams":  hp,
        })

    return experiments


def make_plot(experiments, out=None, show=False):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    xs      = list(range(len(experiments)))
    mrrs    = [e["mrr"] for e in experiments]
    notes   = [e["notes"] for e in experiments]

    # Compute running best and which points set a new best
    running_best   = []
    best_so_far    = -1
    is_improvement = []
    for m in mrrs:
        if m > best_so_far:
            best_so_far = m
            is_improvement.append(True)
        else:
            is_improvement.append(False)
        running_best.append(best_so_far)

    n_kept = sum(is_improvement)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # All experiments — grey dots
    disc_x = [x for x, imp in zip(xs, is_improvement) if not imp]
    disc_y = [m for m, imp in zip(mrrs, is_improvement) if not imp]
    ax.scatter(disc_x, disc_y, color="#555555", s=18, zorder=2, label="All runs")

    # Running best line
    ax.step(xs, running_best, where="post", color="#00cc66", linewidth=1.5,
            zorder=3, label="Running best")

    # Improvement dots
    imp_x = [x for x, imp in zip(xs, is_improvement) if imp]
    imp_y = [m for m, imp in zip(mrrs, is_improvement) if imp]
    ax.scatter(imp_x, imp_y, color="#00cc66", s=55, zorder=4, label="New best")

    # Labels on improvement points
    for x, y, note in zip(imp_x, imp_y, [notes[i] for i, imp in enumerate(is_improvement) if imp]):
        label = note[:40] + "…" if len(note) > 40 else note
        txt = ax.annotate(
            label, xy=(x, y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=6.5, color="#00cc66", rotation=35,
            ha="left", va="bottom",
        )
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="#0f0f0f")])

    n_total = len(experiments)
    ax.set_title(
        f"Autoresearch Progress: {n_total} experiment{'s' if n_total != 1 else ''}, "
        f"{n_kept} improvement{'s' if n_kept != 1 else ''}",
        color="white", fontsize=13, pad=12,
    )
    ax.set_xlabel("Experiment #", color="#aaaaaa")
    ax.set_ylabel("MRR@10 (higher is better)", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(color="#222222", linewidth=0.5)

    legend = ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white",
                       fontsize=9, loc="lower right")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        dest = out or "progress.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.txt")
    p.add_argument("--out",     default=None)
    p.add_argument("--show",    action="store_true")
    args = p.parse_args()

    experiments = parse_results(args.results)
    if not experiments:
        print("No experiments found in results.txt")
        return

    print(f"Parsed {len(experiments)} experiments")
    best = max(experiments, key=lambda e: e["mrr"])
    print(f"Best so far: {best['exp_id']}  mrr@10={best['mrr']:.4f}  "
          f"size={best['size_mb']:.2f}MB  notes={best['notes']}")

    make_plot(experiments, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
