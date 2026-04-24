#!/usr/bin/env python3
"""
Visualize transformer_shape_scan.py outputs: loss, AOFE, AOFE-ratio vs depth,
and depth/width (aspect) ratio vs loss to spot the best shape.

Usage:
  python plot_depth_scan_results.py \\
    --csv results_tiny_gpt_depth_aofe/depth_scan_results.csv \\
    --out-dir results_tiny_gpt_depth_aofe/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot depth scan CSV from transformer_shape_scan.py")
    parser.add_argument("--csv", type=Path, required=True, help="depth_scan_results.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Figure output directory (default: csv parent / figures)")
    parser.add_argument("--loss-key", choices=("final_val_loss", "best_val_loss"), default="final_val_loss")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("Empty CSV")

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = np.array([int(r["n_layer"]) for r in rows])
    C = np.array([int(r["n_embd"]) for r in rows])
    loss = np.array([to_float(r, args.loss_key) for r in rows])
    aofe = np.array([to_float(r, "aofe") for r in rows])
    ratio = np.array([to_float(r, "aofe_ratio") for r in rows])

    # Depth/width aspect ratios (both common conventions)
    depth_over_width = L / np.maximum(C, 1)  # "深/宽"
    width_over_depth = C / np.maximum(L, 1)  # "宽/深"

    i_best = int(np.argmin(loss))
    best = rows[i_best]

    # --- Figure 1: three panels vs depth ---
    fig1, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True)
    x = L

    c_loss = "#0072B2"
    c_aofe = "#D55E00"
    c_ar = "#009E73"

    axes[0].plot(x, loss, "o-", color=c_loss, lw=1.4, ms=7)
    axes[0].scatter([L[i_best]], [loss[i_best]], s=140, c="none", edgecolors="#E69F00", linewidths=2.5, zorder=5)
    axes[0].set_ylabel(args.loss_key.replace("_", " "))
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].set_title("Validation loss vs depth (orange ring = lowest loss in scan)")

    axes[1].plot(x, aofe, "s-", color=c_aofe, lw=1.4, ms=6)
    axes[1].set_ylabel("AOFE")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.7)
    axes[1].set_title("AOFE (off-diagonal Frobenius energy) vs depth")

    axes[2].plot(x, ratio, "^-", color=c_ar, lw=1.4, ms=6)
    axes[2].set_ylabel("AOFE-ratio")
    axes[2].set_xlabel("Depth (n_layer)")
    axes[2].set_ylim(0, max(1.0, float(ratio.max()) * 1.05))
    axes[2].grid(True, linestyle=":", alpha=0.7)
    axes[2].set_title("AOFE-ratio vs depth")

    fig1.tight_layout()
    p1 = out_dir / "depth_scan_loss_aofe_vs_depth.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    fig1.savefig(out_dir / "depth_scan_loss_aofe_vs_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # --- Figure 2: aspect ratio vs loss ---
    fig2, ax = plt.subplots(figsize=(6.8, 5))
    sc = ax.scatter(depth_over_width, loss, c=L, cmap="viridis", s=80, edgecolors="0.3", linewidths=0.6)
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("n_layer (depth)")
    for i, r in enumerate(rows):
        ax.annotate(
            f"L{L[i]}×C{C[i]}",
            (depth_over_width[i], loss[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.85,
        )
    ax.scatter(
        [depth_over_width[i_best]],
        [loss[i_best]],
        s=200,
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=2.5,
        zorder=6,
        label=f"Lowest {args.loss_key}",
    )
    ax.set_xlabel(r"Depth/width ratio  ($n_{\mathrm{layer}} / n_{\mathrm{embd}}$)")
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title("Find shape by depth/width ratio (fixed ~3M params)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # --- Figure 3: width/depth vs loss (alternative aspect view) ---
    fig3, ax = plt.subplots(figsize=(6.8, 5))
    sc = ax.scatter(width_over_depth, loss, c=L, cmap="viridis", s=80, edgecolors="0.3", linewidths=0.6)
    fig3.colorbar(sc, ax=ax).set_label("n_layer (depth)")
    for i in range(len(rows)):
        ax.annotate(f"L{L[i]}×C{C[i]}", (width_over_depth[i], loss[i]), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.85)
    ax.scatter(
        [width_over_depth[i_best]],
        [loss[i_best]],
        s=200,
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=2.5,
        zorder=6,
    )
    ax.set_xlabel(r"Width/depth ratio  ($n_{\mathrm{embd}} / n_{\mathrm{layer}}$)")
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title("Same scan: width/depth vs loss")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig3.tight_layout()
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    # --- Summary text ---
    summary_path = out_dir / "depth_scan_summary.txt"
    lines = [
        f"CSV: {args.csv.resolve()}",
        f"Loss column: {args.loss_key}",
        "",
        f"Lowest loss row: n_layer={best['n_layer']}, n_embd={best['n_embd']}, n_head={best['n_head']}",
        f"  depth/width (L/C) = {int(best['n_layer']) / int(best['n_embd']):.6f}",
        f"  width/depth (C/L) = {int(best['n_embd']) / int(best['n_layer']):.4f}",
        f"  {args.loss_key} = {to_float(best, args.loss_key):.6f}",
        f"  AOFE = {to_float(best, 'aofe'):.6e}, AOFE-ratio = {to_float(best, 'aofe_ratio'):.6f}",
        "",
        "Sorted by loss (best first):",
    ]
    order = np.argsort(loss)
    for j in order:
        r = rows[int(j)]
        lines.append(
            f"  L={r['n_layer']:>2} C={r['n_embd']:>3}  L/C={int(r['n_layer'])/int(r['n_embd']):.5f}  "
            f"{args.loss_key}={to_float(r, args.loss_key):.6f}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote figures to {out_dir.resolve()}")
    print(f"Wrote summary   {summary_path}")
    print(lines[4])
    print(lines[5])
    print(lines[6])


if __name__ == "__main__":
    main()
