#!/usr/bin/env python3
"""
Visualize depth-scan CSVs from transformer_shape_scan.py, mlp_shape_scan.py,
or cnn_shape_scan.py: loss, AOFE, AOFE-ratio vs depth, and depth/width vs loss.

Schema is auto-detected from column names.

Usage:
  python plot_depth_scan_results.py \\
    --csv results_tiny_gpt_depth_aofe/depth_scan_results.csv \\
    --out-dir results_tiny_gpt_depth_aofe/figures

  python plot_depth_scan_results.py \\
    --csv results_mlp_teacher_student_depth_scan/depth_scan_results.csv \\
    --out-dir results_mlp_teacher_student_depth_scan/figures

  python plot_depth_scan_results.py \\
    --csv results_cnn_autoencoder_depth_scan/depth_scan_results.csv \\
    --out-dir results_cnn_autoencoder_depth_scan/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def detect_scan_schema(rows: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """Return (kind, depth_col, width_col) with kind in {'gpt','mlp','cnn'}."""
    keys = set(rows[0].keys())
    if "hidden_layers" in keys and "hidden_width" in keys:
        return "mlp", "hidden_layers", "hidden_width"
    if "blocks" in keys and "width" in keys:
        return "cnn", "blocks", "width"
    if "n_layer" in keys and "n_embd" in keys:
        return "gpt", "n_layer", "n_embd"
    raise SystemExit(
        "Unknown CSV schema: need (n_layer, n_embd), (hidden_layers, hidden_width), or (blocks, width)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot depth scan CSV from transformer / MLP / CNN shape scans"
    )
    parser.add_argument("--csv", type=Path, required=True, help="depth_scan_results.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Figure output directory (default: csv parent / figures)")
    parser.add_argument("--loss-key", choices=("final_val_loss", "best_val_loss"), default="final_val_loss")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("Empty CSV")

    kind, depth_k, width_k = detect_scan_schema(rows)

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = np.array([int(r[depth_k]) for r in rows])
    C = np.array([int(r[width_k]) for r in rows])
    loss = np.array([to_float(r, args.loss_key) for r in rows])
    aofe = np.array([to_float(r, "aofe") for r in rows])
    ratio = np.array([to_float(r, "aofe_ratio") for r in rows])

    # Depth/width aspect ratios (both common conventions)
    depth_over_width = L / np.maximum(C, 1)  # "深/宽"
    width_over_depth = C / np.maximum(L, 1)  # "宽/深"

    i_best = int(np.argmin(loss))
    best = rows[i_best]

    if kind == "mlp":
        depth_xlabel = "Depth (hidden layers)"
        cbar_depth_label = "hidden_layers"
        ann = lambda i: f"L{L[i]}×W{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Depth/width ratio  ($L_{\mathrm{hid}} / W_{\mathrm{hid}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/depth ratio  ($W_{\mathrm{hid}} / L_{\mathrm{hid}}$)"

        if "param_count" in rows[0]:
            pm = int(np.median([int(r["param_count"]) for r in rows]))
            aspect_title = f"Find shape by depth/width (median ~{pm:,} trainable params)"
        else:
            aspect_title = "Find shape by depth/width ratio"
    elif kind == "cnn":
        depth_xlabel = "Depth (residual blocks)"
        cbar_depth_label = "blocks"
        ann = lambda i: f"B{L[i]}×W{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Blocks/width ratio  ($B / W_{\mathrm{ch}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/blocks ratio  ($W_{\mathrm{ch}} / B$)"

        if "param_count" in rows[0]:
            pm = int(np.median([int(r["param_count"]) for r in rows]))
            aspect_title = f"CNN autoencoder: shape by blocks/width (median ~{pm:,} params)"
        else:
            aspect_title = "CNN autoencoder: blocks vs channel width"
    else:
        depth_xlabel = "Depth (n_layer)"
        cbar_depth_label = "n_layer"
        ann = lambda i: f"L{L[i]}×C{C[i]}"

        def xl_depth_over_width() -> str:
            return r"Depth/width ratio  ($n_{\mathrm{layer}} / n_{\mathrm{embd}}$)"

        def xl_width_over_depth() -> str:
            return r"Width/depth ratio  ($n_{\mathrm{embd}} / n_{\mathrm{layer}}$)"

        aspect_title = "Find shape by depth/width ratio (fixed ~3M params)"

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
    axes[2].set_xlabel(depth_xlabel)
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
    cbar.set_label(f"{cbar_depth_label} (depth)")
    for i, r in enumerate(rows):
        ax.annotate(
            ann(i),
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
    ax.set_xlabel(xl_depth_over_width())
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title(aspect_title)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_dir / "depth_scan_loss_vs_depth_over_width.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # --- Figure 3: width/depth vs loss (alternative aspect view) ---
    fig3, ax = plt.subplots(figsize=(6.8, 5))
    sc = ax.scatter(width_over_depth, loss, c=L, cmap="viridis", s=80, edgecolors="0.3", linewidths=0.6)
    fig3.colorbar(sc, ax=ax).set_label(f"{cbar_depth_label} (depth)")
    for i in range(len(rows)):
        ax.annotate(ann(i), (width_over_depth[i], loss[i]), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.85)
    ax.scatter(
        [width_over_depth[i_best]],
        [loss[i_best]],
        s=200,
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=2.5,
        zorder=6,
    )
    ax.set_xlabel(xl_width_over_depth())
    ax.set_ylabel(args.loss_key.replace("_", " "))
    ax.set_title("Same scan: width/depth vs loss")
    ax.grid(True, linestyle=":", alpha=0.7)
    fig3.tight_layout()
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig3.savefig(out_dir / "depth_scan_loss_vs_width_over_depth.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    # --- Summary text ---
    summary_path = out_dir / "depth_scan_summary.txt"
    d_b, w_b = int(best[depth_k]), int(best[width_k])
    lines = [
        f"CSV: {args.csv.resolve()}",
        f"Schema: {kind}",
        f"Loss column: {args.loss_key}",
        "",
    ]
    if kind == "mlp":
        pc = best.get("param_count", "")
        lines.append(
            f"Lowest loss row: hidden_layers={best[depth_k]}, hidden_width={best[width_k]}, param_count={pc}"
        )
    elif kind == "cnn":
        pc = best.get("param_count", "")
        lines.append(
            f"Lowest loss row: blocks={best[depth_k]}, width={best[width_k]}, param_count={pc}"
        )
    else:
        n_head = best.get("n_head", "N/A")
        lines.append(
            f"Lowest loss row: n_layer={best['n_layer']}, n_embd={best['n_embd']}, n_head={n_head}"
        )
    lines.extend(
        [
            f"  depth/width (L/W) = {d_b / max(w_b, 1):.6f}",
            f"  width/depth (W/L) = {w_b / max(d_b, 1):.4f}",
            f"  {args.loss_key} = {to_float(best, args.loss_key):.6f}",
            f"  AOFE = {to_float(best, 'aofe'):.6e}, AOFE-ratio = {to_float(best, 'aofe_ratio'):.6f}",
            "",
            "Sorted by loss (best first):",
        ]
    )
    order = np.argsort(loss)
    for j in order:
        r = rows[int(j)]
        d_i, w_i = int(r[depth_k]), int(r[width_k])
        if kind == "mlp":
            lines.append(
                f"  L={d_i:>2} W={w_i:>4}  L/W={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}"
            )
        elif kind == "cnn":
            lines.append(
                f"  B={d_i:>2} W={w_i:>4}  B/W={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}"
            )
        else:
            lines.append(
                f"  L={d_i:>2} C={w_i:>3}  L/C={d_i / max(w_i, 1):.5f}  "
                f"{args.loss_key}={to_float(r, args.loss_key):.6f}"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote figures to {out_dir.resolve()}")
    print(f"Wrote summary   {summary_path}")
    for ln in lines[4:8]:
        print(ln)


if __name__ == "__main__":
    main()
