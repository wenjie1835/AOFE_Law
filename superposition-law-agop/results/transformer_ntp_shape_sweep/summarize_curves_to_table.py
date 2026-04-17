#!/usr/bin/env python3
"""
从 curves_N*/depthXXXX_dYYY.csv 文件名解析 depth、d_model，
读取每个 CSV 最后一行的 test_ce，输出 depth, d_model, alpha, final_test_ce。

用法:
  python summarize_curves_to_table.py
  python summarize_curves_to_table.py --curves_dir ./curves_N5000000
  python summarize_curves_to_table.py --curves_dir ./curves_N5000000 --out summary_n5000000.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

NAME_RE = re.compile(r"^depth(\d+)_d(\d+)\.csv$")


def final_test_ce(csv_path: Path) -> float:
    last: dict[str, str] | None = None
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    if last is None or "test_ce" not in last:
        raise ValueError(f"no data or missing test_ce: {csv_path}")
    return float(last["test_ce"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize NTP sweep curves to a small table.")
    ap.add_argument(
        "--curves_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "curves_N5000000",
        help="Directory containing depth*_d*.csv files (default: ./curves_N5000000 next to this script).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write CSV (UTF-8). If omitted, print to stdout only.",
    )
    args = ap.parse_args()
    curves_dir: Path = args.curves_dir
    if not curves_dir.is_dir():
        raise SystemExit(f"not a directory: {curves_dir}")

    rows: list[tuple[int, int, float, float]] = []
    for p in sorted(curves_dir.glob("depth*_d*.csv")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        depth = int(m.group(1))
        d_model = int(m.group(2))
        alpha = depth / d_model
        te = final_test_ce(p)
        rows.append((depth, d_model, alpha, te))

    if not rows:
        raise SystemExit(f"no matching depth*_d*.csv under {curves_dir}")

    rows.sort(key=lambda t: t[0])

    fieldnames = ["depth", "d_model", "alpha", "final_test_ce"]
    lines = []
    for depth, d_model, alpha, te in rows:
        lines.append(
            {
                "depth": depth,
                "d_model": d_model,
                "alpha": round(alpha, 6),
                "final_test_ce": round(te, 6),
            }
        )

    # stdout: human-readable
    print("\t".join(fieldnames))
    for row in lines:
        print(
            f"{row['depth']}\t{row['d_model']}\t{row['alpha']}\t{row['final_test_ce']}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(lines)
        print(f"\nWrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
