#!/usr/bin/env python3
"""Parse ab_test.py --all log into per-sequence breakdown markdown.

Reads a log file produced by scripts/ab_test.py and emits:
- Aggregate stats (mean Raw vs IMM, delta, better/worse/same counts)
- Top-K and Bottom-K sequences by delta AUC
- Optional highlighted rows for target sequences

Usage:
    python scripts/parse_ab_log.py --log ab_test_run.log \
        --out reports/ab_log_breakdown.md \
        --highlight dataset5/person17_1 dataset5/group3_4 dataset4/person17
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROW_RE = re.compile(
    r"\[\s*(?P<idx>\d+)\s*/\s*(?P<total>\d+)\]\s+"
    r"(?P<seq>\S+)\s+"
    r"(?P<auc_r>-?\d+\.\d+)\s+(?P<auc_i>-?\d+\.\d+)\s+(?P<d_auc>[+-]?\d+\.\d+)\S?\s+"
    r"(?P<np_r>-?\d+\.\d+)\s+(?P<np_i>-?\d+\.\d+)\s+(?P<d_np>[+-]?\d+\.\d+)"
)


def parse_log(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            m = ROW_RE.search(line)
            if not m:
                continue
            rows.append({
                "idx": int(m["idx"]),
                "total": int(m["total"]),
                "seq": m["seq"],
                "auc_r": float(m["auc_r"]),
                "auc_i": float(m["auc_i"]),
                "d_auc": float(m["d_auc"]),
                "np_r": float(m["np_r"]),
                "np_i": float(m["np_i"]),
                "d_np": float(m["d_np"]),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    auc_r = sum(r["auc_r"] for r in rows) / n
    auc_i = sum(r["auc_i"] for r in rows) / n
    np_r = sum(r["np_r"] for r in rows) / n
    np_i = sum(r["np_i"] for r in rows) / n
    s_raw = 0.6 * auc_r + 0.4 * np_r
    s_imm = 0.6 * auc_i + 0.4 * np_i
    better = sum(1 for r in rows if r["d_auc"] > 0.005)
    worse = sum(1 for r in rows if r["d_auc"] < -0.005)
    same = n - better - worse
    return {
        "n": n, "auc_r": auc_r, "auc_i": auc_i, "np_r": np_r, "np_i": np_i,
        "s_raw": s_raw, "s_imm": s_imm, "delta": s_imm - s_raw,
        "better": better, "worse": worse, "same": same,
    }


def fmt_table_header() -> list[str]:
    return [
        "| # | Sequence | AUC_raw | AUC_imm | dAUC | NP_raw | NP_imm | dNP |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]


def fmt_row(r: dict) -> str:
    return (f"| {r['idx']} | `{r['seq']}` | {r['auc_r']:.3f} | {r['auc_i']:.3f} | "
            f"{r['d_auc']:+.3f} | {r['np_r']:.3f} | {r['np_i']:.3f} | {r['d_np']:+.3f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse ab_test log into markdown")
    parser.add_argument("--log", required=True, help="Path to ab_test log file")
    parser.add_argument("--out", required=True, help="Output markdown path")
    parser.add_argument("--top", type=int, default=10, help="Top/bottom K sequences to list")
    parser.add_argument("--highlight", nargs="*", default=[],
                        help="Sequence ids to highlight in a dedicated section")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        print(f"[ERROR] No parseable rows in {args.log}", file=sys.stderr)
        return 1

    agg = aggregate(rows)
    by_dauc = sorted(rows, key=lambda r: r["d_auc"])
    worst = by_dauc[: args.top]
    best = by_dauc[-args.top:][::-1]

    lines: list[str] = []
    lines.append(f"# AB Log Breakdown — `{os.path.basename(args.log)}`\n")
    lines.append(f"**Source**: `{args.log}`  ")
    lines.append(f"**Sequences parsed**: {agg['n']}\n")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- AUC_raw mean: **{agg['auc_r']:.4f}**")
    lines.append(f"- AUC_imm mean: **{agg['auc_i']:.4f}**")
    lines.append(f"- NP_raw mean: **{agg['np_r']:.4f}**")
    lines.append(f"- NP_imm mean: **{agg['np_i']:.4f}**")
    lines.append(f"- FinalScore (raw, S_acc only): **{agg['s_raw']:.4f}**")
    lines.append(f"- FinalScore (IMM, S_acc only): **{agg['s_imm']:.4f}**")
    lines.append(f"- Δ (IMM − raw): **{agg['delta']:+.4f}**")
    lines.append(f"- IMM better: **{agg['better']}/{agg['n']}**, worse: **{agg['worse']}/{agg['n']}**, "
                 f"same: **{agg['same']}/{agg['n']}**\n")

    if args.highlight:
        lines.append("## Highlighted target sequences\n")
        lines.extend(fmt_table_header())
        wanted = set(args.highlight)
        for r in rows:
            if r["seq"] in wanted:
                lines.append(fmt_row(r))
        lines.append("")

    lines.append(f"## Worst {args.top} sequences for IMM (lowest dAUC)\n")
    lines.extend(fmt_table_header())
    for r in worst:
        lines.append(fmt_row(r))
    lines.append("")

    lines.append(f"## Best {args.top} sequences for IMM (highest dAUC)\n")
    lines.extend(fmt_table_header())
    for r in best:
        lines.append(fmt_row(r))
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {args.out} ({agg['n']} rows, Δ={agg['delta']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
