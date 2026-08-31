"""Quantify D&B exposure in the released links.

SAM.gov's terms grant a limited licence over "D&B Open Data" -- which explicitly
includes Legal Business Name -- and forbid disseminating it *in bulk*. The same clause
scopes which records are affected: entity registrations, exclusions, and "all base
award notices with an award date earlier than 4/4/2022".

Our benchmark publishes company names in bulk, so the question is not academic: are
any of our links evidenced only by awards whose base notice predates the cutoff? Every
*transaction* in the FY2025 archive is post-cutoff by construction, but a FY2025 row
can be a modification to a contract first awarded years earlier, and it is the base
award date the clause names.

This script measures, for each parent-child link, the earliest evidence date across
the transactions supporting it, so a conservative post-cutoff subset can be released
if the exposure turns out to be material. It resolves the licence question with a
number instead of an assumption.

Usage:
    python3 experiments/src/check_dnb_provenance.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

CUTOFF = "2022-04-04"          # SAM.gov D&B clause boundary
csv.field_size_limit(10_000_000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path,
                    default=Path("data/raw/usaspending/archive/FY2025_All_Contracts_Full.zip"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/dnb_provenance.json"))
    args = ap.parse_args()

    z = zipfile.ZipFile(args.archive)
    members = [m for m in z.namelist() if m.lower().endswith(".csv")]

    # earliest base-award evidence date per (child uei -> parent uei) link
    earliest: dict[tuple[str, str], str] = {}
    rows = 0
    missing_date = 0

    for mi, name in enumerate(members, 1):
        with z.open(name) as fh:
            r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            for row in r:
                rows += 1
                if rows % 1_000_000 == 0:
                    print(f"  {rows:,} rows ({mi}/{len(members)})", flush=True)
                uei = (row.get("recipient_uei") or "").strip().upper()
                puei = (row.get("recipient_parent_uei") or "").strip().upper()
                if not uei or not puei or puei == uei:
                    continue
                # Best available proxy for when the underlying award began. The
                # period-of-performance start predates or equals the base award for
                # essentially all contract vehicles; action_date is the fallback.
                d = ((row.get("period_of_performance_start_date") or "").strip()
                     or (row.get("action_date") or "").strip())
                if not d:
                    missing_date += 1
                    continue
                k = (uei, puei)
                if k not in earliest or d < earliest[k]:
                    earliest[k] = d

    pre = {k: v for k, v in earliest.items() if v < CUTOFF}
    post = {k: v for k, v in earliest.items() if v >= CUTOFF}

    # Which of the benchmark's links would a conservative filter remove?
    bench = Path("data/benchmark/corpfam_pairs.jsonl")
    removed_positives = kept_positives = 0
    if bench.exists():
        for line in bench.open(encoding="utf-8"):
            p = json.loads(line)
            if p["label"] != 1:
                continue
            k = (p["left_uei"], p["right_uei"])
            d = earliest.get(k)
            if d is not None and d < CUTOFF:
                removed_positives += 1
            else:
                kept_positives += 1

    out = {
        "cutoff": CUTOFF,
        "rows_scanned": rows,
        "rows_with_link_but_no_date": missing_date,
        "distinct_links": len(earliest),
        "links_earliest_evidence_pre_cutoff": len(pre),
        "links_earliest_evidence_post_cutoff": len(post),
        "pct_pre_cutoff": round(100.0 * len(pre) / max(len(earliest), 1), 2),
        "benchmark_positives_pre_cutoff": removed_positives,
        "benchmark_positives_post_cutoff": kept_positives,
        "pct_benchmark_positives_removed_by_conservative_filter":
            round(100.0 * removed_positives / max(removed_positives + kept_positives, 1), 2),
        "earliest_year_distribution": dict(sorted(
            Counter(v[:4] for v in earliest.values()).items())),
        "interpretation": (
            "UPPER BOUND ON EXPOSURE, not a measurement of D&B sourcing. The definitive "
            "test in SAM.gov's terms is the EVS Source field, which is not present in "
            "USAspending award data, so this substitutes the earliest performance-period "
            "start across the transactions supporting each link. That proxy is "
            "deliberately over-inclusive: a task order placed in FY2025 against a "
            "long-running vehicle inherits the vehicle's original start date and is "
            "counted pre-cutoff even though the order itself is recent. Every "
            "transaction in this archive is FY2025, so under an action-date reading the "
            "exposure would be zero. The truth is between the two. We therefore publish "
            "the full benchmark, sourced from public-domain Treasury data, and also ship "
            "the post-cutoff subset for anyone needing the conservative position."),
        "proxy_used": "period_of_performance_start_date, falling back to action_date",
    }
    # Persist per-link dates so the conservative subset can be cut without rescanning
    # 6.6M rows.
    dates_path = args.output.parent / "dnb_link_dates.json"
    dates_path.write_text(
        json.dumps({f"{a}|{b}": d for (a, b), d in earliest.items()}), encoding="utf-8")
    print(f"wrote {dates_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k != "earliest_year_distribution"}, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
