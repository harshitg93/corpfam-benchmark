"""Extract per-entity attributes from the FY2025 award archive.

Every baseline so far compares two names, which invites a fair objection: of course
name-only methods fail on pairs selected for having dissimilar names, and real entity
resolution uses addresses and other attributes. Standard benchmarks such as Magellan
and WDC Products are multi-attribute for exactly this reason.

The archive carries recipient address, city, state, ZIP, country, phone, and a
doing-business-as name on every award row, for all 104k entities -- far wider coverage
than the 4.8k-entity API sample. This pulls those fields per UEI so a genuine
multi-attribute baseline can be run.

Where an entity's rows disagree on a field, the value backed by the most obligated
dollars wins, the same conflict rule used for parent assignment. Disagreement is itself
recorded, since an entity whose address changes across the year is a different kind of
object from one whose address is stable.

Usage:
    python3 experiments/src/extract_entity_attributes.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10_000_000)

FIELDS = {
    "address": "recipient_address_line_1",
    "city": "recipient_city_name",
    "state": "recipient_state_code",
    "zip": "recipient_zip_4_code",
    "country": "recipient_country_code",
    "county": "recipient_county_name",
    "phone": "recipient_phone_number",
    "dba": "recipient_doing_business_as_name",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path,
                    default=Path("data/raw/usaspending/archive/FY2025_All_Contracts_Full.zip"))
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/raw/usaspending/entity_attributes.jsonl"))
    ap.add_argument("--summary", type=Path,
                    default=Path("experiments/results/entity_attributes.json"))
    args = ap.parse_args()

    wanted: set[str] = set()
    for line in args.pairs.open(encoding="utf-8"):
        p = json.loads(line)
        wanted.add(p["left_uei"])
        wanted.add(p["right_uei"])
    print(f"{len(wanted):,} entities appear in the benchmark", flush=True)

    # uei -> field -> value -> obligated dollars backing that value
    acc: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    rows = 0
    z = zipfile.ZipFile(args.archive)
    for name in [m for m in z.namelist() if m.lower().endswith(".csv")]:
        with z.open(name) as fh:
            r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            for row in r:
                rows += 1
                if rows % 1_000_000 == 0:
                    print(f"  {rows:,} rows", flush=True)
                uei = (row.get("recipient_uei") or "").strip().upper()
                if not uei or uei not in wanted:
                    continue
                try:
                    amt = abs(float(row.get("federal_action_obligation") or 0.0))
                except ValueError:
                    amt = 0.0
                weight = amt if amt > 0 else 1.0
                for key, col in FIELDS.items():
                    v = (row.get(col) or "").strip()
                    if v:
                        acc[uei][key][v] += weight

    cov: Counter = Counter()
    conflict: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for uei, fields in acc.items():
            rec = {"uei": uei}
            for key in FIELDS:
                c = fields.get(key)
                if c:
                    rec[key] = c.most_common(1)[0][0]
                    cov[key] += 1
                    if len(c) > 1:
                        conflict[key] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(acc)
    summary = {
        "rows_scanned": rows,
        "benchmark_entities": len(wanted),
        "entities_with_any_attribute": n,
        "entity_coverage_pct": round(100.0 * n / len(wanted), 2),
        "field_coverage": {
            k: {"entities": cov[k],
                "pct_of_benchmark": round(100.0 * cov[k] / len(wanted), 2),
                "pct_with_conflicting_values": (
                    round(100.0 * conflict[k] / cov[k], 2) if cov[k] else None)}
            for k in FIELDS},
        "conflict_rule": "value backed by the most obligated dollars wins",
        "output": str(args.out),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
