"""Derive corporate-family links from the USAspending bulk contract archive.

This is the second, independent derivation of the same parent-child links the recipient
API returns. Two things motivate it. The API throttles hard under sustained load - roughly
0.1 to 0.6 records per second, which puts the full 101,513-entity roster more than 40 hours
away - whereas one archive file carries a parent link on every award row. And a quantity
derived two ways, from two different publication pipelines, is a quantity that can be
reported; agreement between them is the check, disagreement is a finding.

The archive is award-level, so an entity recurs once per award and its parent link must be
reduced to one value per UEI. Where an entity's awards disagree about the parent, that
disagreement is recorded rather than resolved silently: it is direct evidence about how
stable these registrant-declared links are, which the paper needs.

Self-parent rows are preserved as-is here. Filtering belongs in the benchmark build, not in
extraction, so that the raw self-parent rate stays measurable from this output.

Usage:
    python3 experiments/src/extract_archive_parents.py \
        --zip data/raw/usaspending/archive/FY2025_All_Contracts_Full.zip \
        --out data/raw/usaspending/archive_parents_fy2025.jsonl
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

csv.field_size_limit(10 ** 9)

# Candidate spellings for each field we need. USAspending has renamed these across
# generations of the download format, so resolve against the actual header rather
# than trusting one spelling.
WANT = {
    "uei": ("recipient_uei", "recipient_duns_uei", "awardee_or_recipient_uei"),
    "name": ("recipient_name", "awardee_or_recipient_legal", "recipient_legal_name"),
    "parent_uei": ("recipient_parent_uei", "ultimate_parent_uei",
                   "recipient_parent_duns_uei"),
    "parent_name": ("recipient_parent_name", "ultimate_parent_legal_enti",
                    "ultimate_parent_name", "recipient_parent_legal_name"),
    "amount": ("federal_action_obligation", "total_obligated_amount"),
}


def resolve_columns(header: list[str]) -> dict[str, str]:
    lower = {h.lower().strip(): h for h in header}
    found: dict[str, str] = {}
    for key, options in WANT.items():
        for opt in options:
            if opt in lower:
                found[key] = lower[opt]
                break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", type=Path,
                    default=Path("experiments/results/archive_parents.json"))
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all; for smoke tests")
    args = ap.parse_args()

    if not args.zip.exists():
        print(f"missing {args.zip}", file=sys.stderr)
        return 1

    zf = zipfile.ZipFile(args.zip)
    members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
    print(f"csv members in archive: {len(members)}")
    for m in members:
        print(f"  {m}  ({zf.getinfo(m).file_size / 1e9:.2f} GB uncompressed)")

    # uei -> {parent_uei: [count, summed_obligation]}, plus the name last seen
    links: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    names: dict[str, str] = {}
    parent_names: dict[str, str] = {}
    rows_read = 0
    rows_with_uei = 0
    skipped_members: list[str] = []

    for member in members:
        with zf.open(member) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace",
                                      newline="")
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                continue
            cols = resolve_columns(header)
            if "uei" not in cols or "parent_uei" not in cols:
                skipped_members.append(member)
                print(f"  SKIP {member}: no uei/parent_uei column", flush=True)
                continue
            idx = {k: header.index(v) for k, v in cols.items()}
            print(f"  reading {member} using {cols}", flush=True)

            for row in reader:
                rows_read += 1
                if args.max_rows and rows_read > args.max_rows:
                    break
                try:
                    uei = row[idx["uei"]].strip()
                except IndexError:
                    continue
                if not uei:
                    continue
                rows_with_uei += 1
                puei = row[idx["parent_uei"]].strip() if "parent_uei" in idx else ""
                amt = 0.0
                if "amount" in idx:
                    try:
                        amt = float(row[idx["amount"]] or 0)
                    except ValueError:
                        amt = 0.0
                slot = links[uei][puei]
                slot[0] += 1
                slot[1] += amt
                if "name" in idx and row[idx["name"]].strip():
                    names[uei] = row[idx["name"]].strip()
                if puei and "parent_name" in idx and row[idx["parent_name"]].strip():
                    parent_names[puei] = row[idx["parent_name"]].strip()

                if rows_read % 2_000_000 == 0:
                    print(f"    {rows_read:,} rows, {len(links):,} distinct UEIs",
                          flush=True)

    # Reduce to one parent per UEI: the one backed by the most award rows, with
    # obligation as the tie-break. Ambiguity is recorded, not hidden.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ambiguous = 0
    self_parent = 0
    genuine = 0
    blank_parent = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for uei, cand in links.items():
            ranked = sorted(cand.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
            best_parent, (n_rows, amount) = ranked[0]
            distinct = [p for p in cand if p]
            if len(distinct) > 1:
                ambiguous += 1
            if not best_parent:
                blank_parent += 1
            elif best_parent == uei:
                self_parent += 1
            else:
                genuine += 1
            fh.write(json.dumps({
                "uei": uei,
                "name": names.get(uei),
                "parent_uei": best_parent or None,
                "parent_name": parent_names.get(best_parent) if best_parent else None,
                "award_rows": n_rows,
                "obligated": round(amount, 2),
                "distinct_parents_seen": len(distinct),
                "all_parents": {p: c[0] for p, c in cand.items() if p},
            }, ensure_ascii=False) + "\n")

    manifest = {
        "archive": str(args.zip),
        "csv_members": members,
        "skipped_members": skipped_members,
        "rows_read": rows_read,
        "rows_with_uei": rows_with_uei,
        "distinct_uei": len(links),
        "blank_parent": blank_parent,
        "self_parent": self_parent,
        "genuine_parent_links": genuine,
        "entities_with_conflicting_parents": ambiguous,
        "output": str(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
