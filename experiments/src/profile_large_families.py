"""Per-family name-visibility profile for the largest corporate families.

Motivates the benchmark with named cases a reader can check by hand, and replaces three
hand-picked figures that were previously quoted from an unsaved API call.

Families are selected by parent-level obligated spend from the roster, not chosen by
hand, so the table cannot be accused of picking flattering examples. For each family the
children come from USAspending's `/recipient/children/` endpoint, and each child is
scored name-visible or name-invisible with the same `core_tokens` rule used for the
headline figure in `audit_ground_truth.py` - imported rather than reimplemented, so the
two can never drift apart.

Usage:
    python3 experiments/src/profile_large_families.py --top 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_ground_truth import core_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHILDREN = "https://api.usaspending.gov/api/v2/recipient/children/{uei}/"
HEADERS = {"User-Agent": "corpfam-benchmark/0.1"}


def get(session: requests.Session, url: str):
    backoff = 1.0
    for _ in range(5):
        try:
            r = session.get(url, headers=HEADERS, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, None
        except requests.RequestException:
            time.sleep(backoff)
            backoff *= 2
    return 0, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path,
                    default=ROOT / "data/raw/usaspending/recipients_contracts.jsonl")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--delay", type=float, default=0.4)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "experiments/results/large_family_profile.json")
    args = ap.parse_args()

    # Parent-level records carry the family rollup, which is the right ranking key:
    # we want the families that dominate spend, not the biggest standalone entities.
    parents: dict[str, dict] = {}
    with args.roster.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("recipient_level") != "P" or not r.get("uei"):
                continue
            amt = float(r.get("amount") or 0)
            prev = parents.get(r["uei"])
            if prev is None or amt > prev["amount"]:
                parents[r["uei"]] = {"uei": r["uei"], "name": r.get("name"),
                                     "amount": amt}

    ranked = sorted(parents.values(), key=lambda d: -d["amount"])[: args.top]
    print(f"parent-level entities: {len(parents):,}; profiling top {len(ranked)}\n",
          flush=True)

    session = requests.Session()
    rows = []
    for i, p in enumerate(ranked, 1):
        st, pl = get(session, CHILDREN.format(uei=p["uei"]))
        time.sleep(args.delay)
        if st != 200 or not isinstance(pl, list):
            print(f"  [{i}] {p['name']}: status {st}, skipped", flush=True)
            rows.append({**p, "status": st, "children": None})
            continue

        ptok = core_tokens(p["name"])
        kids, invisible, examples = 0, 0, []
        for c in pl:
            cname = c.get("name")
            if not cname:
                continue
            kids += 1
            if not (core_tokens(cname) & ptok):
                invisible += 1
                if len(examples) < 4:
                    examples.append(cname)

        row = {
            "parent_uei": p["uei"],
            "parent_name": p["name"],
            "family_obligated": p["amount"],
            "children": kids,
            "name_invisible": invisible,
            "pct_name_invisible": round(100 * invisible / kids, 2) if kids else None,
            "examples_name_invisible": examples,
        }
        rows.append(row)
        print(f"  [{i:2d}] {str(p['name'])[:38]:38} children={kids:>4} "
              f"invisible={invisible:>4} ({row['pct_name_invisible']}%)", flush=True)

    scored = [r for r in rows if r.get("children")]
    tot_k = sum(r["children"] for r in scored)
    tot_i = sum(r["name_invisible"] for r in scored)
    out = {
        "roster": str(args.roster),
        "parent_level_entities": len(parents),
        "families_profiled": len(scored),
        "children_total": tot_k,
        "name_invisible_total": tot_i,
        "pct_name_invisible_pooled": round(100 * tot_i / tot_k, 2) if tot_k else None,
        "note": "name-visibility uses core_tokens from audit_ground_truth.py; a child is "
                "name-invisible when it shares no distinctive token with its parent name",
        "families": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\npooled: {tot_i}/{tot_k} = {out['pct_name_invisible_pooled']}% name-invisible")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
