"""Characterise the USAspending recipient roster before building anything on it.

Answers, from the data rather than from assumption:
  - how the roster splits across P (parent), C (child) and R (neither)
  - how spend concentrates, since the evaluation is spend-weighted
  - how many parents there are, which sets the cost of fetching family structure
  - how much name variation exists, which is what a string baseline has to work with

Writes experiments/results/roster_characterisation.json. Every figure in the paper
that describes the corpus should come from that file, not from this script's stdout.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROSTER = os.path.join(ROOT, "data", "raw", "usaspending", "recipients_contracts.jsonl")
OUT = os.path.join(ROOT, "experiments", "results", "roster_characterisation.json")

LEGAL_SUFFIX = re.compile(
    r"\b(incorporated|corporation|company|limited|inc|corp|co|llc|llp|lp|ltd|plc|"
    r"pllc|pc|pa|na|sa|nv|bv|gmbh|ag|spa|srl|pty|pte|kk)\b")
PUNCT = re.compile(r"[^a-z0-9 ]")
WS = re.compile(r"\s+")


def norm(name: str) -> str:
    return WS.sub(" ", PUNCT.sub(" ", name.lower())).strip()


def norm_stripped(name: str) -> str:
    return WS.sub(" ", LEGAL_SUFFIX.sub(" ", norm(name))).strip()


def pct(a: float, b: float) -> float:
    return round(a / b * 100, 4) if b else 0.0


def main() -> int:
    rows = []
    with open(ROSTER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    levels = Counter(r.get("recipient_level") for r in rows)

    def amt(r) -> float:
        v = r.get("amount")
        return float(v) if v is not None else 0.0

    total_amount = sum(amt(r) for r in rows)
    positive = [r for r in rows if amt(r) > 0]
    negative = [r for r in rows if amt(r) < 0]

    by_level_amount = {}
    for lvl in levels:
        by_level_amount[str(lvl)] = round(sum(amt(r) for r in rows
                                              if r.get("recipient_level") == lvl), 2)

    # UEI health. A missing UEI means the record cannot be joined to anything.
    no_uei = [r for r in rows if not r.get("uei")]
    ueis = [r["uei"] for r in rows if r.get("uei")]
    uei_dupes = sum(c - 1 for c in Counter(ueis).values() if c > 1)

    # Spend concentration drives the spend-weighted metric, so measure it explicitly.
    amounts = sorted((amt(r) for r in rows), reverse=True)
    pos_total = sum(a for a in amounts if a > 0)
    concentration = {}
    running = 0.0
    marks = {10: None, 100: None, 1000: None, 10000: None, 50000: None}
    for i, a in enumerate(amounts, start=1):
        if a > 0:
            running += a
        if i in marks:
            concentration[f"top_{i}_share_of_positive_spend_pct"] = pct(running, pos_total)

    # How much name variation is there for a string baseline to exploit or trip over?
    names = [r.get("name") or "" for r in rows]
    n_raw = len(set(names))
    normed = [norm(x) for x in names]
    stripped = [norm_stripped(x) for x in names]
    n_norm = len(set(normed))
    n_strip = len(set(stripped))

    # Collisions matter more than counts: two different UEIs sharing a normalised name
    # is either a genuine duplicate registration or a string baseline's false positive.
    norm_to_uei = {}
    for r, nm in zip(rows, normed):
        if r.get("uei"):
            norm_to_uei.setdefault(nm, set()).add(r["uei"])
    norm_collisions = {k: len(v) for k, v in norm_to_uei.items() if len(v) > 1}

    strip_to_uei = {}
    for r, nm in zip(rows, stripped):
        if r.get("uei"):
            strip_to_uei.setdefault(nm, set()).add(r["uei"])
    strip_collisions = {k: len(v) for k, v in strip_to_uei.items() if len(v) > 1}

    parents = [r for r in rows if r.get("recipient_level") == "P"]
    children = [r for r in rows if r.get("recipient_level") == "C"]

    result = {
        "source_file": os.path.relpath(ROSTER, ROOT),
        "rows": n,
        "levels": {str(k): v for k, v in levels.most_common()},
        "levels_pct": {str(k): pct(v, n) for k, v in levels.most_common()},
        "amount": {
            "total": round(total_amount, 2),
            "positive_total": round(pos_total, 2),
            "rows_with_positive_amount": len(positive),
            "rows_with_negative_amount": len(negative),
            "by_level": by_level_amount,
        },
        "concentration": concentration,
        "uei": {
            "rows_without_uei": len(no_uei),
            "distinct_uei": len(set(ueis)),
            "rows_sharing_a_uei_with_another_row": uei_dupes,
        },
        "names": {
            "distinct_raw": n_raw,
            "distinct_normalised": n_norm,
            "distinct_suffix_stripped": n_strip,
            "collapsed_by_normalisation": n_raw - n_norm,
            "collapsed_by_suffix_stripping": n_norm - n_strip,
            "normalised_names_shared_by_multiple_uei": len(norm_collisions),
            "uei_pairs_colliding_after_normalisation": sum(
                v * (v - 1) // 2 for v in norm_collisions.values()),
            "suffix_stripped_names_shared_by_multiple_uei": len(strip_collisions),
            "uei_pairs_colliding_after_suffix_stripping": sum(
                v * (v - 1) // 2 for v in strip_collisions.values()),
        },
        "family_fetch_cost": {
            "parents_to_query": len(parents),
            "children_expected": len(children),
            "est_serial_minutes_at_0.45s_each": round(len(parents) * 0.45 / 60, 1),
        },
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))

    print("\n--- largest normalised-name collisions (distinct UEIs sharing a name) ---")
    for name, cnt in sorted(norm_collisions.items(), key=lambda x: -x[1])[:12]:
        print(f"  {cnt:>3} UEIs  '{name}'")

    print(f"\nresults -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
