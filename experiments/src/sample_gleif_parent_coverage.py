"""Measure whether GLEIF can independently confirm the corporate-family links we hold.

The decision this informs is narrow: should GLEIF be a *primary* ground-truth source for
CorpFam, or only a cross-validation source? The globally interesting statistic ("what
share of all LEIs report a parent") turns out to be unobtainable through the API anyway -
GLEIF refuses page-based pagination beyond 10,000 results, so there is no way to draw a
uniform sample over the 3.4M-record index by page offset. An earlier version of this
script tried exactly that and looped forever on HTTP 400.

The better question is also the cheaper one. Take the parent-child links we already have
from USAspending and ask, for each, whether GLEIF can reproduce it. That yields a chain of
three conditional rates:

    child locatable in GLEIF by name
      -> of those, a parent relationship is recorded
        -> of those, the parent GLEIF names agrees with the parent SAM names

Each stage can independently disqualify GLEIF as a primary source, and the third stage is
the only one that would qualify it.

A caveat that is itself a finding: GLEIF is keyed by LEI, and we hold no LEIs, so reaching
a GLEIF record at all requires matching on name - the task this benchmark exists to
evaluate. Using GLEIF as ground truth therefore presupposes a solution to the problem
under study. To keep that circularity from inflating the result, only an exact match on
normalised legal name counts as located; fuzzy candidates are counted separately and never
treated as found.

Usage:
    python3 experiments/src/sample_gleif_parent_coverage.py --n 150
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import requests

BASE = "https://api.gleif.org/api/v1"
HEADERS = {"Accept": "application/vnd.api+json", "User-Agent": "corpfam-benchmark/0.1"}

LEGAL_SUFFIX = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "plc", "gmbh", "sa", "nv", "ag", "lc", "pc",
}


def norm_name(s: str) -> str:
    """Lowercase, collapse punctuation and whitespace, drop trailing legal suffixes.

    Trailing suffixes only: "CORPORATION MASTER TRUST" must not normalise to
    "master trust", and an interior "co" is usually part of the name.
    """
    toks = re.findall(r"[a-z0-9]+", str(s).lower())
    while toks and toks[-1] in LEGAL_SUFFIX:
        toks.pop()
    return " ".join(toks)


def get(session: requests.Session, url: str, params: dict | None = None):
    """Return (status, json_or_None). Retries only genuinely transient statuses.

    4xx other than 429 are terminal by design: retrying a 400 is what made the
    previous version of this script hang.
    """
    backoff = 1.0
    for _ in range(4):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
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


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 2), round(100 * min(1.0, c + h), 2)]


def legal_name(rec: dict) -> str:
    return (((rec.get("attributes") or {}).get("entity") or {})
            .get("legalName") or {}).get("name") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path,
                    default=Path("data/raw/usaspending/entity_detail.jsonl"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--delay", type=float, default=0.25)
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/gleif_cross_validation.json"))
    args = ap.parse_args()

    records = [json.loads(l) for l in args.detail.open(encoding="utf-8") if l.strip()]
    genuine = [
        r for r in records
        if r.get("recipient_level") == "C"
        and r.get("parent_uei") and r.get("uei") and r["parent_uei"] != r["uei"]
        and r.get("name") and r.get("parent_name")
    ]
    informative = [r for r in genuine if r["name"] != r["parent_name"]]
    print(f"genuine links available: {len(genuine)}  informative: {len(informative)}",
          flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(informative, min(args.n, len(informative)))

    session = requests.Session()
    located = has_parent = parent_agrees = fuzzy_only = 0
    rows = []

    for i, r in enumerate(sample, 1):
        child, parent = r["name"], r["parent_name"]
        st, pl = get(session, f"{BASE}/lei-records",
                     {"filter[entity.legalName]": child, "page[size]": 5})
        time.sleep(args.delay)

        lei = None
        cand = (pl or {}).get("data") or [] if st == 200 else []
        target = norm_name(child)
        for c in cand:
            if norm_name(legal_name(c)) == target:
                lei = c["id"]
                break
        if lei is None and cand:
            fuzzy_only += 1

        row = {"child": child, "sam_parent": parent, "lei": lei,
               "gleif_candidates": len(cand)}

        if lei:
            located += 1
            gparent = None
            for ep in ("direct-parent", "ultimate-parent"):
                sp, pp = get(session, f"{BASE}/lei-records/{lei}/{ep}")
                time.sleep(args.delay)
                if sp == 200 and pp and pp.get("data"):
                    d = pp["data"]
                    d = d[0] if isinstance(d, list) else d
                    gparent = legal_name(d) or gparent
                    if ep == "direct-parent" and gparent:
                        break
            if gparent:
                has_parent += 1
                agree = norm_name(gparent) == norm_name(parent)
                parent_agrees += agree
                row.update({"gleif_parent": gparent, "agrees": agree})

        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}  located={located} "
                  f"has_parent={has_parent} agrees={parent_agrees}", flush=True)

    n = len(sample)
    out = {
        "source_file": str(args.detail),
        "genuine_links_available": len(genuine),
        "informative_links_available": len(informative),
        "sampled": n,
        "seed": args.seed,
        "method": "exact match on normalised legal name only; fuzzy candidates counted "
                  "but never treated as located",
        "stage1_child_located_in_gleif": {
            "count": located, "pct": round(100 * located / n, 2) if n else None,
            "wilson95_pct": wilson(located, n),
        },
        "stage2_of_located_has_parent": {
            "count": has_parent,
            "pct": round(100 * has_parent / located, 2) if located else None,
            "wilson95_pct": wilson(has_parent, located),
        },
        "stage3_of_those_parent_matches_sam": {
            "count": parent_agrees,
            "pct": round(100 * parent_agrees / has_parent, 2) if has_parent else None,
            "wilson95_pct": wilson(parent_agrees, has_parent),
        },
        "end_to_end_confirmed_pct": round(100 * parent_agrees / n, 2) if n else None,
        "name_matched_only_fuzzily_not_counted": fuzzy_only,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
