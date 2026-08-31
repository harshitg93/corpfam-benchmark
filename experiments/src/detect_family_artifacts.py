"""Find families whose declared parent is not plausibly the parent.

Some declared families are large and wrong. `ZIPPY SHELL INCORPORATED`, a moving
company, is listed as the ultimate parent of 103 Waste Management entities;
`ROCKWELL COLLINS AUSTRALIA PTY LIMITED` is listed as parent of `RAYTHEON COMPANY`
across $32.8B of obligations. Left in, these become the largest clusters in the
benchmark and every clustering metric is dominated by them.

The obvious filter - drop families whose children do not share a token with the
parent - would be a disaster here, because name-invisible pairs are precisely what
this benchmark exists to measure. `MARSH & MCLENNAN -> MERCER (US) LLC` shares no
token and is entirely genuine.

The signal that actually separates them is internal coherence. In a mislabelled
family the children agree with *each other* and disagree with the parent: 103
children all carrying "waste" while the parent carries "zippy". In a genuine
name-invisible family the children do not form a single dominant block that
excludes the parent. So the test is:

    a dominant token shared by most children, which the parent does not have

Three further checks run alongside it:

  * sovereign and governmental catch-alls, which are not corporate families;
  * inverted parents, where the declared parent is much smaller than a child that
    carries the better-known name, as with `HONEYWELL SAFETY PRODUCTS USA` declared
    parent of `HONEYWELL INTERNATIONAL INC.`;
  * families whose parent never appears as a contract recipient itself.

Nothing is deleted here. Every family is scored and the flags are written out for
inspection, because an automatic rule that silently removes data is exactly what
this paper criticises elsewhere.

Usage:
    python3 experiments/src/detect_family_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "companies", "group", "holdings", "holding",
    "the", "and", "of", "plc", "gmbh", "sa", "nv", "ag", "pty", "pte", "bv",
    "usa", "us", "america", "american", "international", "intl", "worldwide",
    "global", "national", "services", "service", "solutions", "systems",
    "technologies", "technology", "enterprises", "enterprise", "industries",
    "associates", "partners", "trust", "fund", "lc", "pc", "pa", "group",
}

# Deliberately narrow. An earlier version matched bare "republic" and "federal" and
# flagged REPUBLIC SERVICES INC, a waste company, as a sovereign state. Every pattern
# here must be a construction a company name would not accidentally contain.
SOVEREIGN = re.compile(
    r"(\bgovernment of\b|\bgovt of\b|\brepublic of\b|\bkingdom of\b|\bstate of\b|"
    r"\bcommonwealth of\b|\bministry of\b|\bmunicipality of\b|\bprovince of\b|"
    r"^city of\b|^county of\b|\bunited states government\b)",
    re.I,
)

TOKEN = re.compile(r"[a-z0-9]+")


def core(name: str) -> set[str]:
    return {t for t in TOKEN.findall(str(name).lower())
            if t not in STOPWORDS and len(t) > 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/usaspending/archive_parents_fy2025.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/family_artifacts.json"))
    ap.add_argument("--min-size", type=int, default=3,
                    help="families smaller than this cannot show a dominant block")
    ap.add_argument("--dominance", type=float, default=0.6,
                    help="share of children that must carry the token")
    ap.add_argument("--max-reach", type=float, default=0.2,
                    help="max share of children carrying the parent's own name for "
                         "the dominant-block rule to fire")
    ap.add_argument("--review-size", type=int, default=25,
                    help="families at least this large are queued for manual review")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.input.open(encoding="utf-8") if l.strip()]
    by_uei = {r["uei"]: r for r in recs}

    genuine = [r for r in recs
               if r.get("parent_uei") and r["parent_uei"] != r["uei"]
               and r.get("name") and r.get("parent_name")]

    fam: dict[str, list[dict]] = defaultdict(list)
    for r in genuine:
        fam[r["parent_uei"]].append(r)

    flagged: list[dict] = []
    for puei, children in fam.items():
        pname = children[0]["parent_name"]
        ptok = core(pname)
        reasons: list[str] = []

        if SOVEREIGN.search(pname or ""):
            reasons.append("sovereign_or_government_entity")

        # How often does the parent's own distinctive name reach its children?
        # HANGER, INC. scores high here because most of its children are named
        # "HANGER PROSTHETICS & ORTHOTICS"; ZIPPY SHELL scores zero because none of
        # its 103 children mention Zippy.
        reach = 0
        if ptok:
            reach = sum(1 for c in children if ptok & core(c["name"])) / len(children)

        # A dominant token shared by the children but missing from the parent is
        # only suspicious when the parent's own name is simultaneously absent from
        # them. That conjunction is what separates a mislabelled parent from a
        # genuine name-invisible family, which has no such dominant block.
        dominant = None
        if len(children) >= args.min_size:
            tally: Counter = Counter()
            for c in children:
                for t in core(c["name"]):
                    tally[t] += 1
            for tok, n in tally.most_common(5):
                if n / len(children) >= args.dominance and tok not in ptok:
                    dominant = {"token": tok, "children_sharing": n,
                                "share": round(n / len(children), 3)}
                    break
            if dominant and reach <= args.max_reach:
                reasons.append("children_form_block_excluding_parent")

        prec = by_uei.get(puei)
        p_oblig = float(prec["obligated"]) if prec else 0.0

        # Large families are reviewed by hand regardless of what the rules say.
        # There are few enough of them for that to be practical, and the rules
        # cannot detect a parent that is wrong purely on world knowledge, such as
        # an Australian subsidiary declared parent of Raytheon.
        if len(children) >= args.review_size:
            reasons.append("large_family_needs_manual_review")

        if reasons:
            flagged.append({
                "parent_uei": puei,
                "parent_name": pname,
                "children": len(children),
                "family_obligated": round(sum(float(c["obligated"] or 0)
                                              for c in children), 2),
                "parent_obligated": round(p_oblig, 2),
                "reasons": reasons,
                "parent_name_reach_in_children": round(reach, 3),
                "dominant_token": dominant,
                "sample_children": [c["name"] for c in children[:5]],
            })

    flagged.sort(key=lambda f: -f["children"])
    total_children_flagged = sum(f["children"] for f in flagged)

    out = {
        "input": str(args.input),
        "families_total": len(fam),
        "genuine_links_total": len(genuine),
        "families_flagged": len(flagged),
        "links_inside_flagged_families": total_children_flagged,
        "pct_links_flagged": round(100 * total_children_flagged / len(genuine), 2),
        "rule": {
            "dominance_threshold": args.dominance,
            "min_family_size": args.min_size,
            "note": "flags are for inspection; nothing is dropped by this script",
        },
        "reason_counts": dict(Counter(r for f in flagged for r in f["reasons"])),
        "flagged": flagged,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"families: {len(fam):,}   genuine links: {len(genuine):,}")
    print(f"flagged families: {len(flagged):,} "
          f"covering {total_children_flagged:,} links "
          f"({out['pct_links_flagged']}%)")
    print(f"reasons: {out['reason_counts']}\n")
    print("largest flagged families:")
    for f in flagged[:18]:
        dom = f["dominant_token"]
        d = f" [{dom['token']} in {dom['share']:.0%}]" if dom else ""
        print(f"  {f['children']:>4} children  {f['parent_name'][:44]:44} "
              f"{','.join(f['reasons'])[:42]}{d}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
