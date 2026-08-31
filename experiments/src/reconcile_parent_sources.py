"""Reconcile the two independent derivations of the corporate-family links.

Source A is the USAspending recipient detail API, one call per entity. Source B is the
FY2025 bulk contract archive, reduced from 6.6M award rows. They come from different
publication pipelines, so agreement is evidence the links are stable and disagreement is
a measurement of how unstable they are. Either way the number is reportable; a single
derivation would not be.

Only entities present in both sources are compared, and the comparison is restricted to
the shuffled (unbiased) API records so the overlap is not itself a biased slice.

Also recomputes the paper's headline name-visibility figure on the archive's much larger
link set, with a Wilson interval, so the claim no longer rests on a few hundred pairs.

Usage:
    python3 experiments/src/reconcile_parent_sources.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "companies", "group", "holdings", "holding",
    "the", "and", "of", "plc", "gmbh", "sa", "nv", "ag", "pty", "pte", "bv",
    "usa", "us", "america", "american", "international", "intl", "worldwide",
    "global", "national", "services", "service", "solutions", "systems",
    "technologies", "technology", "enterprises", "enterprise", "industries",
    "associates", "partners", "trust", "fund", "lc", "pc", "pa",
}
TOKEN = re.compile(r"[a-z0-9]+")


def core(name: str) -> set[str]:
    return {t for t in TOKEN.findall(str(name).lower())
            if t not in STOPWORDS and len(t) > 1}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 2), round(100 * min(1.0, c + h), 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=Path,
                    default=Path("data/raw/usaspending/entity_detail.jsonl"))
    ap.add_argument("--archive", type=Path,
                    default=Path("data/raw/usaspending/archive_parents_fy2025.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/parent_source_reconciliation.json"))
    args = ap.parse_args()

    arch: dict[str, dict] = {}
    with args.archive.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                arch[r["uei"]] = r

    api_all = [json.loads(l) for l in args.api.open(encoding="utf-8") if l.strip()]
    api = {r["uei"]: r for r in api_all
           if r.get("_sample") == "shuffled" and r.get("recipient_level") == "C"
           and r.get("uei")}

    out: dict = {
        "api_records_unbiased": len(api),
        "archive_entities": len(arch),
    }

    overlap = [u for u in api if u in arch]
    out["overlap_entities"] = len(overlap)

    # Compare the parent assignment, treating a self-parent and a null as the same
    # statement: "no distinct parent recorded".
    def parent_of(rec: dict) -> str | None:
        p, u = rec.get("parent_uei"), rec.get("uei")
        return None if (not p or p == u) else p

    def norm(s: str) -> str:
        return " ".join(TOKEN.findall(str(s or "").lower()))

    # Agreement is measured three ways because UEI equality alone overstates
    # disagreement: the same parent frequently appears under a second UEI or a
    # different punctuation of the same name ("KBR  INC." vs "KBR, INC.").
    agree_uei = agree_name = agree_token = 0
    disagree = both_none = one_none = 0
    examples: list[dict] = []
    for u in overlap:
        pa, pb = parent_of(api[u]), parent_of(arch[u])
        if pa is None and pb is None:
            both_none += 1
            continue
        if pa is None or pb is None:
            one_none += 1
            continue
        na, nb = api[u].get("parent_name"), arch[u].get("parent_name")
        if pa == pb:
            agree_uei += 1
            agree_name += 1
            agree_token += 1
            continue
        if norm(na) and norm(na) == norm(nb):
            agree_name += 1
            agree_token += 1
        elif core(na) and (core(na) & core(nb)):
            agree_token += 1
        else:
            disagree += 1
            if len(examples) < 20:
                examples.append({"uei": u, "name": api[u].get("name"),
                                 "api_parent": na, "archive_parent": nb})

    both_claim = sum(1 for u in overlap
                     if parent_of(api[u]) is not None
                     and parent_of(arch[u]) is not None)
    out["comparison"] = {
        "both_report_a_distinct_parent": both_claim,
        "agree_exact_parent_uei": agree_uei,
        "agree_on_normalised_parent_name": agree_name,
        "agree_on_any_shared_distinctive_token": agree_token,
        "substantively_different_parent": disagree,
        "pct_agree_uei": round(100 * agree_uei / both_claim, 2) if both_claim else None,
        "pct_agree_name": round(100 * agree_name / both_claim, 2) if both_claim else None,
        "pct_agree_token": round(100 * agree_token / both_claim, 2) if both_claim else None,
        "wilson95_pct_agree_name": wilson(agree_name, both_claim),
        "both_report_no_parent": both_none,
        "only_one_source_reports_a_parent": one_none,
        "disagreement_examples": examples,
    }

    # Headline name-visibility, recomputed on the archive's full link set.
    genuine = [r for r in arch.values()
               if r.get("parent_uei") and r["parent_uei"] != r["uei"]
               and r.get("name") and r.get("parent_name")]
    identical = [r for r in genuine if r["name"] == r["parent_name"]]
    informative = [r for r in genuine if r["name"] != r["parent_name"]]
    invisible = [r for r in genuine if not (core(r["name"]) & core(r["parent_name"]))]
    inv_info = [r for r in informative
                if not (core(r["name"]) & core(r["parent_name"]))]

    out["name_visibility_archive"] = {
        "genuine_links": len(genuine),
        "name_identical_trivial": len(identical),
        "informative_links": len(informative),
        "name_invisible_of_genuine": len(invisible),
        "pct_of_genuine": round(100 * len(invisible) / len(genuine), 2) if genuine else None,
        "wilson95_pct_of_genuine": wilson(len(invisible), len(genuine)),
        "name_invisible_of_informative": len(inv_info),
        "pct_of_informative": round(100 * len(inv_info) / len(informative), 2) if informative else None,
        "wilson95_pct_of_informative": wilson(len(inv_info), len(informative)),
    }

    # Family size distribution, which determines whether clustering is meaningful.
    fam = Counter(r["parent_uei"] for r in genuine)
    sizes = Counter(fam.values())
    out["family_sizes"] = {
        "distinct_parents": len(fam),
        "size_distribution_top": dict(sorted(sizes.items())[:12]),
        "largest_families": [
            {"parent_uei": p,
             "parent_name": next((r["parent_name"] for r in genuine
                                  if r["parent_uei"] == p), None),
             "children": c}
            for p, c in fam.most_common(10)
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    slim = json.loads(json.dumps(out))
    slim["comparison"].pop("disagreement_examples", None)
    print(json.dumps(slim, indent=2))
    print("\ndisagreement examples:")
    for e in examples[:8]:
        print(f"  {e['name']}\n     api={e['api_parent']}\n     arc={e['archive_parent']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
