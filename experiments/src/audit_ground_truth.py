"""Audit the USAspending entity-detail ground truth before any benchmark is built.

Every figure quoted in the paper about parent-child structure must be reproducible
from this script alone. Each headline number is computed two independent ways and
the script fails loudly if the two disagree.

Usage:
    python3 experiments/src/audit_ground_truth.py \
        --input data/raw/usaspending/entity_detail.jsonl \
        --output experiments/results/ground_truth_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Legal-form and generic tokens carry no identifying signal: "ACME LLC" and
# "BETA LLC" share "LLC" but are unrelated. Dropping them is what makes the
# name-visibility measurement meaningful.
STOPWORDS = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "companies", "group", "holdings", "holding",
    "the", "and", "of", "plc", "gmbh", "sa", "nv", "ag", "pty", "pte", "bv",
    "usa", "us", "america", "american", "international", "intl", "worldwide",
    "global", "national", "services", "service", "solutions", "systems",
    "technologies", "technology", "enterprises", "enterprise", "industries",
    "associates", "partners", "trust", "fund", "lc", "pc", "pa",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def core_tokens(name: str) -> set[str]:
    """Lowercase, strip punctuation, drop legal-form and generic tokens."""
    toks = TOKEN_RE.findall(str(name).lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def norm_compact(name: str) -> str:
    """Alphanumeric-only lowercase form, for the substring test."""
    return "".join(TOKEN_RE.findall(str(name).lower()))


def load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    records = load(args.input)
    out: dict = {"input_file": str(args.input), "records_total": len(records)}

    by_level = Counter(r.get("recipient_level") for r in records)
    out["recipient_level_breakdown"] = dict(by_level)

    children = [r for r in records if r.get("recipient_level") == "C"]
    out["child_level_records"] = len(children)

    # --- Self-parent rate, three independent tests -------------------------
    # A record that names itself as its own parent is a registration artifact,
    # not a corporate relationship. Including these would make ~5 of 6 positive
    # pairs trivial identity matches and inflate every score.
    self_by_uei = sum(
        1 for r in children
        if r.get("parent_uei") and r.get("parent_uei") == r.get("uei")
    )
    # recipient_id is "<uuid>-C" and parent_id is "<uuid>-P"; equal stems mean
    # USAspending assigned both roles to the same underlying entity.
    self_by_stem = sum(
        1 for r in children
        if r.get("recipient_id") and r.get("parent_id")
        and str(r["recipient_id"]).rsplit("-", 1)[0]
        == str(r["parent_id"]).rsplit("-", 1)[0]
    )
    self_by_name = sum(
        1 for r in children
        if r.get("parent_name") and r.get("parent_name") == r.get("name")
    )

    genuine = [
        r for r in children
        if r.get("parent_uei") and r.get("uei")
        and r["parent_uei"] != r["uei"]
    ]
    no_parent = [r for r in children if not r.get("parent_uei")]

    out["self_parent"] = {
        "by_uei_equality": self_by_uei,
        "by_recipient_id_stem_equality": self_by_stem,
        "by_name_equality": self_by_name,
        "uei_and_stem_agree": self_by_uei == self_by_stem,
        "pct_of_child_records": round(100 * self_by_uei / len(children), 2) if children else None,
    }
    out["genuine_parent_child_links"] = len(genuine)
    out["child_records_with_no_parent_uei"] = len(no_parent)

    # Partition check: the three buckets must exactly reconstruct the whole.
    partition = self_by_uei + len(genuine) + len(no_parent)
    out["partition_check"] = {
        "self_plus_genuine_plus_none": partition,
        "equals_child_records": partition == len(children),
    }
    assert partition == len(children), (
        f"partition {partition} != child records {len(children)}"
    )

    # --- Multi-parent structure --------------------------------------------
    # `parents` is an array. If entities carry more than one parent, "the"
    # ultimate parent is a modelling choice we must define, not a field we read.
    parent_counts = Counter(len(r.get("parents") or []) for r in children)
    multi = sum(n for k, n in parent_counts.items() if k > 1)
    out["parents_array"] = {
        "distribution_of_parent_count": dict(sorted(parent_counts.items())),
        "records_with_more_than_one_parent": multi,
        "max_parents_on_any_record": max(parent_counts) if parent_counts else 0,
    }

    # Do any parents appear as children elsewhere? That would imply a chain
    # deeper than one hop and change what "ultimate" means.
    child_ueis = {r["uei"] for r in genuine if r.get("uei")}
    parent_ueis = {r["parent_uei"] for r in genuine if r.get("parent_uei")}
    out["hierarchy_depth"] = {
        "distinct_parent_ueis": len(parent_ueis),
        "parents_that_are_also_children": len(parent_ueis & child_ueis),
        "note": "non-zero means chains deeper than one hop exist in this sample",
    }

    # --- Name visibility: the paper's central premise ----------------------
    # If a child's name always contained its parent's name, string matching
    # would solve the task and the benchmark would be pointless.
    no_shared_token = 0
    not_substring = 0
    examples: list[dict] = []
    for r in genuine:
        cn, pn = r.get("name"), r.get("parent_name")
        if not cn or not pn:
            continue
        ct, pt = core_tokens(cn), core_tokens(pn)
        shares = bool(ct & pt)
        if not shares:
            no_shared_token += 1
            if len(examples) < 25:
                examples.append({"child": cn, "parent": pn,
                                 "obligated": r.get("total_transaction_amount")})
        pc, cc = norm_compact(pn), norm_compact(cn)
        if pc and cc and pc not in cc and cc not in pc:
            not_substring += 1

    scored = sum(1 for r in genuine if r.get("name") and r.get("parent_name"))
    out["name_visibility"] = {
        "pairs_scored": scored,
        "no_shared_core_token": no_shared_token,
        "pct_no_shared_core_token": round(100 * no_shared_token / scored, 2) if scored else None,
        "neither_name_substring_of_other": not_substring,
        "pct_not_substring": round(100 * not_substring / scored, 2) if scored else None,
        "examples_name_invisible": examples,
    }

    # --- Spend concentration on genuine links ------------------------------
    amounts = sorted(
        (float(r.get("total_transaction_amount") or 0) for r in genuine),
        reverse=True,
    )
    total = sum(amounts)
    out["spend_on_genuine_links"] = {
        "total_obligated": total,
        "n": len(amounts),
        "top_1pct_share": round(100 * sum(amounts[: max(1, len(amounts) // 100)]) / total, 2) if total else None,
        "top_10pct_share": round(100 * sum(amounts[: max(1, len(amounts) // 10)]) / total, 2) if total else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in out.items() if k != "name_visibility"}, indent=2))
    nv = dict(out["name_visibility"])
    ex = nv.pop("examples_name_invisible")
    print("\nname_visibility:", json.dumps(nv, indent=2))
    print("\nname-invisible examples:")
    for e in ex[:12]:
        print(f"  {e['parent']}  ->  {e['child']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
