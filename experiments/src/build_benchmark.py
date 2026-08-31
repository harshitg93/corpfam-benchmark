"""Build the CorpFam benchmark from the FY2025 archive links.

Produces a pairwise family-linkage task with leak-free splits, three grades of negative,
and a name-visibility stratum on every pair.

Design decisions, and the mistakes that produced them:

*Splits are assigned to connected components, not families.* Assigning whole families
is not sufficient. Thirty-six entities appear as a child in one family and the parent of
another, and two hundred normalised name strings occur in more than one family, so a
family-level assertion still let eight entities straddle train and test. Families are
therefore unioned whenever they share an entity or a normalised name, and the resulting
component is assigned as a unit. The build asserts entity-level and name-level
disjointness and fails loudly rather than reconciling silently.

*Pairs are deduplicated globally.* A negative could previously be emitted under more
than one grade, which put 2,369 redundant rows in the dataset and 631 in the test split,
silently reweighting the evaluation.

*Negatives match the role structure of positives.* Each negative pairs a child with a
different family's parent. An earlier build drew child-child negatives; because parents
rarely transact and so have no address, "both sides have an address" separated the
classes almost perfectly and a supervised model reached 88 F1 on the invisible stratum
by learning entity role rather than ownership.

*Exclusions are keyed on UEI and fully logged, including retentions.* An exclusion list
keyed on raw name strings is indefensible in a paper about name instability. More
importantly, recording only rejections makes "inspected 28, rejected 8" indistinguishable
from "removed 8 inconvenient cases", and the manual exclusions fall overwhelmingly on the
invisible stratum -- the very stratum the paper is about. The full inspection log ships,
and --no-exclusions reproduces every headline number on the unfiltered data.

Usage:
    python3 experiments/src/build_benchmark.py
    python3 experiments/src/build_benchmark.py --no-exclusions --outdir data/benchmark_unfiltered
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    "associates", "partners", "trust", "fund", "lc", "pc", "pa",
}
TOKEN = re.compile(r"[a-z0-9]+")

# Every large family was inspected against its child list. Both verdicts are recorded:
# publishing only the rejections would make a biased purge indistinguishable from a
# careful audit, and these exclusions fall overwhelmingly on the invisible stratum.
INSPECTION_LOG = {
    "ROCKWELL COLLINS AUSTRALIA PTY LIMITED": ("exclude",
        "declared parent of RAYTHEON COMPANY and HAMILTON SUNDSTRAND; an Australian "
        "Pty Ltd is not the ultimate parent of RTX entities"),
    "ZIPPY SHELL INCORPORATED": ("exclude",
        "a moving company declared parent of 103 Waste Management entities"),
    "GOVERNMENT OF THE UNITED STATES": ("exclude",
        "sovereign catch-all, not a corporate family"),
    "WICO LIMITED": ("exclude",
        "declared parent of General Dynamics operating companies"),
    "PAE-PARSONS GLOBAL LOGISTICS SERVICES, LLC": ("exclude",
        "a joint venture declared parent of PAE and Amentum entities"),
    "HONEYWELL SAFETY PRODUCTS USA, INC.": ("exclude",
        "inverted: declared parent of HONEYWELL INTERNATIONAL INC., its own parent"),
    "BAY DISPOSAL, LLC": ("exclude",
        "declared parent of Waste Connections entities"),
    "PUROLATOR FACET, INC.": ("exclude",
        "declared parent of PARKER-HANNIFIN CORPORATION and Meggitt entities"),
    # Retained after inspection. Several of these were flagged by an automatic
    # name-dissimilarity rule and are genuine conglomerates; keeping the record shows
    # the rule's false-positive rate rather than hiding it.
    "ARCTIC SLOPE REGIONAL CORPORATION": ("retain",
        "flagged automatically; ASRC Builders, Analytical Services and Broadleaf are "
        "genuine ASRC subsidiaries"),
    "HEICO CORP": ("retain", "genuine acquisitive conglomerate, wholly name-invisible"),
    "TRANSDIGM GROUP INCORPORATED": ("retain", "genuine; Breeze-Eastern, Tactair"),
    "BERKSHIRE HATHAWAY INC.": ("retain", "genuine; MidAmerican, TTI, Kitco"),
    "REPUBLIC SERVICES INC": ("retain", "genuine; BFI, Allied Waste, US Ecology"),
    "AMETEK INC": ("retain", "genuine; Vision Research, NSI-MI, Navitar"),
    "LEONARDO SPA": ("retain", "genuine; DRS entities are Leonardo DRS"),
    "L3HARRIS TECHNOLOGIES, INC": ("retain", "genuine; L3 Technologies, Aerojet"),
    "HANGER, INC.": ("retain", "genuine; Hanger Prosthetics & Orthotics"),
    "NORTHROP GRUMMAN CORPORATION": ("retain", "genuine"),
    "THE BOEING COMPANY": ("retain", "genuine"),
    "LOCKHEED MARTIN CORP": ("retain", "genuine"),
    "TETRA TECH, INC.": ("retain", "genuine; Management Systems International"),
    "HDR, INC": ("retain", "genuine"),
    "AECOM": ("retain", "genuine"),
    "LEIDOS HOLDINGS, INC.": ("retain", "genuine"),
    "JACOBS ENGINEERING GROUP INC.": ("retain", "genuine; CH2M Hill"),
    "GENERAL DYNAMICS CORP": ("retain", "genuine; CSRA"),
    "TELEDYNE TECHNOLOGIES INCORPORATED": ("retain", "genuine"),
    "UNIFIRST CORPORATION": ("retain", "genuine"),
}
EXCLUDED_PARENT_NAMES = {k for k, (v, _) in INSPECTION_LOG.items() if v == "exclude"}
# Retained for backward compatibility with importers of the previous module.
EXCLUDED_PARENTS = {k: r for k, (v, r) in INSPECTION_LOG.items() if v == "exclude"}


def core(name: str) -> set[str]:
    return {t for t in TOKEN.findall(str(name).lower())
            if t not in STOPWORDS and len(t) > 2}


def norm(name: str) -> str:
    return " ".join(TOKEN.findall(str(name).lower()))


class DSU:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path,
                    default=Path("data/raw/usaspending/archive_parents_fy2025.jsonl"))
    ap.add_argument("--artifacts", type=Path,
                    default=Path("experiments/results/family_artifacts.json"))
    ap.add_argument("--outdir", type=Path, default=Path("data/benchmark"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("experiments/results/benchmark_build.json"))
    ap.add_argument("--no-exclusions", action="store_true",
                    help="keep every declared family; for the sensitivity analysis")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in args.links.open(encoding="utf-8") if l.strip()]

    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    sovereign_names = sorted({f["parent_name"] for f in artifacts["flagged"]
                              if "sovereign_or_government_entity" in f["reasons"]})

    links = [r for r in recs
             if r.get("parent_uei") and r["parent_uei"] != r["uei"]
             and r.get("name") and r.get("parent_name")]

    # Resolve the name-keyed decisions to UEIs once, and key on UEI thereafter.
    name_to_ueis: dict[str, set] = defaultdict(set)
    for r in links:
        name_to_ueis[r["parent_name"]].add(r["parent_uei"])
    excluded_ueis: set[str] = set()
    for nm in EXCLUDED_PARENT_NAMES:
        excluded_ueis |= name_to_ueis.get(nm, set())
    sovereign_ueis: set[str] = set()
    for nm in sovereign_names:
        sovereign_ueis |= name_to_ueis.get(nm, set())

    def visibility(child: str, parent: str) -> str:
        if norm(child) == norm(parent):
            return "identical"
        return "visible" if (core(child) & core(parent)) else "invisible"

    dropped: Counter = Counter()
    dropped_by_stratum: Counter = Counter()
    kept = []
    for r in links:
        v = visibility(r["name"], r["parent_name"])
        if not args.no_exclusions and r["parent_uei"] in excluded_ueis:
            dropped["manually_adjudicated_artifact"] += 1
            dropped_by_stratum[f"manual_{v}"] += 1
        elif not args.no_exclusions and r["parent_uei"] in sovereign_ueis:
            dropped["sovereign_entity"] += 1
            dropped_by_stratum[f"sovereign_{v}"] += 1
        else:
            kept.append(r)

    fam: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        fam[r["parent_uei"]].append(r)

    # --- Leak-free splits -------------------------------------------------------
    # Union families that share any entity or any normalised name, then assign whole
    # components. A family-level assignment is not enough: entities that are a child in
    # one family and the parent of another stitch families together.
    d = DSU()
    for puei, children in fam.items():
        d.find(f"F:{puei}")
        d.union(f"F:{puei}", f"E:{puei}")
        d.union(f"F:{puei}", f"N:{norm(children[0]['parent_name'])}")
        for c in children:
            d.union(f"F:{puei}", f"E:{c['uei']}")
            d.union(f"F:{puei}", f"N:{norm(c['name'])}")

    comp_of_fam = {p: d.find(f"F:{p}") for p in fam}
    comps = sorted(set(comp_of_fam.values()))
    rng.shuffle(comps)

    # Assign components largest-first to whichever split has the largest deficit
    # *relative to its own target*. Using the absolute deficit sends every early large
    # component to train, because train's target is three times the others': that put
    # all 902 multi-child families in train and left the clustering task with no
    # evaluation data at all. Normalising by target keeps large components spread
    # across splits while still landing near 60/20/20 by link count.
    comp_families: dict[str, list[str]] = defaultdict(list)
    for p, c in comp_of_fam.items():
        comp_families[c].append(p)
    comp_size = {c: sum(len(fam[p]) for p in ps) for c, ps in comp_families.items()}
    total_links = sum(comp_size.values())
    targets = {"train": 0.6 * total_links, "val": 0.2 * total_links,
               "test": 0.2 * total_links}
    filled = {"train": 0.0, "val": 0.0, "test": 0.0}
    split_of_comp: dict[str, str] = {}
    for c in sorted(comps, key=lambda c: (-comp_size[c], c)):
        sp = max(targets, key=lambda s: (targets[s] - filled[s]) / targets[s])
        split_of_comp[c] = sp
        filled[sp] += comp_size[c]
    split_of = {p: split_of_comp[comp_of_fam[p]] for p in fam}

    pairs = []
    for puei, children in fam.items():
        for c in children:
            pairs.append({
                "left_uei": c["uei"], "left_name": c["name"],
                "right_uei": puei, "right_name": c["parent_name"],
                "label": 1, "negative_kind": None,
                "visibility": visibility(c["name"], c["parent_name"]),
                "family_uei": puei, "split": split_of[puei],
                "left_obligated": c.get("obligated") or 0.0,
            })

    # --- Negatives --------------------------------------------------------------
    by_child: dict[str, list[dict]] = defaultdict(list)
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for puei, children in fam.items():
        sp = split_of[puei]
        by_parent[sp].append({"uei": puei, "name": children[0]["parent_name"],
                              "family": puei})
        for c in children:
            by_child[sp].append({"uei": c["uei"], "name": c["name"], "family": puei,
                                 "obligated": c.get("obligated") or 0.0})

    def block_key(name: str) -> str:
        t = sorted(core(name))
        return t[0] if t else ""

    emitted: set[tuple] = {(p["left_uei"], p["right_uei"]) for p in pairs}
    neg_counts: Counter = Counter()

    for split, children in by_child.items():
        parents = by_parent[split]
        if len(children) < 4 or len(parents) < 4:
            continue
        p_blocks: dict[str, list[dict]] = defaultdict(list)
        p_tokens: dict[str, list[dict]] = defaultdict(list)
        for e in parents:
            p_blocks[block_key(e["name"])].append(e)
            for t in core(e["name"]):
                p_tokens[t].append(e)

        n_pos = sum(1 for p in pairs if p["split"] == split and p["label"] == 1)
        per_kind = (n_pos * args.neg_per_pos) // 3

        def emit(child: dict, parent: dict, kind: str) -> bool:
            if child is None or parent is None:
                return False
            if child["family"] == parent["family"] or child["uei"] == parent["uei"]:
                return False
            key = (child["uei"], parent["uei"])
            if key in emitted:  # global dedupe, across grades and against positives
                return False
            emitted.add(key)
            pairs.append({
                "left_uei": child["uei"], "left_name": child["name"],
                "right_uei": parent["uei"], "right_name": parent["name"],
                "label": 0, "negative_kind": kind,
                "visibility": visibility(child["name"], parent["name"]),
                "family_uei": None, "split": split,
                "left_obligated": child["obligated"],
            })
            neg_counts[kind] += 1
            return True

        made = guard = 0
        while made < per_kind and guard < per_kind * 40:
            guard += 1
            made += emit(rng.choice(children), rng.choice(parents), "random")

        made = guard = 0
        while made < per_kind and guard < per_kind * 60:
            guard += 1
            c = rng.choice(children)
            toks = [t for t in core(c["name"]) if t in p_tokens]
            if toks:
                made += emit(c, rng.choice(p_tokens[rng.choice(toks)]), "hard_string")

        made = guard = 0
        while made < per_kind and guard < per_kind * 60:
            guard += 1
            c = rng.choice(children)
            grp = p_blocks.get(block_key(c["name"]))
            if grp:
                made += emit(c, rng.choice(grp), "same_block")

    # --- Assertions -------------------------------------------------------------
    ent_splits: dict[str, set] = defaultdict(set)
    name_splits: dict[str, set] = defaultdict(set)
    for p in pairs:
        if p["label"] != 1:
            continue
        for u, nm in ((p["left_uei"], p["left_name"]),
                      (p["right_uei"], p["right_name"])):
            ent_splits[u].add(p["split"])
            name_splits[norm(nm)].add(p["split"])
    ent_leak = {u: s for u, s in ent_splits.items() if len(s) > 1}
    name_leak = {n: s for n, s in name_splits.items() if len(s) > 1}
    assert not ent_leak, f"entity spans splits: {list(ent_leak)[:5]}"
    assert not name_leak, f"normalised name spans splits: {list(name_leak)[:5]}"
    seen_pairs = Counter((p["left_uei"], p["right_uei"]) for p in pairs)
    assert max(seen_pairs.values()) == 1, "duplicate pair emitted"

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_file = args.outdir / "corpfam_pairs.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    pos = [p for p in pairs if p["label"] == 1]
    # Per-stratum base rates, without which per-stratum F1 cannot be interpreted.
    strat: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        if p["split"] == "test":
            strat[p["visibility"]]["n"] += 1
            strat[p["visibility"]]["pos"] += p["label"]
    base_rates = {}
    for v, c in strat.items():
        r = c["pos"] / c["n"] if c["n"] else 0.0
        base_rates[v] = {
            "test_pairs": c["n"], "test_positives": c["pos"],
            "positive_rate_pct": round(100 * r, 2),
            "all_positive_f1": round(100 * (2 * r / (r + 1)) if r else 0.0, 2),
            "all_positive_recall": 100.0,
        }

    manifest = {
        "seed": args.seed,
        "exclusions_applied": not args.no_exclusions,
        "source_links": len(links),
        "dropped": dict(dropped),
        "dropped_by_stratum": dict(dropped_by_stratum),
        "links_kept": len(kept),
        "families": len(fam),
        "components_after_union": len(comps),
        "pairs_total": len(pairs),
        "positives": len(pos),
        "negatives": len(pairs) - len(pos),
        "negative_kinds": dict(neg_counts),
        "split_sizes": dict(Counter(p["split"] for p in pairs)),
        "positive_split_sizes": dict(Counter(p["split"] for p in pos)),
        "visibility_of_positives": dict(Counter(p["visibility"] for p in pos)),
        "test_base_rates": base_rates,
        "negative_kind_by_stratum": {
            f"{k}|{v}": n for (k, v), n in sorted(Counter(
                (p["negative_kind"], p["visibility"])
                for p in pairs if p["label"] == 0).items())},
        "leak_checks": {
            "family_level": "passed",
            "entity_level": "passed - no entity appears in two splits",
            "normalised_name_level": "passed - no name string appears in two splits",
            "duplicate_pairs": "passed - every (child,parent) pair appears once",
        },
        "inspection_log": {k: {"verdict": v, "reason": r}
                           for k, (v, r) in sorted(INSPECTION_LOG.items())},
        "excluded_parent_ueis": sorted(excluded_ueis),
        "sovereign_parents_excluded_all": sovereign_names,
        "sovereign_parent_ueis": sorted(sovereign_ueis),
        "output": str(out_file),
        "sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    show = {k: v for k, v in manifest.items()
            if k not in ("inspection_log", "sovereign_parents_excluded_all",
                         "sovereign_parent_ueis", "excluded_parent_ueis", "sha256",
                         "negative_kind_by_stratum")}
    print(json.dumps(show, indent=2))
    print(f"\nwrote {out_file}\nwrote {args.manifest}")


if __name__ == "__main__":
    main()
