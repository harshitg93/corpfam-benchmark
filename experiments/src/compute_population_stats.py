"""Population statistics computed on the full benchmark, not the biased audit sample.

Two numbers in the paper were derived from the wrong population.

Spend concentration was taken from the per-entity API sample, which was collected by an
interrupted sorted fetch: essentially every UEI in it begins with the letter C. Our own
design notes flag that sample as unusable for population estimates, and it was used for
a population estimate anyway. It is recomputed here over every link in the benchmark
using the obligated amount already carried on each pair.

The second number is new and is the one that actually explains why attribute features
cannot rescue this task: the share of parents that ever appear as an award recipient at
all. A parent that never transacts has no address, no phone and no location, so there
is nothing for an attribute comparison to compare against -- and that is a property of
what a holding company *is*, not a gap in the data source.

Usage:
    python3 experiments/src/compute_population_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--attrs", type=Path,
                    default=Path("data/raw/usaspending/entity_attributes.jsonl"))
    ap.add_argument("--archive-summary", type=Path,
                    default=Path("experiments/results/archive_parents.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/population_stats.json"))
    args = ap.parse_args()

    pos = [json.loads(l) for l in args.pairs.open(encoding="utf-8")]
    pos = [p for p in pos if p["label"] == 1]

    amounts = sorted((abs(float(p.get("left_obligated") or 0.0)) for p in pos),
                     reverse=True)
    total = sum(amounts)
    n = len(amounts)

    def top_share(frac: float) -> float:
        k = max(1, int(round(n * frac)))
        return 100.0 * sum(amounts[:k]) / total if total else 0.0

    # Which parents ever transact in their own right?
    attr_ueis = {json.loads(l)["uei"] for l in args.attrs.open(encoding="utf-8")}
    parents = {p["right_uei"] for p in pos}
    children = {p["left_uei"] for p in pos}
    parents_transacting = parents & attr_ueis

    out = {
        "population": "all positive links in the benchmark",
        "links": n,
        "total_obligated": round(total, 2),
        "links_with_zero_obligated": sum(1 for a in amounts if a == 0.0),
        "spend_concentration": {
            "top_1pct_share_pct": round(top_share(0.01), 2),
            "top_5pct_share_pct": round(top_share(0.05), 2),
            "top_10pct_share_pct": round(top_share(0.10), 2),
        },
        "supersedes": ("ground_truth_audit.json spend_on_genuine_links, which was "
                       "computed on an interrupted sorted API fetch in which nearly "
                       "every UEI begins with 'C' and which is not a population "
                       "sample"),
        "parent_footprint": {
            "distinct_parents": len(parents),
            "distinct_children": len(children),
            "parents_appearing_as_an_award_recipient": len(parents_transacting),
            "parents_with_no_operating_footprint": len(parents) - len(parents_transacting),
            "pct_parents_transacting": round(
                100.0 * len(parents_transacting) / len(parents), 2) if parents else None,
            "pct_parents_no_footprint": round(
                100.0 * (len(parents) - len(parents_transacting)) / len(parents), 2)
            if parents else None,
            "why_it_matters": ("A parent that never receives an award has no address, "
                               "phone or location in this data. Attribute-based "
                               "matching has nothing to compare on the parent side for "
                               "the large majority of families, which is a structural "
                               "property of holding companies rather than a "
                               "shortcoming of the source."),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
