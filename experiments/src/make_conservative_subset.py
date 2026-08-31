"""Cut the conservative post-cutoff release of CorpFam.

Some users will need the strict reading of SAM.gov's D&B bulk-redistribution clause
rather than our own. This emits the subset of the benchmark whose positive pairs are
supported only by awards beginning on or after the cutoff, so that position is
available without anyone having to re-derive it.

Negatives are filtered to entities that survive in the positive set, so the resulting
file is self-consistent rather than referring to entities that no longer appear.

Usage:
    python3 experiments/src/make_conservative_subset.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CUTOFF = "2022-04-04"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--dates", type=Path,
                    default=Path("experiments/results/dnb_link_dates.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/benchmark/corpfam_pairs_conservative.jsonl"))
    args = ap.parse_args()

    dates = json.loads(args.dates.read_text(encoding="utf-8"))
    pairs = [json.loads(l) for l in args.pairs.open(encoding="utf-8") if l.strip()]

    keep_pos = []
    for p in pairs:
        if p["label"] != 1:
            continue
        d = dates.get(f'{p["left_uei"]}|{p["right_uei"]}')
        if d is None or d >= CUTOFF:
            keep_pos.append(p)

    ok_entities = {p["left_uei"] for p in keep_pos} | {p["right_uei"] for p in keep_pos}
    keep_neg = [p for p in pairs if p["label"] == 0
                and p["left_uei"] in ok_entities and p["right_uei"] in ok_entities]

    kept = keep_pos + keep_neg
    with args.out.open("w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "cutoff": CUTOFF,
        "positives": len(keep_pos),
        "negatives": len(keep_neg),
        "total_pairs": len(kept),
        "families": len({p["family_uei"] for p in keep_pos}),
        "visibility_of_positives": dict(Counter(p["visibility"] for p in keep_pos)),
        "split_sizes": dict(Counter(p["split"] for p in kept)),
        "output": str(args.out),
    }
    Path("experiments/results/conservative_subset.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
