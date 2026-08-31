"""Family-level clustering task for CorpFam.

Pairwise linkage is not what a procurement or credit system actually needs. The
deliverable is a *partition*: every supplier record assigned to a corporate family, so
that spend can be rolled up. Pairwise F1 does not measure that, because pairwise errors
compound through transitivity -- one false link merges two families into a single
cluster and corrupts both rollups.

This task therefore evaluates the end-to-end job: given a set of entities, recover the
families. It is restricted to families with at least two children, since a partition
task over singleton families is trivially solved by predicting no links at all.

Two conditions are reported, which is the point of the design:

  realistic    candidates come from blocking, as in a deployed pipeline
  oracle       every within-set pair is scored, blocking removed

The gap between them isolates how much of the failure is the blocker versus the
matcher. Given that blocking recovers almost no name-invisible link, the expectation is
that the realistic condition is bounded well below the oracle -- and that even the
oracle is bounded well below usable.

Metrics are the standard entity-resolution clustering set: pairwise P/R/F1, B-cubed
P/R/F1 (which credits partial cluster recovery rather than scoring all-or-nothing), and
exact cluster recovery.

Usage:
    python3 experiments/src/run_clustering_task.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from run_string_baselines import TfidfChar, core, norm

MAX_BLOCK = 100


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


def bcubed(truth: dict[str, str], pred: dict[str, str]) -> tuple[float, float, float]:
    """B-cubed precision/recall/F1 over the shared entity set."""
    t_members: dict[str, set] = defaultdict(set)
    p_members: dict[str, set] = defaultdict(set)
    for e, c in truth.items():
        t_members[c].add(e)
    for e, c in pred.items():
        p_members[c].add(e)
    P = R = 0.0
    ents = [e for e in truth if e in pred]
    for e in ents:
        tc, pc = t_members[truth[e]], p_members[pred[e]]
        inter = len(tc & pc)
        P += inter / len(pc)
        R += inter / len(tc)
    n = len(ents) or 1
    P, R = P / n, R / n
    f = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, f


def pairwise(truth: dict[str, str], pred: dict[str, str]) -> tuple[float, float, float]:
    def pairs(m: dict[str, str]) -> set[frozenset]:
        g: dict[str, list[str]] = defaultdict(list)
        for e, c in m.items():
            g[c].append(e)
        out = set()
        for members in g.values():
            for a, b in combinations(sorted(members), 2):
                out.add(frozenset((a, b)))
        return out

    T, P = pairs(truth), pairs(pred)
    tp = len(T & P)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(T) if T else 0.0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/clustering_task.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.pairs.open(encoding="utf-8") if l.strip()]
    pos = [p for p in rows if p["label"] == 1]

    fam: dict[str, set] = defaultdict(set)
    names: dict[str, str] = {}
    split_of: dict[str, str] = {}
    for p in pos:
        fam[p["right_uei"]].add(p["left_uei"])
        names[p["left_uei"]] = p["left_name"]
        names[p["right_uei"]] = p["right_name"]
        split_of[p["right_uei"]] = p["split"]

    multi = {f: ch for f, ch in fam.items() if len(ch) >= 2}
    print(f"{len(fam):,} families total; {len(multi):,} have 2+ children "
          f"and form the clustering task", flush=True)

    # Train the idf on training families only, then evaluate on test families.
    train_names = [names[u] for f, ch in multi.items() if split_of[f] == "train"
                   for u in list(ch) + [f]]
    tfidf = TfidfChar()
    tfidf.fit(train_names or [names[u] for u in names])

    def build_split(split: str) -> tuple:
        fams = {f: ch for f, ch in multi.items() if split_of[f] == split}
        ents = sorted({u for f, ch in fams.items() for u in list(ch) + [f]})
        truth = {u: f for f, ch in fams.items() for u in list(ch) + [f]}

        oracle_pairs = [frozenset((a, b)) for a, b in combinations(ents, 2)]
        idx: dict[str, list[str]] = defaultdict(list)
        for u in ents:
            for t in core(names[u]):
                idx[t].append(u)
        blocked = set()
        for members in idx.values():
            if 1 < len(members) <= MAX_BLOCK:
                for a, b in combinations(sorted(set(members)), 2):
                    blocked.add(frozenset((a, b)))
        return fams, ents, truth, {"realistic_blocked": sorted(blocked, key=sorted),
                                   "oracle_all_pairs": oracle_pairs}

    def evaluate(fams, ents, truth, scored, thr) -> dict:
        d = DSU()
        for u in ents:
            d.find(u)
        for c, s in scored:
            if s >= thr:
                a, b = sorted(c)
                d.union(a, b)
        pred = {u: d.find(u) for u in ents}
        pp, pr, pf = pairwise(truth, pred)
        bp, br, bf = bcubed(truth, pred)

        # A family counts as exactly recovered when its predicted cluster is exactly
        # its true membership -- no member missing, nothing else merged in.
        cluster_members: dict[str, set] = defaultdict(set)
        for u in ents:
            cluster_members[pred[u]].add(u)
        exact = sum(1 for f, ch in fams.items()
                    if cluster_members[pred[f]] == set(ch) | {f})

        return {"threshold": round(thr, 3),
                "pairwise_precision": round(100 * pp, 2),
                "pairwise_recall": round(100 * pr, 2),
                "pairwise_f1": round(100 * pf, 2),
                "bcubed_precision": round(100 * bp, 2),
                "bcubed_recall": round(100 * br, 2),
                "bcubed_f1": round(100 * bf, 2),
                "exact_families_recovered": exact,
                "families": len(fams),
                "exact_recovery_pct": round(100 * exact / len(fams), 2),
                "clusters_predicted": len(cluster_members)}

    val = build_split("val")
    test = build_split("test")
    results: dict = {}

    for cond in ("realistic_blocked", "oracle_all_pairs"):
        # Threshold is chosen on validation and applied unchanged to test. Choosing it
        # per split would tune on test, which is the very thing this paper objects to.
        v_scored = [(c, tfidf.score(*[names[u] for u in sorted(c)]))
                    for c in val[3][cond]]
        best_thr, best_bf = 0.5, -1.0
        for i in range(4, 20):
            thr = i / 20
            bf = evaluate(val[0], val[1], val[2], v_scored, thr)["bcubed_f1"]
            if bf > best_bf:
                best_bf, best_thr = bf, thr

        t_scored = [(c, tfidf.score(*[names[u] for u in sorted(c)]))
                    for c in test[3][cond]]
        r = evaluate(test[0], test[1], test[2], t_scored, best_thr)
        results[cond] = {"entities": len(test[1]), "families": len(test[0]),
                         "candidate_pairs": len(test[3][cond]),
                         "threshold_selected_on_validation": best_thr,
                         "validation_bcubed_f1": best_bf, **r}
        print(f"  {cond:18} thr={best_thr:.2f}  B3-F1={r['bcubed_f1']:5.1f}  "
              f"pairF1={r['pairwise_f1']:5.1f}  "
              f"exact={r['exact_recovery_pct']:5.1f}%  "
              f"cands={len(test[3][cond]):,}", flush=True)

    out = {
        "task": ("Partition entities into corporate families. Restricted to families "
                 "with 2+ children; singleton families make the partition trivial."),
        "families_total": len(fam),
        "families_multi_child": len(multi),
        "conditions": {
            "realistic_blocked": "candidates from token blocking, as deployed",
            "oracle_all_pairs": "every within-split pair scored, blocking removed",
        },
        "protocol": ("Threshold chosen on validation by B-cubed F1 and applied "
                     "unchanged to test. Scores are TF-IDF character 3-gram cosine; "
                     "clusters are connected components over the thresholded graph."),
        "test_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
