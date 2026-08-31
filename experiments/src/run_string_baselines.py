"""String-similarity baselines on CorpFam.

The point of these is not to win. It is to establish, with numbers, that the family
linkage problem splits cleanly in two: pairs whose relationship is visible in the
string, where cheap methods are close to sufficient, and pairs where it is not, where
they collapse to near zero no matter how they are tuned. A single blended F1 hides
exactly that, so every metric here is reported per visibility stratum.

Four scorers, all dependency-free so the benchmark reproduces from a bare Python:

  exact      normalised string equality; the trivial baseline
  jaccard    overlap of distinctive tokens, legal forms removed
  seqratio   difflib character-sequence ratio, a stand-in for edit distance
  tfidf_char character 3-gram cosine with idf weighting, the strongest of the four

Thresholds are chosen on validation and applied unchanged to test. Choosing them on
test would inflate every number and is the single easiest way to publish a result
that does not replicate.

Usage:
    python3 experiments/src/run_string_baselines.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
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


def toks(name: str) -> list[str]:
    return TOKEN.findall(str(name).lower())


def core(name: str) -> set[str]:
    return {t for t in toks(name) if t not in STOPWORDS and len(t) > 2}


def norm(name: str) -> str:
    return " ".join(toks(name))


def ngrams(s: str, n: int = 3) -> Counter:
    s = f"  {norm(s)}  "
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


class TfidfChar:
    """Character n-gram cosine with idf, fitted on the training half only."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def fit(self, names: list[str]) -> None:
        df: Counter = Counter()
        for nm in names:
            df.update(set(ngrams(nm)))
        n = max(len(names), 1)
        self.idf = {g: math.log(n / (1 + c)) + 1.0 for g, c in df.items()}

    def score(self, a: str, b: str) -> float:
        ga, gb = ngrams(a), ngrams(b)
        if not ga or not gb:
            return 0.0
        va = {g: c * self.idf.get(g, 1.0) for g, c in ga.items()}
        vb = {g: c * self.idf.get(g, 1.0) for g, c in gb.items()}
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        common = set(va) & set(vb)
        return sum(va[g] * vb[g] for g in common) / (na * nb)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(100 * f, 2), "tp": tp, "fp": fp, "fn": fn}


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - m), 2), round(100 * min(1.0, c + m), 2)]


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Area under the precision-recall curve, computed globally.

    Reported globally rather than per stratum because precision depends on the pool of
    negatives, and the negative pool is not comparable across strata: negatives almost
    never land in the identical stratum at all, and the grades are strongly associated
    with stratum by construction. Per-stratum recall is the base-rate-invariant
    quantity and is what the tables lead with.
    """
    total_pos = sum(labels)
    if not total_pos:
        return 0.0
    # Ties must be handled as a block. Several scorers are effectively binary, and the
    # pairs file lists positives before negatives, so a stable sort would rank every
    # tied positive above every tied negative and report a perfect score for exact
    # string matching -- an artifact of file order, not a measurement. Accumulating
    # whole tie groups makes the result independent of input ordering.
    groups: dict[float, list[int]] = defaultdict(list)
    for i, s in enumerate(scores):
        groups[s].append(labels[i])
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for s in sorted(groups, reverse=True):
        blk = groups[s]
        tp += sum(blk)
        fp += len(blk) - sum(blk)
        recall = tp / total_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return 100.0 * ap


def evaluate(pairs: list[dict], scores: list[float], thr: float) -> dict:
    tp = fp = fn = 0
    w_tp = w_fn = 0.0
    strat = defaultdict(lambda: [0, 0, 0])
    strat_neg: dict[str, int] = defaultdict(int)
    kinds = defaultdict(lambda: [0, 0])
    for p in pairs:
        if p["label"] == 0:
            strat_neg[p["visibility"]] += 1
    for p, s in zip(pairs, scores):
        pred = s >= thr
        w = max(float(p.get("left_obligated") or 0.0), 0.0)
        if p["label"] == 1:
            if pred:
                tp += 1
                w_tp += w
                strat[p["visibility"]][0] += 1
            else:
                fn += 1
                w_fn += w
                strat[p["visibility"]][2] += 1
        else:
            if pred:
                fp += 1
                strat[p["visibility"]][1] += 1
                kinds[p["negative_kind"]][0] += 1
            kinds[p["negative_kind"]][1] += 1

    out = {"overall": prf(tp, fp, fn), "threshold": round(thr, 4)}
    out["spend_weighted_recall"] = (round(100 * w_tp / (w_tp + w_fn), 2)
                                    if (w_tp + w_fn) else None)
    out["by_visibility"] = {k: prf(v[0], v[1], v[2]) for k, v in sorted(strat.items())}
    # Recall is the headline per-stratum number: it depends only on the positives of
    # that stratum, so unlike F1 it is unaffected by how negatives are distributed
    # across strata. F1 per stratum is retained in the JSON but is not comparable
    # across strata and is not tabulated.
    # Every per-stratum cell carries its own majority-class baseline. Without it an F1
    # is uninterpretable: strata here have positive rates from ~10% to ~97%, so the
    # trivial all-positive classifier scores anywhere from 18 to 99 depending only on
    # which stratum you are looking at. Two separate results in this project were
    # nearly published as findings when they were in fact the base rate -- once at 88
    # and once at 59.9 against a 59.4 floor -- so the comparison is computed here
    # rather than left to the reader or to an arbitrary threshold.
    for k, v in out["by_visibility"].items():
        n_pos = v["tp"] + v["fn"]
        n_all = n_pos + v["fp"] + (strat_neg.get(k, 0) - v["fp"])
        v["positives"] = n_pos
        v["stratum_pairs"] = n_all
        rate = n_pos / n_all if n_all else 0.0
        base_f1 = (2 * rate / (rate + 1)) if rate else 0.0
        v["positive_rate_pct"] = round(100 * rate, 2)
        v["all_positive_f1"] = round(100 * base_f1, 2)
        v["f1_lift_over_majority"] = round(v["f1"] - 100 * base_f1, 2)
        v["beats_majority_baseline"] = bool(v["f1"] > 100 * base_f1 + 0.05)
        v["recall_wilson95"] = wilson(v["tp"], n_pos)
        v["f1_not_comparable_across_strata"] = True
    out["false_positive_rate_by_negative_kind"] = {
        k: {"fp": v[0], "n": v[1], "rate_pct": round(100 * v[0] / v[1], 2) if v[1] else None}
        for k, v in sorted(kinds.items())
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/string_baselines.json"))
    args = ap.parse_args()

    pairs = [json.loads(l) for l in args.pairs.open(encoding="utf-8") if l.strip()]
    by_split = defaultdict(list)
    for p in pairs:
        by_split[p["split"]].append(p)
    print({k: len(v) for k, v in by_split.items()})

    tfidf = TfidfChar()
    tfidf.fit([p["left_name"] for p in by_split["train"]]
              + [p["right_name"] for p in by_split["train"]])

    def jaccard(a: str, b: str) -> float:
        ca, cb = core(a), core(b)
        if not ca or not cb:
            return 0.0
        return len(ca & cb) / len(ca | cb)

    scorers = {
        "exact": lambda a, b: 1.0 if norm(a) == norm(b) else 0.0,
        "jaccard": jaccard,
        "seqratio": lambda a, b: SequenceMatcher(None, norm(a), norm(b)).ratio(),
        "tfidf_char": tfidf.score,
    }

    results = {}
    for name, fn in scorers.items():
        print(f"scoring {name} ...", flush=True)
        sc = {sp: [fn(p["left_name"], p["right_name"]) for p in by_split[sp]]
              for sp in ("val", "test")}

        # Pick the threshold that maximises validation F1.
        cand = sorted({round(s, 3) for s in sc["val"]})
        if len(cand) > 200:
            step = len(cand) // 200
            cand = cand[::step]
        best_thr, best_f1 = 0.5, -1.0
        for t in cand:
            r = evaluate(by_split["val"], sc["val"], t)["overall"]["f1"]
            if r > best_f1:
                best_f1, best_thr = r, t

        res = evaluate(by_split["test"], sc["test"], best_thr)
        res["average_precision"] = round(average_precision(
            [p["label"] for p in by_split["test"]], sc["test"]), 2)
        results[name] = {
            "validation_f1_at_chosen_threshold": best_f1,
            "test": res,
        }
        vis = res["by_visibility"]
        print(f"  thr={best_thr:.3f}  AP={res['average_precision']:.1f}  "
              f"recall: identical={vis.get('identical', {}).get('recall', 0):.1f}  "
              f"visible={vis.get('visible', {}).get('recall', 0):.1f}  "
              f"INVISIBLE={vis.get('invisible', {}).get('recall', 0):.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
