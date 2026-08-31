"""Blocking analysis for CorpFam: is the candidate pair ever generated?

Deployed entity resolution never scores all pairs. A blocking step proposes candidates
and anything it fails to propose is unrecoverable regardless of the matcher. This
measures how much of each difficulty stratum survives that step.

Two things must be said honestly up front, because a blocking specialist will notice
both immediately.

First, an identity. The invisible stratum is *defined* as an empty intersection of
``core()`` tokens, and token blocking and first-token blocking key on exactly that
function. Their zero recall on the stratum is therefore true by construction and
requires no experiment. Reporting it as a discovery would be circular.

The empirical content is what happens once the key is relaxed away from that identity.
Character q-grams, sorted-neighbourhood adjacency, phonetic keys, and semantic
nearest-neighbours in an embedding space are all *not* functions of ``core()``, and a
learned or attribute-keyed blocker is not a function of the name at all. If those also
fail, the architectural claim is empirical rather than definitional. That is the
question this script exists to answer.

Second, the population. An earlier version blocked over only the entities appearing in
our own pairs file, which flatters reduction ratio and pairs quality because the roster
has already been filtered to entities of interest. Blocking is measured here over the
full recipient roster, which is the population a deployed system indexes.

Metrics are the standard ones: pair completeness (recall of true links into the
candidate set), reduction ratio, and pairs quality.

Usage:
    PYTHONPATH=experiments/src python3 experiments/src/run_blocking_analysis.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
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
MAX_BLOCK = 100          # purge oversized blocks; applied to every keyed scheme
ANN_K = 20               # neighbours per entity for the embedding blocker


def core(name: str) -> set[str]:
    return {t for t in TOKEN.findall(str(name).lower())
            if t not in STOPWORDS and len(t) > 2}


def norm(name: str) -> str:
    return " ".join(TOKEN.findall(str(name).lower()))


def qgrams(name: str, q: int = 4) -> set[str]:
    s = norm(name).replace(" ", "")
    return {s[i:i + q] for i in range(len(s) - q + 1)} if len(s) >= q else {s}


def soundex(word: str) -> str:
    """Classic Soundex. A phonetic key is not a function of core() token identity, so
    it can in principle bridge spelling divergence that token blocking cannot."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4",
             **dict.fromkeys("mn", "5"), "r": "6"}
    out = word[0].upper()
    prev = codes.get(word[0], "")
    for ch in word[1:]:
        c = codes.get(ch, "")
        if c and c != prev:
            out += c
        if ch not in "hw":
            prev = c
        if len(out) == 4:
            break
    return (out + "000")[:4]


def from_index(index: dict[str, list[str]], cap: int = MAX_BLOCK) -> set[frozenset]:
    cands: set[frozenset] = set()
    for members in index.values():
        uniq = sorted(set(members))
        if len(uniq) < 2 or len(uniq) > cap:
            continue
        for a, b in combinations(uniq, 2):
            cands.add(frozenset((a, b)))
    return cands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--roster", type=Path,
                    default=Path("data/raw/usaspending/archive_parents_fy2025.jsonl"))
    ap.add_argument("--attrs", type=Path,
                    default=Path("data/raw/usaspending/entity_attributes.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/blocking_analysis.json"))
    ap.add_argument("--skip-embedding", action="store_true")
    args = ap.parse_args()

    # True links to be recovered.
    truth: dict[frozenset, str] = {}
    bench_names: dict[str, str] = {}
    for line in args.pairs.open(encoding="utf-8"):
        p = json.loads(line)
        bench_names[p["left_uei"]] = p["left_name"]
        bench_names[p["right_uei"]] = p["right_name"]
        if p["label"] == 1 and p["left_uei"] != p["right_uei"]:
            truth[frozenset((p["left_uei"], p["right_uei"]))] = p["visibility"]

    # Full deployment roster.
    names: dict[str, str] = {}
    for line in args.roster.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("uei") and r.get("name"):
            names[r["uei"]] = r["name"]
        if r.get("parent_uei") and r.get("parent_name"):
            names.setdefault(r["parent_uei"], r["parent_name"])
    for u, nm in bench_names.items():
        names.setdefault(u, nm)

    ueis = sorted(names)
    n = len(ueis)
    all_pairs = n * (n - 1) // 2
    by_vis = Counter(truth.values())
    print(f"roster: {n:,} entities -> {all_pairs:,} pairs in the naive space")
    print(f"true links: {len(truth):,} {dict(by_vis)}\n", flush=True)

    schemes: dict[str, set] = {}

    def timed(label: str, fn) -> None:
        t0 = time.time()
        schemes[label] = (fn(), time.time() - t0)
        print(f"  built {label}: {len(schemes[label][0]):,} candidates "
              f"in {schemes[label][1]:.1f}s", flush=True)

    def token_blocking() -> set[frozenset]:
        idx = defaultdict(list)
        for u in ueis:
            for t in core(names[u]):
                idx[t].append(u)
        return from_index(idx)

    def qgram_blocking() -> set[frozenset]:
        idx = defaultdict(list)
        for u in ueis:
            for g in qgrams(names[u]):
                idx[g].append(u)
        return from_index(idx)

    def first_token_blocking() -> set[frozenset]:
        idx = defaultdict(list)
        for u in ueis:
            t = sorted(core(names[u]))
            if t:
                idx[t[0]].append(u)
        return from_index(idx)

    def sorted_neighbourhood(window: int = 20) -> set[frozenset]:
        order = sorted(ueis, key=lambda u: norm(names[u]))
        out = set()
        for i, a in enumerate(order):
            for b in order[i + 1:i + window]:
                out.add(frozenset((a, b)))
        return out

    def phonetic_blocking() -> set[frozenset]:
        idx = defaultdict(list)
        for u in ueis:
            toks = sorted(core(names[u]))
            if toks:
                idx["-".join(soundex(t) for t in toks[:2])].append(u)
        return from_index(idx)

    def attribute_blocking() -> set[frozenset]:
        """Keyed on ZIP and on state+city -- not a function of the name at all."""
        attrs = {}
        if args.attrs.exists():
            for line in args.attrs.open(encoding="utf-8"):
                r = json.loads(line)
                attrs[r["uei"]] = r
        idx = defaultdict(list)
        for u in ueis:
            a = attrs.get(u)
            if not a:
                continue
            z = re.sub(r"\D", "", str(a.get("zip") or ""))[:5]
            if z:
                idx[f"zip:{z}"].append(u)
            if a.get("state") and a.get("city"):
                idx[f"sc:{a['state']}|{norm(a['city'])}"].append(u)
        return from_index(idx)

    timed("Token blocking", token_blocking)
    timed("Q-gram blocking (q=4)", qgram_blocking)
    timed("First-token blocking", first_token_blocking)
    timed("Sorted neighbourhood (w=20)", sorted_neighbourhood)
    timed("Phonetic (Soundex)", phonetic_blocking)
    timed("Attribute (ZIP / city-state)", attribute_blocking)

    if not args.skip_embedding:
        def embedding_ann() -> set[frozenset]:
            """Semantic nearest neighbours. This is the key test: an embedding
            neighbourhood is not a function of core() token identity, so if it also
            fails on the invisible stratum the barrier is not the choice of key."""
            import numpy as np
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            uniq = [names[u] for u in ueis]
            emb = model.encode(uniq, batch_size=256, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True)
            emb = emb.astype("float32")
            out = set()
            step = 512
            for i in range(0, len(ueis), step):
                sims = emb[i:i + step] @ emb.T
                for r in range(sims.shape[0]):
                    sims[r, i + r] = -1.0
                idx = np.argpartition(-sims, ANN_K, axis=1)[:, :ANN_K]
                for r in range(sims.shape[0]):
                    a = ueis[i + r]
                    for c in idx[r]:
                        out.add(frozenset((a, ueis[int(c)])))
                if (i // step) % 20 == 0:
                    print(f"    ann {i:,}/{len(ueis):,}", flush=True)
            return out

        timed(f"Embedding ANN (MiniLM, k={ANN_K})", embedding_ann)

    string_keys = ["Token blocking", "Q-gram blocking (q=4)", "First-token blocking",
                   "Sorted neighbourhood (w=20)", "Phonetic (Soundex)"]
    schemes["Union: string-keyed"] = (
        set().union(*[schemes[k][0] for k in string_keys if k in schemes]), 0.0)
    schemes["Union: all schemes"] = (
        set().union(*[v[0] for k, v in schemes.items()
                      if not k.startswith("Union")]), 0.0)

    results = {}
    for label, (cands, secs) in schemes.items():
        cov = Counter(truth[c] for c in cands if c in truth)
        got = sum(cov.values())
        pc = got / len(truth) if truth else 0.0
        rr = 1.0 - (len(cands) / all_pairs) if all_pairs else 0.0
        pq = got / len(cands) if cands else 0.0
        results[label] = {
            "candidates": len(cands),
            "pair_completeness_pct": round(100 * pc, 2),
            "reduction_ratio_pct": round(100 * rr, 4),
            "pairs_quality_pct": round(100 * pq, 4),
            "build_seconds": round(secs, 2),
            "keyed_on_core_tokens": label in ("Token blocking",
                                              "First-token blocking"),
            "by_visibility": {
                v: {"recovered": cov.get(v, 0), "total": t,
                    "pair_completeness_pct": round(100.0 * cov.get(v, 0) / t, 2)}
                for v, t in by_vis.items()},
        }
        v = results[label]["by_visibility"]
        print(f"{label:34} PC={100*pc:5.1f}%  RR={100*rr:8.4f}%  "
              f"cands={len(cands):>11,}  "
              f"INVISIBLE={v.get('invisible', {}).get('pair_completeness_pct', 0):5.2f}",
              flush=True)

    out = {
        "roster_entities": n,
        "naive_pair_space": all_pairs,
        "true_links": len(truth),
        "true_links_by_visibility": dict(by_vis),
        "max_block_size_purged_above": MAX_BLOCK,
        "ann_k": ANN_K,
        "note_identity": (
            "Token blocking and first-token blocking key on core() and the invisible "
            "stratum is defined by an empty core() intersection, so their zero recall "
            "there is an identity rather than a measurement. The informative rows are "
            "the schemes not keyed on core(): q-gram, sorted neighbourhood, phonetic, "
            "attribute and embedding ANN."),
        "schemes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
