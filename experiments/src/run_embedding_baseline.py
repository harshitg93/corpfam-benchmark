"""Sentence-embedding baseline on CorpFam.

The obvious objection to the string results is that string methods are simply the
wrong tool, and that a semantic encoder would close the gap. This script tests that
directly, under exactly the protocol used for the string baselines: cosine similarity
between encoded names, threshold chosen on validation, applied unchanged to test,
reported per visibility stratum.

The hypothesis being tested is that it will not close the gap, and the reason matters
more than the number. That HEICO owns Blue Aerospace is not a fact about the meaning
of either string. No amount of semantic similarity recovers it, because the
relationship is not encoded in the names at all -- it is a fact about ownership that
lives in a filing. If a strong encoder also collapses on the invisible stratum, then
the stratum is measuring knowledge rather than similarity, and that is the finding.

Usage:
    python3 experiments/src/run_embedding_baseline.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_string_baselines import evaluate  # identical protocol, identical metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/embedding_baseline.json"))
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    pairs = [json.loads(l) for l in args.pairs.open(encoding="utf-8") if l.strip()]
    by_split = defaultdict(list)
    for p in pairs:
        by_split[p["split"]].append(p)

    need = sorted({p["left_name"] for p in pairs} | {p["right_name"] for p in pairs})
    print(f"{len(pairs):,} pairs, {len(need):,} unique names to encode", flush=True)

    model = SentenceTransformer(args.model)
    emb = model.encode(need, batch_size=args.batch, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    idx = {n: i for i, n in enumerate(need)}
    print("encoded", emb.shape, flush=True)

    def cos(split: str) -> list[float]:
        ps = by_split[split]
        a = emb[[idx[p["left_name"]] for p in ps]]
        b = emb[[idx[p["right_name"]] for p in ps]]
        return (a * b).sum(axis=1).tolist()  # rows are unit-norm, so dot == cosine

    sc = {sp: cos(sp) for sp in ("val", "test")}

    cand = sorted({round(s, 3) for s in sc["val"]})
    if len(cand) > 200:
        cand = cand[::len(cand) // 200]
    best_thr, best_f1 = 0.5, -1.0
    for t in cand:
        f = evaluate(by_split["val"], sc["val"], t)["overall"]["f1"]
        if f > best_f1:
            best_f1, best_thr = f, t

    res = {
        "model": args.model,
        "unique_names_encoded": len(need),
        "embedding_dim": int(emb.shape[1]),
        "validation_f1_at_chosen_threshold": best_f1,
        "test": evaluate(by_split["test"], sc["test"], best_thr),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"embedding": res}, indent=2), encoding="utf-8")

    v = res["test"]["by_visibility"]
    print(f"\nthr={best_thr:.3f}  test F1={res['test']['overall']['f1']:.1f}  "
          f"identical={v.get('identical', {}).get('f1', 0):.1f}  "
          f"visible={v.get('visible', {}).get('f1', 0):.1f}  "
          f"INVISIBLE={v.get('invisible', {}).get('f1', 0):.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
