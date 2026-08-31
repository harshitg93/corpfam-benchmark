"""Multi-attribute baseline: does address evidence rescue the invisible stratum?

The sharpest objection to this paper is that it is circular. Pairs are labelled
"invisible" precisely because their names share nothing, and then name-based methods
are shown to fail on them. A reviewer can reasonably say that real entity resolution
uses addresses, and that established benchmarks are multi-attribute for that reason.

The objection deserves a measurement, not a rebuttal. The archive carries address,
city, state, ZIP, country, phone, and a doing-business-as name for every recipient, so
this trains a supervised classifier over name *and* attribute features and reports it
on the same protocol as everything else.

Three feature sets are compared, and the comparison is the result:

  name_only        the existing string signals, as a control
  attributes_only  address, geography, phone -- no name similarity whatsoever
  combined         both

If attributes were the missing ingredient, `attributes_only` should do well on the
invisible stratum, where names carry nothing. Whether it does is an empirical question
with a publishable answer either way: if addresses help, that is a useful finding about
how to build these systems; if they do not, the stratum is hard for reasons no amount
of conventional attribute engineering will fix, and the circularity objection is
answered on the evidence.

Usage:
    PYTHONPATH=experiments/src python3 experiments/src/run_multiattribute_baseline.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from run_string_baselines import TfidfChar, core, evaluate, norm

TOKEN = re.compile(r"[a-z0-9]+")
STREET = {"st", "street", "ave", "avenue", "rd", "road", "dr", "drive", "blvd",
          "boulevard", "ste", "suite", "fl", "floor", "n", "s", "e", "w", "north",
          "south", "east", "west", "po", "box", "hwy", "highway", "pkwy", "parkway",
          "ln", "lane", "ct", "court", "cir", "circle", "way", "plaza", "unit"}


def norm_addr(s: str) -> str:
    return " ".join(t for t in TOKEN.findall(str(s or "").lower())
                    if t not in STREET)


def digits(s: str) -> str:
    return re.sub(r"\D", "", str(s or ""))


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - m), 2), round(100 * min(1.0, c + m), 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--attrs", type=Path,
                    default=Path("data/raw/usaspending/entity_attributes.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/multiattribute_baseline.json"))
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    attrs: dict[str, dict] = {}
    for line in args.attrs.open(encoding="utf-8"):
        r = json.loads(line)
        attrs[r["uei"]] = r

    rows = [json.loads(l) for l in args.pairs.open(encoding="utf-8") if l.strip()]
    by_split = defaultdict(list)
    for p in rows:
        by_split[p["split"]].append(p)

    tfidf = TfidfChar()
    tfidf.fit([p["left_name"] for p in by_split["train"]]
              + [p["right_name"] for p in by_split["train"]])

    NAME_F = ["tfidf", "jaccard", "seqratio", "exact_name"]
    ATTR_F = ["same_address", "addr_sim", "same_city", "same_state", "same_zip5",
              "same_phone", "same_country", "dba_hit", "both_have_addr"]

    def feats(p: dict) -> dict[str, float]:
        a, b = p["left_name"], p["right_name"]
        ca, cb = core(a), core(b)
        f = {
            "tfidf": tfidf.score(a, b),
            "jaccard": (len(ca & cb) / len(ca | cb)) if (ca and cb) else 0.0,
            "seqratio": SequenceMatcher(None, norm(a), norm(b)).ratio(),
            "exact_name": 1.0 if norm(a) == norm(b) else 0.0,
        }
        ra, rb = attrs.get(p["left_uei"], {}), attrs.get(p["right_uei"], {})
        aa, ab = norm_addr(ra.get("address")), norm_addr(rb.get("address"))
        both = 1.0 if (aa and ab) else 0.0
        f["both_have_addr"] = both
        f["same_address"] = 1.0 if (both and aa == ab) else 0.0
        f["addr_sim"] = (SequenceMatcher(None, aa, ab).ratio() if both else 0.0)
        f["same_city"] = 1.0 if (ra.get("city") and
                                 norm(ra.get("city")) == norm(rb.get("city"))) else 0.0
        f["same_state"] = 1.0 if (ra.get("state") and
                                  ra.get("state") == rb.get("state")) else 0.0
        za, zb = digits(ra.get("zip"))[:5], digits(rb.get("zip"))[:5]
        f["same_zip5"] = 1.0 if (za and za == zb) else 0.0
        pa, pb = digits(ra.get("phone")), digits(rb.get("phone"))
        f["same_phone"] = 1.0 if (len(pa) >= 10 and pa == pb) else 0.0
        f["same_country"] = 1.0 if (ra.get("country") and
                                    ra.get("country") == rb.get("country")) else 0.0
        # A doing-business-as name can carry the parent brand even when the legal
        # name does not, which is one of the few non-name signals that could
        # plausibly bridge the invisible stratum.
        dba_a, dba_b = ra.get("dba", ""), rb.get("dba", "")
        hit = 0.0
        if dba_a and (core(dba_a) & cb):
            hit = 1.0
        if dba_b and (core(dba_b) & ca):
            hit = 1.0
        f["dba_hit"] = hit
        return f

    print("computing features ...", flush=True)
    F = {sp: [feats(p) for p in by_split[sp]] for sp in ("train", "val", "test")}

    cov = sum(1 for f in F["test"] if f["both_have_addr"]) / max(len(F["test"]), 1)
    print(f"test pairs where both sides have an address: {100*cov:.1f}%", flush=True)

    # Attribute features can only speak when both sides carry attributes. Parents
    # frequently never transact and so have no address at all, which leaves most pairs
    # with an all-zero attribute vector: evaluated over the full split, an
    # attribute-only model has nothing to learn from and collapses to predicting the
    # majority class, producing a meaningless F1 at the base rate. The honest question
    # is whether attributes help *where they exist*, so the primary evaluation is
    # restricted to pairs with attributes on both sides and the covered subset is
    # reported alongside.
    sub = {sp: [i for i, f in enumerate(F[sp]) if f["both_have_addr"]] for sp in F}
    print("pairs with attributes on both sides: "
          + ", ".join(f"{sp}={len(sub[sp]):,}" for sp in ("train", "val", "test")),
          flush=True)

    sub_strata = Counter(by_split["test"][i]["visibility"]
                         for i in sub["test"] if by_split["test"][i]["label"] == 1)

    def run(cols: list[str], idx: dict[str, list[int]] | None, tag: str) -> dict:
        def take(sp: str, arr: list) -> list:
            return arr if idx is None else [arr[i] for i in idx[sp]]

        Xs = {sp: np.array([[f[c] for c in cols] for f in take(sp, F[sp])])
              for sp in F}
        ys = {sp: np.array([p["label"] for p in take(sp, by_split[sp])]) for sp in F}
        items = {sp: take(sp, by_split[sp]) for sp in F}
        if len(set(ys["train"])) < 2 or len(ys["val"]) < 10:
            return {"error": "insufficient data in this subset"}

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xs["train"], ys["train"])
        sv = clf.predict_proba(Xs["val"])[:, 1].tolist()
        st = clf.predict_proba(Xs["test"])[:, 1].tolist()

        cands = sorted({round(s, 3) for s in sv})
        if len(cands) > 200:
            cands = cands[::len(cands) // 200]
        bt, bf = 0.5, -1.0
        for t in cands:
            f1 = evaluate(items["val"], sv, t)["overall"]["f1"]
            if f1 > bf:
                bf, bt = f1, t
        res = evaluate(items["test"], st, bt)
        for v, d in res["by_visibility"].items():
            d["recall_wilson95"] = wilson(d["tp"], d["tp"] + d["fn"])
        # A model that fires on essentially everything has learned nothing; say so
        # rather than letting a base-rate F1 be read as a result.
        res["degenerate_all_positive"] = bool(res["overall"]["recall"] > 99.0
                                              and res["overall"]["precision"] < 40.0)
        return {"features": cols, "n_test": len(items["test"]),
                "coefficients": {c: round(float(w), 4)
                                 for c, w in zip(cols, clf.coef_[0])},
                "validation_f1": bf, "test": res}

    results = {}
    for setname, cols in (("name_only", NAME_F),
                          ("attributes_only", ATTR_F),
                          ("combined", NAME_F + ATTR_F)):
        results[f"{setname}__attributed_subset"] = run(cols, sub, setname)
        X = {sp: np.array([[f[c] for c in cols] for f in F[sp]]) for sp in F}
        y = {sp: np.array([p["label"] for p in by_split[sp]]) for sp in F}

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X["train"], y["train"])
        sv = clf.predict_proba(X["val"])[:, 1].tolist()
        st = clf.predict_proba(X["test"])[:, 1].tolist()

        cands = sorted({round(s, 3) for s in sv})
        if len(cands) > 200:
            cands = cands[::len(cands) // 200]
        best_thr, best_f1 = 0.5, -1.0
        for t in cands:
            f1 = evaluate(by_split["val"], sv, t)["overall"]["f1"]
            if f1 > best_f1:
                best_f1, best_thr = f1, t

        res = evaluate(by_split["test"], st, best_thr)
        for v, d in res["by_visibility"].items():
            d["recall_wilson95"] = wilson(d["tp"], d["tp"] + d["fn"])
        res["degenerate_all_positive"] = bool(res["overall"]["recall"] > 99.0
                                              and res["overall"]["precision"] < 40.0)
        results[f"{setname}__full_split"] = {
            "features": cols,
            "n_test": len(by_split["test"]),
            "coefficients": {c: round(float(w), 4)
                             for c, w in zip(cols, clf.coef_[0])},
            "validation_f1": best_f1,
            "test": res,
        }
        for scope in ("attributed_subset", "full_split"):
            r = results.get(f"{setname}__{scope}", {})
            if "test" not in r:
                continue
            t = r["test"]
            v = t["by_visibility"]
            flag = "  [DEGENERATE]" if t.get("degenerate_all_positive") else ""
            print(f"{setname:16} {scope:18} n={r['n_test']:>6,}  "
                  f"F1={t['overall']['f1']:5.1f}  "
                  f"ident={v.get('identical', {}).get('f1', 0):5.1f}  "
                  f"vis={v.get('visible', {}).get('f1', 0):5.1f}  "
                  f"INVIS={v.get('invisible', {}).get('f1', 0):5.1f}{flag}", flush=True)

    # How often do genuine parent-child pairs actually share an address? This is the
    # mechanism question behind the numbers above.
    share = Counter()
    tot = Counter()
    for p, f in zip(by_split["test"], F["test"]):
        if p["label"] == 1 and f["both_have_addr"]:
            tot[p["visibility"]] += 1
            if f["same_address"]:
                share[p["visibility"]] += 1

    out = {
        "attribute_source": "FY2025 award archive, all recipient rows",
        "entities_with_attributes": len(attrs),
        "test_pairs_with_address_on_both_sides_pct": round(100 * cov, 2),
        "attributed_subset_sizes": {sp: len(sub[sp]) for sp in sub},
        "attributed_subset_positive_strata": dict(sub_strata),
        "primary_scope": ("attributed_subset -- pairs with attributes on both sides. "
                          "Over the full split an attribute-only model has no signal "
                          "for ~92% of pairs and degenerates to predicting the "
                          "majority class, so its full-split F1 is a base-rate "
                          "artifact rather than a measurement."),
        "feature_sets": results,
        "true_pairs_sharing_an_address": {
            v: {"pairs_with_both_addresses": tot[v], "same_address": share.get(v, 0),
                "pct": round(100.0 * share.get(v, 0) / tot[v], 2) if tot[v] else None}
            for v in sorted(tot)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\ntrue parent-child pairs sharing an address, by stratum:")
    for v, d in out["true_pairs_sharing_an_address"].items():
        print(f"  {v:10} {d['same_address']:>5}/{d['pairs_with_both_addresses']:<6} "
              f"= {d['pct']}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
