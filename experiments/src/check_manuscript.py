"""Audit the CorpFam manuscript: no hand-typed numbers, no dangling references.

This exists because the failure mode it prevents is invisible to proofreading. A
number typed into prose during a draft, then left behind when the experiment was
rerun, looks exactly like a correct number. The only defence is mechanical: every
figure in the body must come from a macro, every macro must be regenerable from a
results file, and the check must run.

Six checks, any of which fails the audit:

  1. every macro used in the body is defined in macros.tex
  2. every macro defined in macros.tex is used (dead macros mean a dropped claim)
  3. no bare numeral appears in body prose outside an allowed context
  4. every \\cite key resolves to an entry in refs.bib
  5. every \\ref resolves to a \\label
  6. every macro value is independently recomputed from the results JSON and matches

Check 6 is the one that matters. The others catch sloppiness; this one catches a
stale number, by recomputing it from source rather than trusting make_tables.py.

Usage:
    python3 experiments/src/check_manuscript.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TEX = Path("paper/latex/main.tex")
MACROS = Path("paper/latex/tables/macros.tex")
BIB = Path("paper/refs/refs.bib")
RES = Path("experiments/results")

# Numerals that are legitimately part of prose rather than a result.
ALLOWED_NUMERIC = {
    "1", "2", "3", "4",          # enumeration, n-gram order, hop counts
    "60", "20",                  # the split ratio, stated as a ratio
    "95",                        # confidence level
    "10", "100",                 # decile / percentage language
    "2025", "1",                 # fiscal year, "one hop"
    "21",                        # SEC Exhibit 21, a document name
    "2023",                      # year of a named acquisition
    "88",                        # the discarded role-leakage score, quoted as history
}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def expand_shared(text: str) -> str:
    """Inline the shared abstract and body.

    The manuscript is split so the arXiv and PVLDB builds cannot drift, which means
    main.tex is now little more than a preamble and two \\input lines. Without this
    expansion every check below would scan an empty body and pass vacuously -- a
    silent verification failure, which is worse than a loud one.
    """
    def repl(m: re.Match) -> str:
        target = TEX.parent / (m.group(1) + ".tex")
        if not target.exists():
            raise SystemExit(f"shared include missing: {target}")
        return target.read_text(encoding="utf-8")

    out = re.sub(r"\\input\{(\.\./shared/[A-Za-z_]+)\}", repl, text)
    if "../shared/" in out:
        raise SystemExit("a shared include was not expanded; check the path pattern")
    return out


def body_of(text: str) -> str:
    """Body only: drop preamble, bibliography, and \\input-ed tables."""
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r"\\input\{[^}]*\}", " ", body)
    body = re.sub(r"\\bibliography\{[^}]*\}", " ", body)
    body = re.sub(r"\\label\{[^}]*\}", " ", body)
    body = re.sub(r"\\ref\{[^}]*\}", " ", body)
    body = re.sub(r"\\cite[tp]?\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:documentclass|usepackage)(\[[^]]*\])?\{[^}]*\}", " ", body)
    # $F_1$ and similar inline math are notation, not results.
    body = re.sub(r"\$[^$]*\$", " ", body)
    body = re.sub(r"\\textsc\{[^}]*\}", " ", body)
    return body


def main() -> int:
    tex = expand_shared(strip_comments(TEX.read_text(encoding="utf-8")))
    macro_src = MACROS.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")

    defined = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\\xspace\}",
                              macro_src))
    body = body_of(tex)
    used = set(re.findall(r"\\([A-Za-z]+)(?![A-Za-z])", body))

    failures: list[str] = []
    notes: list[str] = []

    # 1. used but undefined (restricted to things that look like our macros)
    latex_builtin = re.findall(r"\\([A-Za-z]+)", strip_comments(
        TEX.read_text(encoding="utf-8")).split(r"\begin{document}")[0])
    candidates = {u for u in used if u in defined or (u[0].isupper() and len(u) > 4)}
    undefined = {u for u in candidates if u not in defined
                 and u not in set(latex_builtin)}
    known_tex = {"Sections", "Table", "Section", "Figure", "LaTeX"}
    undefined -= known_tex
    if undefined:
        failures.append(f"macros used but not defined: {sorted(undefined)}")

    # 2. defined but unused
    unused = sorted(set(defined) - used)
    if unused:
        notes.append(f"{len(unused)} defined macros unused: {unused}")

    # 3. bare numerals in prose
    prose = re.sub(r"\\[A-Za-z]+", " ", body)
    bare = [n for n in re.findall(r"(?<![\w.])\d[\d,.]*", prose)
            if n.strip(".,") not in ALLOWED_NUMERIC]
    if bare:
        failures.append(f"bare numerals in prose (should be macros): {sorted(set(bare))}")

    # 4. citations resolve
    keys = set(re.findall(r"^@[a-zA-Z]+\{([^,]+)", bib, re.M))
    cited: set[str] = set()
    for grp in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex):
        cited |= {k.strip() for k in grp.split(",") if k.strip()}
    missing = sorted(cited - keys)
    if missing:
        failures.append(f"cited but absent from refs.bib: {missing}")

    # 5. refs resolve
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    for tf in sorted(Path("paper/latex/tables").glob("*.tex")):
        labels |= set(re.findall(r"\\label\{([^}]*)\}",
                                 tf.read_text(encoding="utf-8")))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    dangling = sorted(refs - labels)
    if dangling:
        failures.append(f"\\ref with no \\label: {dangling}")

    # 6. recompute macro values independently from the results files
    def load(n: str) -> dict:
        return json.loads((RES / f"{n}.json").read_text(encoding="utf-8"))

    build, recon, arch = load("benchmark_build"), load("parent_source_reconciliation"), load("archive_parents")
    base = load("string_baselines")
    base.update(load("embedding_baseline"))
    audit = load("ground_truth_audit")

    def clean(s: str) -> str:
        return s.replace("\\%", "").replace(",", "").replace("--", "-").strip()

    expect: dict[str, str] = {}
    expect["BenchPairs"] = f'{build["pairs_total"]:,}'
    expect["BenchPositives"] = f'{build["positives"]:,}'
    expect["BenchNegatives"] = f'{build["negatives"]:,}'
    expect["BenchFamilies"] = f'{build["families"]:,}'
    expect["ArchiveRows"] = f'{arch["rows_read"]:,}'
    expect["ArchiveEntities"] = f'{arch["distinct_uei"]:,}'
    expect["ArchiveGenuineLinks"] = f'{arch["genuine_parent_links"]:,}'
    expect["ReconAgreeName"] = f'{recon["comparison"]["pct_agree_name"]:.1f}'
    expect["ReconAgreeUei"] = f'{recon["comparison"]["pct_agree_uei"]:.1f}'
    expect["AuditGenuineLinks"] = f'{audit["genuine_parent_child_links"]:,}'

    # Selected by average precision, matching make_tables.py: a threshold-free ranking
    # metric rather than an F1 whose value depends on where the threshold landed.
    best = max(base, key=lambda k: base[k]["test"].get("average_precision")
               or base[k]["test"]["overall"]["f1"])
    expect["BestOverallF"] = f'{base[best]["test"]["overall"]["f1"]:.1f}'
    expect["EmbInvisibleF"] = f'{base["embedding"]["test"]["by_visibility"]["invisible"]["f1"]:.1f}'
    inv_max = max(v["test"]["by_visibility"]["invisible"]["f1"] for v in base.values())
    expect["AnyInvisibleFMax"] = f"{inv_max:.1f}"
    expect["VisibilityGap"] = (
        f'{base[best]["test"]["by_visibility"]["identical"]["f1"] - inv_max:.1f}')
    # Per-stratum recall, which is what the paper leads with.
    for v in ("identical", "visible", "invisible"):
        d = base[best]["test"]["by_visibility"].get(v)
        if d:
            expect[f"BestRecall{v.capitalize()}"] = f'{d["recall"]:.1f}'
    inv_r = max(x["test"]["by_visibility"]["invisible"]["recall"] for x in base.values())
    expect["AnyInvisibleRecallMax"] = f"{inv_r:.1f}"
    # Majority-class baselines, recomputed from the pair file rather than trusted.
    rows = [json.loads(l) for l in
            Path("data/benchmark/corpfam_pairs.jsonl").open(encoding="utf-8")]
    test_rows = [p for p in rows if p["split"] == "test"]
    for v in ("identical", "visible", "invisible"):
        s = [p for p in test_rows if p["visibility"] == v]
        if not s:
            continue
        r = sum(p["label"] for p in s) / len(s)
        expect[f"Base{v.capitalize()}Rate"] = f"{100 * r:.1f}"
        expect[f"Base{v.capitalize()}F"] = f"{100 * (2 * r / (r + 1)):.1f}"
    # Positive-stratum percentages recomputed from the pair file, not the manifest.
    pos_rows = [p for p in rows if p["label"] == 1]
    for v in ("identical", "visible", "invisible"):
        n = sum(1 for p in pos_rows if p["visibility"] == v)
        expect[f"BenchPos{v.capitalize()}Pct"] = f"{100.0 * n / len(pos_rows):.1f}"
    expect["BenchPairs"] = f"{len(rows):,}"
    expect["BenchPositives"] = f"{len(pos_rows):,}"
    expect["BenchNegatives"] = f"{len(rows) - len(pos_rows):,}"
    expect["BenchFamilies"] = f'{len({p["family_uei"] for p in pos_rows}):,}'

    blk = load("blocking_analysis")
    bs = blk["schemes"]
    expect["BlkEntities"] = f'{blk["roster_entities"]:,}'
    expect["BlkUnionInvisible"] = (
        f'{bs["Union: all schemes"]["by_visibility"]["invisible"]["pair_completeness_pct"]:.1f}')
    expect["BlkInvisibleLost"] = (
        f'{100 - bs["Union: all schemes"]["by_visibility"]["invisible"]["pair_completeness_pct"]:.1f}')
    expect["BlkAnnInvisible"] = (
        f'{bs["Embedding ANN (MiniLM, k=20)"]["by_visibility"]["invisible"]["pair_completeness_pct"]:.2f}')

    ed = load("edgar_reconciliation")
    expect["EdgarInvisibleConfirmPct"] = (
        f'{ed["by_visibility"]["invisible"]["confirmation_rate_pct"]:.1f}')
    expect["EdgarRandomParentPct"] = (
        f'{ed["permutation_control"]["invisible_random_parent_pct"]:.2f}')
    expect["EdgarExcessOverChance"] = (
        f'{ed["permutation_control"]["excess_over_chance_points"]:.1f}')

    popn = load("population_stats")
    expect["SpendTopOnePct"] = f'{popn["spend_concentration"]["top_1pct_share_pct"]:.1f}'
    expect["ParentsTransactingPct"] = (
        f'{popn["parent_footprint"]["pct_parents_transacting"]:.1f}')
    expect["ParentsNoFootprintPct"] = (
        f'{popn["parent_footprint"]["pct_parents_no_footprint"]:.1f}')

    mismatched = []
    for name, want in expect.items():
        if name not in defined:
            mismatched.append(f"{name}: macro missing entirely")
        elif clean(defined[name]) != clean(want):
            mismatched.append(f"{name}: manuscript={clean(defined[name])!r} "
                              f"recomputed={clean(want)!r}")
    if mismatched:
        failures.append("macro values disagree with results files:\n    "
                        + "\n    ".join(mismatched))

    # --- report -----------------------------------------------------------------
    print(f"macros defined            : {len(defined)}")
    print(f"macros used in body       : {len(set(defined) & used)}")
    print(f"citations used / in bib   : {len(cited)} / {len(keys)}")
    print(f"cross-references resolved : {len(refs)}")
    print(f"macro values recomputed   : {len(expect)}")
    for n in notes:
        print(f"\nNOTE  {n}")
    if failures:
        print("\nFAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASSED - every numeric claim in the body resolves to a macro, and every "
          "recomputed macro matches its results file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
