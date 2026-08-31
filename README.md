# CorpFam: A Public Benchmark for Corporate-Family Resolution

Deciding whether two supplier records belong to the same **corporate family** — an
acquired subsidiary and its ultimate parent — is a prerequisite for spend
consolidation, credit exposure aggregation, and sanctions screening. It is usually
evaluated as a special case of entity matching. This benchmark argues that framing
hides a near-total failure, and gives the data to show it.

**Headline finding — the pipeline fails before matching does.** Blocking decides which
pairs a matcher ever sees. Across seven schemes over the full 114k-entity recipient
roster — including phonetic keys, attribute keys that ignore the name entirely, and
semantic nearest neighbours, **none of which is a function of the token overlap that
defines the stratum** — no scheme exceeds 3% on name-invisible links, and their union
reaches **6.8%**. Roughly **93% of the hard links never enter the candidate set**, so no
improvement at the matching stage can reach them.

At the matching stage the best method recovers **100% of identical pairs and 4.2% of
invisible ones**.

**We report per-stratum recall, not F1.** The strata have positive rates of 97.3%, 10.2%
and 16.9%, so a classifier that answers "yes" to everything scores F1 98.7 / 18.5 / 29.0
without doing anything. Per-stratum F1 largely measures the base rate; recall depends
only on a stratum's own positives.

**These links are real, not dirty labels.** Against SEC Exhibit 21 subsidiary schedules —
filed under securities law, sharing no provenance with procurement data — **48.5%** of
invisible links are corroborated, versus **0.33%** when the same procedure is run against
permuted parents. That is a 48-point excess over chance.

## The benchmark

| | |
|---|---|
| Candidate pairs | 54,864 |
| Positive pairs | 13,716 |
| Corporate families | 10,307 |
| Source award rows | 6,638,350 (FY2025 US federal contracts) |
| Splits | 60/20/20 over **connected components**, so no entity and no normalised name appears in two splits |

Every pair carries a `visibility` label, which is the primary axis of evaluation:

| Stratum | Definition | Positives | Best match recall | All-yes baseline F1 | Blocking recall (union) |
|---|---|---|---|---|---|
| `identical` | names equal after normalisation | 7,541 (55.0%) | 100.0% | 98.7 | 100.0% |
| `visible` | share ≥1 distinctive token | 2,953 (21.5%) | 49.9% | 18.5 | 91.0% |
| `invisible` | share **no** distinctive token | 3,222 (23.5%) | **4.2%** | 29.0 | **6.8%** |

Negatives are generated at three grades — `random`, `hard_string` (shares a distinctive
token), and `same_block` (shares a blocking key) — because evaluating against random
negatives alone is the standard way to overstate precision.

**Every negative pairs a child with a different family's parent**, matching the role
structure of a positive. An earlier build drew negatives as child–child pairs; because
parent entities rarely transact and so carry no address, "both sides have an address"
separated the classes almost perfectly and a supervised model hit 88 F1 on the invisible
stratum by learning entity *role* rather than ownership. Any benchmark of hierarchical
relations is exposed to this, since the two ends of a hierarchical edge are different
kinds of object.

## Ground truth, and its limits

Labels come from the **ultimate parent** field that suppliers self-report during US
federal registration. They are not hand-annotated and not model-derived.

Two things are documented rather than quietly fixed:

- **The self-parent trap.** 86.3% of entities reporting a parent report *themselves*.
  Treating every populated parent field as a family link would build a dataset of
  entities joined to themselves, trivially solvable and meaningless.
- **The registry disagrees with itself.** Two independent views of the same
  registration — the per-entity API and the bulk archive — agree on the exact parent
  UEI in only 64.9% of cases and on the normalised parent name in 73.8%. That bounds
  what any method can honestly be credited with.

Family artifacts (the archive claims a moving company is the parent of 103
waste-management entities) were **adjudicated by hand**, not filtered by rule. The
obvious automatic rule also flags Arctic Slope Regional Corporation, HEICO, TransDigm
and Berkshire Hathaway — real conglomerates that look exactly like mislabelled families
from the outside, and the most valuable rows in the dataset. All 8 exclusions are listed
with a written reason in `experiments/results/benchmark_build.json`.

## Two release tiers

The benchmark publishes company names in bulk, so redistribution was checked rather than
assumed. Source data is public-domain Treasury award data, but SAM.gov's terms restrict
bulk use of "D&B Open Data" (which includes legal business name) for records tied to base
awards dated before **4 April 2022**.

The definitive test in those terms is an EVS Source field that USAspending award data does
not carry, so the exposure is **bounded**, not measured, using the earliest
performance-period start across each link's supporting awards:

| Release | Positives | Pairs | Use when |
|---|---|---|---|
| `corpfam_pairs.jsonl` | 13,716 | 54,864 | Default |
| `corpfam_pairs_conservative.jsonl` | 6,522 | 16,889 | You need the strict D&B reading |
| `benchmark_unfiltered/corpfam_pairs.jsonl` | 14,288 | — | Sensitivity: no hand exclusions at all |

**Exclusion sensitivity.** Hand adjudication removes 8 declared parents, and 84.7% of the
links it drops are name-invisible — the stratum the paper is about. So the unfiltered
build ships too, and the headline is *stronger* there: best invisible recall is **0.9%**
without exclusions versus 4.2% with them. Removing the filter makes the task harder, not
easier.

The 52.5% flagged as pre-cutoff is an **upper bound**: a FY2025 task order against a
long-running vehicle inherits that vehicle's original start date and is counted pre-cutoff
even though the order is recent. Under an action-date reading — every transaction in this
archive is an FY2025 action — the exposure is zero. The truth is in between, so both tiers
ship.

## Reproducing

No third-party dependencies are needed for the benchmark or the string baselines.

```bash
# Build
python3 experiments/src/extract_archive_parents.py     # archive -> parent links
python3 experiments/src/detect_family_artifacts.py     # candidate artifacts for review
python3 experiments/src/build_benchmark.py             # -> data/benchmark/corpfam_pairs.jsonl
python3 experiments/src/extract_entity_attributes.py   # address/phone/DBA per entity

# Ground-truth validation
python3 experiments/src/reconcile_parent_sources.py    # API vs archive agreement
python3 experiments/src/fetch_edgar_exhibit21.py       # SEC Exhibit 21 retrieval
python3 experiments/src/reconcile_edgar.py             # independent corroboration
python3 experiments/src/check_dnb_provenance.py        # redistribution exposure bound
python3 experiments/src/make_conservative_subset.py    # -> conservative release tier

# Evaluation
python3 experiments/src/run_blocking_analysis.py       # candidate generation ceiling
python3 experiments/src/run_string_baselines.py        # 4 string baselines
export PY=PYTHONPATH=experiments/src
$PY python3 experiments/src/run_embedding_baseline.py     # sentence-transformers
$PY python3 experiments/src/run_multiattribute_baseline.py # sklearn
$PY python3 experiments/src/run_clustering_task.py         # family partition task

# Paper
python3 experiments/src/make_tables.py                 # -> paper/latex/tables/*.tex
python3 experiments/src/check_manuscript.py            # audit the manuscript
cd paper/latex && tectonic -X compile main.tex         # arXiv build
cd paper/pvldb && tectonic -X compile main.tex         # PVLDB build
```

The build is seeded (`20260827`) and deterministic.

### Withdrawn: the language-model baseline

An earlier revision included a local open-weight LLM as a sixth matcher, with a
knowledge probe alongside it. **It was withdrawn and its results deleted.** Padded
batches produced NaN logits on the MPS backend, which silently voided a full run before
the cause was found, and the rerun did not establish enough confidence in the scorer to
publish a number from it. The scripts and their results files were removed rather than
left in the tree, because dead code implying a missing experiment is worse than an
absent experiment.

Nothing in the manuscript depends on it. The reported baselines are the four string
methods and the sentence encoder, and every "no method exceeds" claim is quantified over
exactly those five. If you want an LLM number for this benchmark, run one — the
stratification is the point of the release, and it will make the result interpretable.

## No hand-typed numbers

Every figure in the manuscript is defined as a LaTeX macro generated by
`make_tables.py` from a results JSON file. `check_manuscript.py` enforces this: it
fails if a bare numeral appears in prose, if a macro is used but undefined, if a
citation or cross-reference dangles, or — most importantly — if a macro's value
disagrees with a value **recomputed independently** from the results files.

```
$ python3 experiments/src/check_manuscript.py
macro values recomputed   : 37
PASSED - every numeric claim in the body resolves to a macro, and every
recomputed macro matches its results file.
```

## Building the paper

```bash
python3 experiments/src/make_tables.py       # results JSON -> paper/latex/tables/
python3 experiments/src/check_manuscript.py  # audit: no hand-typed numbers
cd paper/latex && tectonic -X compile main.tex   # preprint build
cd paper/pvldb && tectonic -X compile main.tex   # PVLDB build, same body
python3 experiments/src/make_arxiv_bundle.py     # -> paper/arxiv-submission/
python3 experiments/src/pdf_pages.py             # page counts, incl. excluding refs
```

Both builds `\input` the same `paper/shared/body.tex`, so they cannot drift.
`paper/arxiv-submission/` is generated: it is the same manuscript flattened to one
directory with no parent-relative paths, because arXiv compiles from the archive root.

## Provenance

Everything derives from one download. The chain is auditable from the raw bytes, not
just from our outputs.

| Stage | Artefact | Recorded in |
|---|---|---|
| Source archive, 1.98 GB | `FY2025_All_Contracts_Full.zip`, SHA256 `69e90f43…74fcbb`, 7 CSV members | `experiments/results/source_archive_provenance.json` |
| Parent links | 6,638,350 award rows → 104,459 entities → 14,288 genuine links | `archive_parents.json` |
| Benchmark | 54,864 pairs, SHA256 `11fd85f0…514089`, seed `20260827` | `benchmark_build.json` |
| Independent check | SEC Exhibit 21, 132 filings | `data/raw/edgar/exhibit21.jsonl`, `edgar_reconciliation.json` |

`MANIFEST.md` in the release carries the SHA256 and byte size of every data file, and
the build refuses to release a benchmark whose digest disagrees with the one the results
were computed from.

**Adjudication is published in full.** `benchmark_build.json` holds the inspection log
for all 28 hand-reviewed families with a verdict and a written reason for each — the 20
retentions as well as the 8 exclusions. A log of rejections alone would be
indistinguishable from a purge of inconvenient cases, and the retentions are the
informative half: they record where the automatic rule was wrong. The complete list of 74
sovereign catch-all parents ships rather than a sample, and exclusions are keyed on
entity identifier rather than name string.

The 1.98 GB source archive is not redistributed. It is served in full by the publisher
and its SHA256 is recorded above, so a re-download is checkable.

## Data and ethics

All source data is public United States federal award data from USAspending.gov and
public SEC EDGAR filings. Both are public-domain United States Government work under
17 U.S.C. 105(a). No confidential, proprietary, or client information was used at any
stage. See `LICENSE` for the dual CC0 / MIT terms and for the one non-copyright
constraint, SAM.gov's D&B bulk-dissemination clause, which is why two release tiers ship.

## Layout

```
data/benchmark/                      the benchmark, full and conservative tiers
data/benchmark_unfiltered/           sensitivity build: no hand exclusions at all
data/raw/edgar/exhibit21.jsonl       SEC Exhibit 21 corpus, the independent check
experiments/src/                     build, baselines, audit, verification
experiments/results/                 every number in the paper, as JSON
paper/shared/                        abstract and body, shared by both builds
paper/latex/, paper/pvldb/           the two builds; tables/ is generated, do not edit
paper/arxiv-submission/              generated flat upload bundle
```

Internal design notes and venue planning are not part of this release. They are working
material rather than method, and nothing the paper claims depends on them.
