"""Assemble the public artifact release from the working repository.

PVLDB requires the availability URL to point at a public repository holding all
experimental data and software, with instructions a stranger can follow. That is almost
this repository, but not quite: two directories must not be published.

  notes/        internal working memory. Records the author's own constraints, motivating
                figures from a source that is not in the paper, and planned experiments
                that were never run. Publishing plans as though they were method invites
                exactly the "missing experiment" reading this release is meant to avoid.
  publication/  venue strategy and the arXiv endorsement drafts, which name individual
                researchers and rank them by likelihood of replying. Not ours to publish.

Everything else ships, including the EDGAR Exhibit 21 corpus, both benchmark tiers, the
unfiltered sensitivity build, every results file, and the manuscript source.

The release therefore gets its own git history rather than a branch of this one, because
a branch would carry the excluded directories in its history and publish them anyway.

Also written: MANIFEST.md, which records the SHA256 and byte size of every shipped data
file, and cross-checks the benchmark's own recorded digest so a download can be verified
against the file the paper was computed from.

Usage:
    python3 experiments/src/build_artifact.py /tmp/corpfam-artifact
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Copied wholesale. Directories are copied recursively, honouring EXCLUDE_SUFFIX.
INCLUDE = [
    "data/benchmark",
    "data/benchmark_unfiltered",
    "data/raw/edgar",
    "experiments/src",
    "experiments/results",
    "paper",
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "LICENSE",
    "CITATION.cff",
]
# Never published. Asserted rather than assumed: see the module docstring.
EXCLUDE_TREES = ["notes", "publication"]
EXCLUDE_SUFFIX = {".aux", ".blg", ".log", ".out", ".xdv", ".fls", ".fdb_latexmk",
                  ".synctex.gz", ".pyc"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_PATHS = {
    # The arXiv bundle ships source only; its PDF is a build artifact.
    "paper/arxiv-submission/main.pdf",
    # A literature review written as advice to the author -- "your sharpest theoretical
    # contribution", "you must report prompt sensitivity" -- and partly about a baseline
    # that was withdrawn. refs.bib is what the manuscript actually needs.
    "paper/refs/literature-landscape.md",
}

# Data files whose digests go in the manifest. Everything a reader downloads to use the
# benchmark, plus the evidence file behind the corroboration result.
DIGEST = [
    "data/benchmark/corpfam_pairs.jsonl",
    "data/benchmark/corpfam_pairs_conservative.jsonl",
    "data/benchmark_unfiltered/corpfam_pairs.jsonl",
    "data/raw/edgar/exhibit21.jsonl",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keep(rel: Path) -> bool:
    if rel.name in EXCLUDE_NAMES or rel.suffix in EXCLUDE_SUFFIX:
        return False
    if rel.as_posix() in EXCLUDE_PATHS:
        return False
    if any(part in EXCLUDE_TREES for part in rel.parts):
        return False
    return ".bbl" not in rel.suffix or rel.as_posix().startswith("paper/arxiv-submission")


def copy_into(dest: Path) -> list[Path]:
    copied: list[Path] = []
    for entry in INCLUDE:
        src = Path(entry)
        if not src.exists():
            if entry in {"LICENSE", "CITATION.cff"}:
                raise SystemExit(f"{entry} is missing; the release needs it")
            raise SystemExit(f"missing input: {entry}")
        if src.is_file():
            if not keep(src):
                continue
            (dest / src).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src)
            copied.append(src)
            continue
        for p in sorted(src.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(Path("."))
            if not keep(rel):
                continue
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / rel)
            copied.append(rel)
    return copied


def write_manifest(dest: Path, copied: list[Path]) -> dict:
    build = json.loads(Path("experiments/results/benchmark_build.json")
                       .read_text(encoding="utf-8"))
    prov = json.loads(Path("experiments/results/source_archive_provenance.json")
                      .read_text(encoding="utf-8"))

    digests = {}
    for rel in DIGEST:
        p = dest / rel
        if not p.exists():
            raise SystemExit(f"{rel} did not make it into the release")
        digests[rel] = (sha256(p), p.stat().st_size)

    recorded = build.get("sha256")
    primary = digests["data/benchmark/corpfam_pairs.jsonl"][0]
    if recorded and recorded != primary:
        raise SystemExit(
            "the shipped benchmark is not the file the results were computed from\n"
            f"  benchmark_build.json records {recorded}\n"
            f"  the shipped file hashes to   {primary}\n"
            "Rerun build_benchmark.py, or work out which one is stale, before releasing.")

    lines = [
        "# Release manifest",
        "",
        "Digests of every data file in the release. Verify a download with",
        "`shasum -a 256 <file>` and compare.",
        "",
        "| File | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for rel, (digest, size) in digests.items():
        lines.append(f"| `{rel}` | {size:,} | `{digest}` |")
    lines += [
        "",
        "## Source archive",
        "",
        "Everything derives from one download, recorded so the chain is auditable from",
        "the raw bytes rather than only from our outputs. It is not redistributed here:",
        f"it is {prov['bytes'] / 1e9:.2f} GB and is served in full by the publisher.",
        "",
        f"- File: `{prov['file']}`",
        f"- URL: {prov['url']}",
        f"- Bytes: {prov['bytes']:,}",
        f"- SHA256: `{prov['sha256']}`",
        f"- CSV members: {len(prov['csv_members'])}",
        "",
        "## Adjudication and exclusions",
        "",
        f"`experiments/results/benchmark_build.json` carries the full inspection log:",
        f"{len(build.get('inspection_log', {}))} families inspected by hand, each with a",
        "verdict and a written reason. **Retentions are published alongside rejections**,",
        "because a log of rejections only is indistinguishable from a purge of",
        "inconvenient cases, and the retentions record where the automatic rule was",
        "wrong. The complete sovereign catch-all parent list",
        f"(`sovereign_parents_excluded_all`, {len(build.get('sovereign_parents_excluded_all', []))} entries)",
        "ships in full rather than as a sample.",
        "",
        f"Exclusions are keyed on entity identifier, not on name string:",
        f"{len(build.get('excluded_parent_ueis', []))} parent UEIs excluded after",
        "adjudication. In a paper about name instability that is the only defensible key.",
        "",
        f"Build seed: `{build['seed']}`. The build is deterministic.",
        "",
        "## What is not in this release",
        "",
        "- The 2 GB source archive. Re-downloadable from the URL above; its SHA256 is",
        "  recorded so a re-download is checkable.",
        "- Per-entity API dumps and run logs. Regenerated by the fetch scripts.",
        "- Internal design notes and venue/submission planning. Working material, not",
        "  method; nothing the paper claims depends on it.",
        "",
    ]
    (dest / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")

    return {"files": len(copied) + 1, "digests": digests,
            "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    dest = Path(argv[1]).expanduser().resolve()
    if dest.exists():
        if not (dest / ".git").exists() and any(dest.iterdir()):
            raise SystemExit(f"{dest} is not empty and is not a git checkout")
        for p in sorted(dest.iterdir()):
            if p.name == ".git":
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    copied = copy_into(dest)

    # The two exclusions are the whole reason this script exists, so verify rather than
    # trust the copy logic.
    leaked = [str(p.relative_to(dest)) for p in dest.rglob("*")
              if p.is_file() and any(part in EXCLUDE_TREES
                                     for part in p.relative_to(dest).parts)]
    if leaked:
        raise SystemExit(f"excluded material leaked into the release: {leaked[:10]}")

    stats = write_manifest(dest, copied)

    print(f"{dest}: {stats['files']} files, {stats['bytes'] / 1e6:.1f} MB")
    for rel, (digest, size) in stats["digests"].items():
        print(f"  {size:>12,d} B  {digest[:16]}...  {rel}")
    print(f"\nexcluded from the release: {', '.join(EXCLUDE_TREES)} (verified absent)")
    if (dest / ".git").exists():
        r = subprocess.run(["git", "status", "--porcelain"], cwd=dest,
                           capture_output=True, text=True)
        changed = len([l for l in r.stdout.splitlines() if l.strip()])
        print(f"git checkout at destination: {changed} paths differ from HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
