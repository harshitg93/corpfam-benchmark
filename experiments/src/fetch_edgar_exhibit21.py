"""Cross-validate CorpFam parent-child links against SEC EDGAR Exhibit 21.

Both existing views of our ground truth -- the USAspending per-entity API and the bulk
award archive -- ultimately derive from the same federal registration. Agreement
between them bounds internal consistency but cannot establish correctness: if a
supplier misdeclared its parent at registration, both views inherit the error
identically.

Exhibit 21 breaks that dependency. It is the "Subsidiaries of the Registrant" exhibit
filed with a 10-K, prepared by the parent's own counsel under securities law rather
than by a vendor filling in a procurement form. Where a link appears in both sources,
it is corroborated by two independent legal regimes. Where EDGAR contradicts the
registry, we have located a real error rather than a disagreement between two copies
of the same claim.

Coverage will be partial by construction: only SEC registrants file 10-Ks, so private
parents, foreign parents, and government entities are absent. That is a limitation to
report, not a defect to hide -- and the subset it does cover is the high-spend subset
that matters most.

EDGAR requires a declared User-Agent and rate-limits to 10 requests/second; we stay
well under it.

Usage:
    python3 experiments/src/fetch_edgar_exhibit21.py --max-parents 400
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

UA = "CorpFam academic benchmark research (harshitg93@gmail.com)"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
RATE = 0.18  # seconds between requests, comfortably under EDGAR's 10/s

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


def norm(s: str) -> str:
    return " ".join(TOKEN.findall(str(s).lower()))


def core(s: str) -> set[str]:
    return {t for t in TOKEN.findall(str(s).lower())
            if t not in STOPWORDS and len(t) > 2}


class Edgar:
    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
        self.last = 0.0

    def get(self, url: str) -> requests.Response | None:
        wait = RATE - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = self.s.get(url, timeout=30)
            return r if r.status_code == 200 else None
        except requests.RequestException:
            return None


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</tr>|</p>|</div>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", "\t", html)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#8217;", "'").replace("&rsquo;", "'")
                .replace("&#38;", "&").replace("&quot;", '"'))
    return html


def parse_subsidiaries(text: str) -> list[str]:
    """Pull plausible company names out of an Exhibit 21.

    Exhibit 21 has no mandated format: some are HTML tables, some are flat lists, and
    a jurisdiction column is usually but not always present. Rather than guess at
    structure we take line/cell fragments and keep those that look like company names,
    which is imprecise in both directions but transparent about it.
    """
    out: list[str] = []
    for raw in re.split(r"[\n\r]+", strip_html(text)):
        for cell in raw.split("\t"):
            c = " ".join(cell.split())
            c = c.strip(" .,;:()[]|-_*")
            if not (3 < len(c) < 120):
                continue
            low = c.lower()
            if any(k in low for k in (
                    "subsidiar", "jurisdiction", "state of", "incorporat", "exhibit",
                    "registrant", "organization", "country", "percent", "ownership",
                    "table of", "page ", "name of")):
                continue
            if not re.search(r"[A-Za-z]{3}", c):
                continue
            if sum(ch.isdigit() for ch in c) > len(c) * 0.4:
                continue
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/benchmark/corpfam_pairs.jsonl"))
    ap.add_argument("--max-parents", type=int, default=400)
    ap.add_argument("--outdir", type=Path, default=Path("data/raw/edgar"))
    ap.add_argument("--output", type=Path,
                    default=Path("experiments/results/edgar_cross_validation.json"))
    args = ap.parse_args()

    # Families, largest first: those are the ones likely to be SEC registrants and
    # they carry the most links.
    fam: dict[str, dict] = {}
    for line in args.pairs.open(encoding="utf-8"):
        p = json.loads(line)
        if p["label"] != 1:
            continue
        f = fam.setdefault(p["right_uei"], {"parent": p["right_name"], "children": []})
        f["children"].append({"uei": p["left_uei"], "name": p["left_name"],
                              "visibility": p["visibility"]})
    ranked = sorted(fam.items(), key=lambda kv: -len(kv[1]["children"]))
    targets = ranked[:args.max_parents]
    print(f"{len(fam):,} families; probing the {len(targets)} largest", flush=True)

    ed = Edgar()
    r = ed.get(TICKERS)
    if r is None:
        print("could not fetch EDGAR company list", file=sys.stderr)
        return 1
    companies = list(json.loads(r.text).values())
    by_norm: dict[str, int] = {}
    for c in companies:
        by_norm.setdefault(norm(c["title"]), int(c["cik_str"]))
    print(f"EDGAR registrants indexed: {len(by_norm):,}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = args.outdir / "exhibit21.jsonl"
    done = set()
    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            done.add(json.loads(line)["parent_uei"])

    stats: Counter = Counter()
    fh = cache.open("a", encoding="utf-8")

    for i, (puei, info) in enumerate(targets, 1):
        if puei in done:
            continue
        pname = info["parent"]
        cik = by_norm.get(norm(pname))
        if cik is None:  # try a core-token match before giving up
            ctoks = core(pname)
            if ctoks:
                for nm, c in by_norm.items():
                    if core(nm) == ctoks:
                        cik = c
                        break
        if cik is None:
            stats["parent_not_an_sec_registrant"] += 1
            fh.write(json.dumps({"parent_uei": puei, "parent_name": pname,
                                 "cik": None, "status": "no_cik"}) + "\n")
            continue

        sub = ed.get(SUBS.format(cik=cik))
        if sub is None:
            stats["submissions_fetch_failed"] += 1
            continue
        recent = json.loads(sub.text).get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accs = recent.get("accessionNumber", [])
        acc = next((accs[j] for j, f in enumerate(forms) if f == "10-K"), None)
        if acc is None:
            stats["no_10k_on_file"] += 1
            fh.write(json.dumps({"parent_uei": puei, "parent_name": pname,
                                 "cik": cik, "status": "no_10k"}) + "\n")
            continue

        idx = ed.get(ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""))
                     + "/index.json")
        ex_url = None
        if idx is not None:
            for item in json.loads(idx.text).get("directory", {}).get("item", []):
                n = item.get("name", "").lower()
                if re.search(r"ex-?2?1|exhibit21|ex21", n) and n.endswith(
                        (".htm", ".html", ".txt")):
                    ex_url = (ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""))
                              + "/" + item["name"])
                    break
        if ex_url is None:
            stats["no_exhibit21_in_10k"] += 1
            fh.write(json.dumps({"parent_uei": puei, "parent_name": pname,
                                 "cik": cik, "status": "no_ex21"}) + "\n")
            continue

        doc = ed.get(ex_url)
        subs = parse_subsidiaries(doc.text) if doc is not None else []
        stats["exhibit21_retrieved"] += 1
        fh.write(json.dumps({"parent_uei": puei, "parent_name": pname, "cik": cik,
                             "status": "ok", "url": ex_url,
                             "n_parsed": len(subs), "subsidiaries": subs}) + "\n")
        fh.flush()
        if i % 20 == 0:
            print(f"  {i}/{len(targets)}  {dict(stats)}", flush=True)

    fh.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "families_total": len(fam),
        "parents_probed": len(targets),
        "outcomes": dict(stats),
        "cache": str(cache),
        "note": ("Retrieval pass only. Matching of declared children against parsed "
                 "Exhibit 21 subsidiary lists is done by "
                 "reconcile_edgar.py so the expensive fetch is not repeated."),
    }, indent=2), encoding="utf-8")
    print(json.dumps(dict(stats), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
