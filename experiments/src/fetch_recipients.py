"""Fetch the USAspending recipient roster.

The roster is the spine of this benchmark. Each entry carries a UEI, a display
name, a spend total, and a recipient_level of P (parent), C (child) or R
(neither). The P/C structure is the corporate-family ground truth: it comes from
SAM.gov entity registration, not from anything we inferred.

Writes newline-delimited JSON so a partial run is still usable, plus a manifest
recording exactly what was retrieved and when.

Usage:
    python experiments/src/fetch_recipients.py [--award-type contracts]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://api.usaspending.gov/api/v2/recipient/"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "usaspending")
RESULTS = os.path.join(ROOT, "experiments", "results")

PAGE_LIMIT = 1000
MAX_RETRIES = 5


def fetch_page(session: requests.Session, page: int, award_type: str) -> dict:
    payload = {
        "order": "desc",
        "sort": "amount",
        "page": page,
        "limit": PAGE_LIMIT,
        "award_type": award_type,
    }
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(API, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  page {page} attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"page {page} failed after {MAX_RETRIES} attempts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--award-type", default="contracts",
                    choices=["contracts", "grants", "direct_payments", "loans",
                             "other_financial_assistance", "all"])
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RAW, f"recipients_{args.award_type}.jsonl")

    session = requests.Session()
    session.headers.update({"User-Agent": "supplier-resolution-benchmark/0.1"})

    started = datetime.now(timezone.utc).isoformat()
    first = fetch_page(session, 1, args.award_type)
    total = first["page_metadata"]["total"]
    pages = (total + PAGE_LIMIT - 1) // PAGE_LIMIT
    print(f"total recipients: {total:,} across {pages} pages", flush=True)

    seen_ids: set[str] = set()
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        page_data = first
        for page in range(1, pages + 1):
            if page > 1:
                page_data = fetch_page(session, page, args.award_type)
            rows = page_data.get("results", [])
            if not rows:
                print(f"  page {page} returned no rows; stopping", flush=True)
                break
            for row in rows:
                rid = row.get("id")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if page % 20 == 0 or page == pages:
                print(f"  page {page}/{pages}  written {written:,}", flush=True)
            time.sleep(0.15)

    with open(out_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    manifest = {
        "source": API,
        "award_type": args.award_type,
        "api_reported_total": total,
        "pages_requested": pages,
        "page_limit": PAGE_LIMIT,
        "rows_written": written,
        # The API's own total minus what we kept. Do not compute this from
        # pages * PAGE_LIMIT: the final page is short, so that conflates unfilled
        # page slots with genuine duplicates and overstates it badly.
        "duplicate_ids_skipped": max(total - written, 0),
        "output": os.path.relpath(out_path, ROOT),
        "sha256": digest,
        "bytes": os.path.getsize(out_path),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    mpath = os.path.join(RESULTS, f"fetch_recipients_{args.award_type}.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {written:,} recipients -> {out_path}")
    print(f"sha256 {digest}")
    print(f"manifest -> {mpath}")

    if written < total * 0.95:
        print(f"WARNING: wrote {written:,} but API reported {total:,}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
