"""Fetch per-entity detail for every distinct UEI in the roster.

This is where the actual ground truth lives. The roster only says whether a record
is parent-level, child-level or neither; it does not say *whose* child. The detail
endpoint supplies:

  parent_uei / parent_name  -> the corporate-family link
  alternate_names           -> prior and variant legal names for the same entity
  location                  -> a blocking and disambiguation signal
  business_types            -> entity characteristics

Two things established by inspection before writing this, both of which shape the
design:

1. The -P and -C detail records for one UEI return *identical* parent links and
   alternate names. They differ only in total_transaction_amount, where -P is the
   family rollup and -C is the entity standing alone. So one call per distinct UEI
   is sufficient; calling both would double the load for nothing.

2. USAspending makes an entity its own parent when it has no real parent. Lockheed
   Martin's parent_uei is Lockheed Martin's own UEI. A record is therefore a genuine
   child only when parent_uei != uei, and any analysis that skips that test will
   report ~80,000 spurious families.

Resumable: reruns skip UEIs already present in the output file, so an interrupted
run costs nothing.

Usage:
    python experiments/src/fetch_entity_detail.py [--workers 4] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROSTER = os.path.join(ROOT, "data", "raw", "usaspending", "recipients_contracts.jsonl")
OUT = os.path.join(ROOT, "data", "raw", "usaspending", "entity_detail.jsonl")
MANIFEST = os.path.join(ROOT, "experiments", "results", "fetch_entity_detail.json")

API = "https://api.usaspending.gov/api/v2/recipient/{rid}/"
MAX_RETRIES = 5
SHUFFLE_SEED = 20260827

KEEP = ("recipient_id", "uei", "duns", "name", "recipient_level", "parent_id",
        "parent_uei", "parent_name", "parent_duns", "parents", "alternate_names",
        "business_types", "location", "total_transaction_amount", "total_transactions")

_lock = threading.Lock()
_state = {"done": 0, "failed": 0, "started": time.time()}


def load_targets() -> list[tuple[str, str]]:
    """One (uei, recipient_id) per distinct UEI, preferring the -C record.

    The C record is the entity in its own right rather than the family rollup. The
    parent link and alternate names are the same either way, but preferring C keeps
    total_transaction_amount interpretable as the entity's own spend.

    Order is a deterministic shuffle, not a sort. Fetching in UEI order means an
    interrupted run leaves a single alphabetical slice: the first 4,806 records
    collected this way were *all* C-prefixed, 2.6% of the roster, and their median
    obligation sat above the 97.5th percentile of same-sized random draws. Shuffling
    on a fixed seed makes every prefix of the run a uniform random sample, so partial
    results are valid on their own and the order still reproduces exactly.
    """
    best: dict[str, str] = {}
    with open(ROSTER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uei, rid, lvl = r.get("uei"), r.get("id"), r.get("recipient_level")
            if not uei or not rid:
                continue
            if uei not in best or lvl == "C":
                best[uei] = rid
    targets = sorted(best.items())
    random.Random(SHUFFLE_SEED).shuffle(targets)
    return targets


def already_done() -> set[str]:
    if not os.path.exists(OUT):
        return set()
    seen = set()
    with open(OUT, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["uei"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def fetch_one(session: requests.Session, uei: str, rid: str, fh, delay: float) -> None:
    url = API.format(rid=rid)
    backoff = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 200:
                d = resp.json()
                rec = {k: d.get(k) for k in KEEP}
                rec["uei"] = rec.get("uei") or uei
                rec["_requested_id"] = rid
                # Records written before the shuffle fix lack this field, which is
                # how the biased alphabetical slice stays separable from the uniform
                # sample when computing population estimates.
                rec["_sample"] = "shuffled"
                with _lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    _state["done"] += 1
                    if _state["done"] % 2000 == 0:
                        fh.flush()
                        el = time.time() - _state["started"]
                        rate = _state["done"] / el if el else 0
                        print(f"  {_state['done']:,} done  {_state['failed']} failed  "
                              f"{rate:.1f}/s  {el / 60:.1f} min elapsed", flush=True)
                time.sleep(delay)
                return
            if resp.status_code == 404:
                with _lock:
                    _state["failed"] += 1
                return
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            with _lock:
                _state["failed"] += 1
            return
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                with _lock:
                    _state["failed"] += 1
                return
            time.sleep(backoff)
            backoff *= 2
    with _lock:
        _state["failed"] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.15,
                    help="per-worker pause between requests; keeps load polite")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

    targets = load_targets()
    done = already_done()
    todo = [(u, r) for u, r in targets if u not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"distinct UEIs in roster: {len(targets):,}", flush=True)
    print(f"already fetched:         {len(done):,}", flush=True)
    print(f"to fetch this run:       {len(todo):,}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    est = len(todo) * args.delay / max(args.workers, 1) / 60
    print(f"estimated wall time:     ~{est:.0f} min at {args.workers} workers\n",
          flush=True)

    started = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": "corpfam-benchmark/0.1"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers * 2,
                                            pool_maxsize=args.workers * 2)
    session.mount("https://", adapter)

    with open(OUT, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for uei, rid in todo:
                pool.submit(fetch_one, session, uei, rid, fh, args.delay)

    total_lines = sum(1 for line in open(OUT, encoding="utf-8") if line.strip())
    with open(OUT, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    manifest = {
        "source": API,
        "distinct_uei_in_roster": len(targets),
        "fetched_this_run": _state["done"],
        "failed_this_run": _state["failed"],
        "total_records_in_file": total_lines,
        "coverage_pct": round(total_lines / len(targets) * 100, 4),
        "workers": args.workers,
        "per_worker_delay_s": args.delay,
        "output": os.path.relpath(OUT, ROOT),
        "sha256": digest,
        "bytes": os.path.getsize(OUT),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nfetched {_state['done']:,} this run, {_state['failed']} failed")
    print(f"file now holds {total_lines:,} records "
          f"({manifest['coverage_pct']:.2f}% of distinct UEIs)")
    print(f"manifest -> {os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
