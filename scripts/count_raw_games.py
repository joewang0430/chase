#!/usr/bin/env python3
# python3 scripts/count_raw_games.py --root dataset/raw
from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Dict, Iterable, Iterator, Tuple


def _iter_jsonl_paths(root: str) -> Iterator[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                yield os.path.join(dirpath, name)


def _open_text_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _safe_int(x, default: int | None = None) -> int | None:
    try:
        return int(x)
    except Exception:
        return default


def count_complete_games(raw_root: str) -> Tuple[int, int, int, int]:
    """Return (complete_games, total_game_ids_seen, lines_read, files_read).

    A game is counted as complete if its game_id has at least one record with pcs==12
    and at least one record with pcs==53.
    """
    # bit flags: 1 => saw pcs=12, 2 => saw pcs=53
    flags: Dict[str, int] = {}

    lines_read = 0
    files_read = 0

    for path in _iter_jsonl_paths(raw_root):
        files_read += 1
        try:
            with _open_text_maybe_gz(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines_read += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # skip malformed lines
                        continue
                    gid = obj.get("game_id")
                    if not gid:
                        continue
                    pcs = _safe_int(obj.get("pcs"), default=None)
                    if pcs is None:
                        continue
                    cur = flags.get(gid, 0)
                    if pcs == 12:
                        cur |= 1
                    elif pcs == 53:
                        cur |= 2
                    if cur:
                        flags[gid] = cur
        except Exception:
            # unreadable file: skip
            continue

    complete = sum(1 for v in flags.values() if (v & 1) and (v & 2))
    return complete, len(flags), lines_read, files_read


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Count complete games under dataset/raw by detecting pcs=12..53 per game_id"
    )
    ap.add_argument(
        "--root",
        type=str,
        default=os.path.join("dataset", "raw"),
        help="raw dataset root (default: dataset/raw)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    raw_root = os.path.abspath(args.root)
    if not os.path.isdir(raw_root):
        print(f"[ERROR] not a directory: {raw_root}")
        return 2

    complete, total_ids, lines_read, files_read = count_complete_games(raw_root)
    print(f"raw_root: {raw_root}")
    print(f"files_read: {files_read}")
    print(f"lines_read: {lines_read}")
    print(f"game_ids_seen: {total_ids}")
    print(f"complete_games(pcs=12 & pcs=53): {complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
