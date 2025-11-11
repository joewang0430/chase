#!/usr/bin/env python3
"""
Strip 'timestamp' and 'engine_time' keys from each JSON object line in a JSONL file.
Creates a .bak backup and performs an in-place replacement safely via a temp file.

Usage:
  python3 scripts/strip_full_book_meta.py --path botzone/data/full_book.jsonl

Defaults to the repo's botzone/data/full_book.jsonl if --path is omitted.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Tuple

STRIP_KEYS = {"timestamp", "engine_time"}

def process_line(line: str) -> Tuple[str, bool]:
    line_strip = line.strip()
    if not line_strip:
        return line, False
    try:
        obj = json.loads(line_strip)
    except json.JSONDecodeError:
        # Preserve original line if it's not valid JSON
        return line, False
    changed = False
    for k in list(STRIP_KEYS):
        if k in obj:
            obj.pop(k, None)
            changed = True
    out = json.dumps(obj, separators=(",", ":")) + "\n"
    return out, changed


def strip_file(path: str) -> Tuple[int, int]:
    """Return (total_lines, changed_lines)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    # Backup
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)

    total = 0
    changed = 0

    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".full_book_strip_", dir=dir_name, text=True)
    os.close(fd)

    try:
        with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                total += 1
                out_line, did_change = process_line(line)
                if did_change:
                    changed += 1
                fout.write(out_line)
        # Atomic-ish replace
        os.replace(tmp_path, path)
    except Exception:
        # On failure, keep original and remove temp
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            # Keep backup; user can restore manually
            pass
        raise

    return total, changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strip 'timestamp' and 'engine_time' from JSONL file.")
    parser.add_argument("--path", default=None, help="Path to JSONL file (default: botzone/data/full_book.jsonl in repo root)")
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(repo_root, "botzone", "data", "full_book.jsonl")
    path = args.path or default_path

    try:
        total, changed = strip_file(path)
        print(f"[strip] Done. file={path} lines={total} changed={changed} backup={path}.bak")
        return 0
    except Exception as e:
        print(f"[strip] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
