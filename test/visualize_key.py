# python test/visualize_key.py 0000000810000000_0000001008000000_B
# python test/visualize_key.py
#!/usr/bin/env python3
"""Interactive helper to visualize a book/position key.

Usage:
  python test/visualize_key.py               # enters interactive loop
  python test/visualize_key.py <key_string>  # one-shot print

Key format expected (current project schema):
  <16 hex for black>_<16 hex for white>_<side>
Example:
  0000000810000000_0000001008000000_B

Type 'exit' or empty line to quit in interactive mode.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure project root on sys.path so we can import utils.converters
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.converters import Converters
except Exception as e:  # pragma: no cover
    print(f"[ERROR] Failed to import Converters: {e}")
    sys.exit(1)


def show(key: str):
    board = Converters.key_to_visual(key)
    print(board)


def interactive():
    print("Enter key (or 'exit' to quit):")
    while True:
        try:
            key = input('key> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if not key or key.lower() in {'exit', 'quit'}:
            break
        show(key)
        print('-' * 40)


def main():
    if len(sys.argv) > 1:
        show(sys.argv[1])
    else:
        interactive()

if __name__ == '__main__':
    main()
