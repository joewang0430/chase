# 这个文件有点问题，黑白有的时候只能当做黑，不过也差不多能用。

#!/usr/bin/env python3
"""
Quick randomness check for c/eva_noise_random.c using ctypes.

Given a human-readable board (ASCII with ●/○/· and a/b/c headers),
this script:
- converts it to bitboards via utils.converters.Converters.visual_to_bb,
- compiles and loads the C library if needed,
- calls choose_move(my_bb, opp_bb) N times,
- tallies how many times each legal move is chosen.

Usage examples:
  python3 -m test.test_eva_random            # use built-in demo board, 100 samples
  python3 test/test_eva_random.py --n 200
  python3 test/test_eva_random.py --board-file /path/to/board.txt --n 500

Notes:
- On macOS this builds a .dylib next to the C source the first time.
- Requires clang (or gcc) installed locally.
"""

import os
import sys
import subprocess
import ctypes
import argparse
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.converters import Converters


# DEMO_BOARD = """
#   a b c d e f g h
# 1 · · · · · · · ·
# 2 · · · · · · · ·
# 3 · · ○ · ● · · ·
# 4 · · · ○ ● · · ·
# 5 · · ● ● ○ · · ·
# 6 · · · ○ ○ ○ · ·
# 7 · · · · · · · ·
# 8 · · · · · · · ·
# """.strip("\n")

# DEMO_BOARD = """
#    a b c d e f g h
#  1 · · · · · · · ·
#  2 · · · · · · · ·
#  3 · · · · · · · ·
#  4 · · ○ ○ ○ · · ·
#  5 · · · ○ ● · · ·
#  6 · · · · · · · ·
#  7 · · · · · · · ·
#  8 · · · · · · · ·
# """.strip("\n")

# DEMO_BOARD = """
#    a b c d e f g h
#  1 · · · · · · · ·
#  2 · · · · · · · ·
#  3 · · ○ ○ ○ · · ·
#  4 · · ● ● ○ · · ·
#  5 · · · ● ○ · · ·
#  6 · · · · · · · ·
#  7 · · · · · · · ·
#  8 · · · · · · · ·
# """.strip("\n")

DEMO_BOARD = """
   a b c d e f g h
 1 ○ · · ○ ○ ○ ○ ·
 2 ○ ○ ○ ● ○ ○ · ·
 3 ○ ○ ○ ○ ○ ○ · ·
 4 ○ ● ● ● ○ ● · ·
 5 ● ● ○ ○ ● ● · ·
 6 · ● · ○ ● ● · ·
 7 · · ● ○ ● ● · ·
 8 · · ○ ○ ○ · · ·
""".strip("\n")

def compile_lib(src: str, out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cc_candidates = ["clang", "gcc"]
    is_darwin = sys.platform == "darwin"
    for cc in cc_candidates:
        try:
            # Prefer -dynamiclib on macOS; fallback to -shared elsewhere
            if is_darwin:
                cmd = [cc, "-O3", "-std=c11", "-fPIC", "-dynamiclib", "-o", out, src]
            else:
                cmd = [cc, "-O3", "-std=c11", "-fPIC", "-shared", "-o", out, src]
            subprocess.run(cmd, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Failed to compile C library; please install clang or gcc.")


def ensure_lib() -> str:
    src = os.path.join(ROOT, "c", "eva_noise_random.c")
    # Put the built library alongside the source for simplicity
    if sys.platform == "darwin":
        lib = os.path.join(ROOT, "c", "eva_noise_random.dylib")
    else:
        lib = os.path.join(ROOT, "c", "eva_noise_random.so")

    # Build if missing or stale
    need_build = (not os.path.exists(lib)) or (
        os.path.getmtime(lib) < os.path.getmtime(src)
    )
    if need_build:
        compile_lib(src, lib)
    return lib


def bit_scan(mask: int):
    """Yield indices (0..63) of set bits from LSB to MSB."""
    while mask:
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        yield idx
        mask ^= lsb


def main():
    ap = argparse.ArgumentParser(description="Test randomness of eva_noise_random.c choose_move")
    ap.add_argument("--n", type=int, default=100, help="number of samples (calls)")
    ap.add_argument("--board-file", type=str, default=None, help="path to a text board file; if omitted, use built-in demo")
    args = ap.parse_args()

    # Read board text
    if args.board_file:
        with open(args.board_file, "r", encoding="utf-8") as f:
            board_text = f.read()
    else:
        board_text = DEMO_BOARD

    my_bb, opp_bb = Converters.visual_to_bb(board_text)

    # Load library
    lib_path = ensure_lib()
    lib = ctypes.CDLL(lib_path)

    # C functions
    # int choose_move(uint64_t my_pieces, uint64_t opp_pieces)
    lib.choose_move.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    lib.choose_move.restype = ctypes.c_int

    # u64 generate_moves(Board board)
    class CBoard(ctypes.Structure):
        _fields_ = [("board", ctypes.c_uint64 * 2)]

    lib.generate_moves.argtypes = (CBoard,)
    lib.generate_moves.restype = ctypes.c_uint64

    # Compute legal moves via C for consistency
    b = CBoard()
    b.board[0] = ctypes.c_uint64(my_bb)
    b.board[1] = ctypes.c_uint64(opp_bb)
    legal_mask = int(lib.generate_moves(b))

    if legal_mask == 0:
        print("No legal moves (PASS). Nothing to sample.")
        return

    legal_coords = [Converters.dmv_to_mv(i) for i in bit_scan(legal_mask)]
    idx_to_coord = {i: Converters.dmv_to_mv(i) for i in range(64)}

    # Tally selections
    counts: Dict[str, int] = {coord: 0 for coord in legal_coords}
    total = 0
    for _ in range(max(1, args.n)):
        pos = int(lib.choose_move(ctypes.c_uint64(my_bb), ctypes.c_uint64(opp_bb)))
        if pos < 0 or pos > 63:
            # treat as PASS or invalid; skip tally
            continue
        if (legal_mask >> pos) & 1:
            coord = idx_to_coord[pos]
            counts[coord] = counts.get(coord, 0) + 1
            total += 1
        else:
            # Chosen move not in legal set (should not happen); record under "<illegal>"
            counts["<illegal>"] = counts.get("<illegal>", 0) + 1

    # Output summary
    print("Board:\n" + Converters.bb_to_visual(my_bb, opp_bb))
    print("\nLegal moves:", ", ".join(sorted(legal_coords)))
    print(f"Samples: {args.n}, tallied: {total}")
    print("\nCounts (legal moves only):")
    for coord in sorted(legal_coords):
        print(f"  {coord}: {counts.get(coord, 0)}")
    if "<illegal>" in counts:
        print(f"  <illegal>: {counts['<illegal>']}")


if __name__ == "__main__":
    main()
