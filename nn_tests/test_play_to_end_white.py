import os
import ctypes as C

# Path to the compiled shared library
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "..", "botzone", "data", "ending_mac.so")
LIB_PATH = os.path.normpath(LIB_PATH)

# Sentinel from C (INT_MIN/2)
PLAY_TO_END_SENTINEL = -1073741824

# Symbols for printing
BLACK = "●"
WHITE = "○"
EMPTY = "·"

# Mapping helpers
# bit 0 = a1, ..., bit 7 = h1, bit 8 = a2, ..., bit 63 = h8

def coord_to_bit(row_idx_0, col_idx_0):
    return 1 << (row_idx_0 * 8 + col_idx_0)


def parse_board_to_bitboards(rows):
    assert len(rows) == 8, "Need exactly 8 rows"
    black = 0
    white = 0
    for r, line in enumerate(rows):
        parts = [p for p in line.strip().split(" ") if p]
        if len(parts) != 8:
            raise ValueError(f"Row {r+1} must have 8 entries, got: {parts}")
        for c, ch in enumerate(parts):
            bit = coord_to_bit(r, c)
            if ch == BLACK:
                black |= bit
            elif ch == WHITE:
                white |= bit
            elif ch in (EMPTY, ".", "-"):
                pass
            else:
                raise ValueError(f"Unknown cell symbol '{ch}' at r={r+1}, c={c+1}")
    return black, white


def bitboards_to_rows(black, white):
    rows = []
    for r in range(8):
        cells = []
        for c in range(8):
            bit = coord_to_bit(r, c)
            if black & bit:
                cells.append(BLACK)
            elif white & bit:
                cells.append(WHITE)
            else:
                cells.append(EMPTY)
        rows.append(" ".join(cells))
    return rows


def print_board(rows):
    header = "   a b c d e f g h"
    print(header)
    for i, row in enumerate(rows, start=1):
        print(f" {i} {row}")


def load_lib():
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError(f"Shared library not found at: {LIB_PATH}\nPlease compile ending.c into ending_mac.so")
    lib = C.CDLL(LIB_PATH)
    lib.play_to_end.argtypes = [C.c_uint64, C.c_uint64, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.play_to_end.restype = C.c_int
    return lib


def main():
    # Input board (White to move next)
    #   a b c d e f g h
    # 1 · · ● · ○ · ○ ●
    # 2 · · · ● ○ ○ ● ●
    # 3 · ○ ○ ○ ● ● ● ●
    # 4 ● ○ ● ○ ● ● ● ●
    # 5 ● ○ ● ○ ○ ● ● ●
    # 6 ● ○ ● ○ ● ● ● ●
    # 7 ● ● ● ● ● ● ● ●
    # 8 ● ● ● ● ● ● ● ●
    input_rows = [
        "· · ● · ○ · ○ ●",
        "· · · ● ○ ○ ● ●",
        "· ○ ○ ○ ● ● ● ●",
        "● ○ ● ○ ● ● ● ●",
        "● ○ ● ○ ○ ● ● ●",
        "● ○ ● ○ ● ● ● ●",
        "● ● ● ● ● ● ● ●",
        "● ● ● ● ● ● ● ●",
    ]

    black, white = parse_board_to_bitboards(input_rows)

    # White to move -> my = white, opp = black
    my = C.c_uint64(white)
    opp = C.c_uint64(black)
    out_my = C.c_uint64(0)
    out_opp = C.c_uint64(0)

    lib = load_lib()

    res = lib.play_to_end(my, opp, C.byref(out_my), C.byref(out_opp))

    print("Initial board (White to move):")
    print_board(input_rows)
    print()

    if res == PLAY_TO_END_SENTINEL:
        print("play_to_end skipped (likely empties > 8).")
        return

    final_my = out_my.value   # White final
    final_opp = out_opp.value # Black final

    final_black = final_opp
    final_white = final_my

    final_rows = bitboards_to_rows(final_black, final_white)

    b_cnt = bin(final_black).count("1")
    w_cnt = bin(final_white).count("1")

    print("Final board after play_to_end (White to move at start):")
    print_board(final_rows)
    print()
    outcome = "Black win" if b_cnt > w_cnt else ("White win" if w_cnt > b_cnt else "Draw")
    print(f"Counts -> Black: {b_cnt}, White: {w_cnt}  =>  {outcome}")


if __name__ == "__main__":
    main()
