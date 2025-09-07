#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare play_to_end vs iterative solve_endgame rollout on a given position.

Board assumed: BLACK = 'B' (●), WHITE = 'W' (○), empty = '.'
Starting side: Black to move (same assumption as play_to_end).

Current test board (user supplied, appears to have 0 empties though stated "8 empties"):
   a b c d e f g h
 1 ○ ○ ○ ○ ○ ○ ○ ○
 2 ● ● ● ○ ● ● ● ○
 3 ○ ● ○ ● ● ● ● ○
 4 ○ ● ● ● ● ○ ● ○
 5 ○ ● ● ● ● ● ● ○
 6 ○ ● ○ ● ● ● ● ○
 7 ○ ● ● ● ● ● ● ○
 8 ○ ● ○ ○ ○ ○ ○ ○

If you want to test a true 8-empties position, edit TEST_ROWS below with '.' characters.
"""
import os, ctypes, time, subprocess
from ctypes import c_uint64, c_int, byref, POINTER

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'botzone', 'data', 'ending.c')
SO_PATH = os.path.join(ROOT, 'botzone', 'data', 'ending_mac.so')

TEST_ROWS = [
    'WWWWWWWW',
    'BBBWBBBW',  # NOTE: user row had 3 blacks then W then B B B W; keep as given? Adjusted to match pattern? Using original: ● ● ● ○ ● ● ● ○ -> BBBWBBBW
    'WBWBBBBW',
    'W WBBB W'.replace(' ', 'W').replace('BWW', 'BBB'),  # placeholder fix not used; will override next
    'W W W W W'.replace(' ', ''),  # placeholder
    # Actually use the exact rows from earlier final board to avoid confusion:
]
# Overwrite with exact board (final board from earlier test) to ensure correctness.
TEST_ROWS = [
    'WWWWWWWW',
    'BBBWBBBW',  # row2
    'WBWBBBBW',  # row3
    'W WBBBWBW'.replace(' ', ''),  # row4 -> W WBBBWBW => WWBBBWBW
    'WWWBBBWW',  # row5
    'WBWBWBW?'.replace('?', '').replace('WBWBWBW', '.WWWBWBW'),  # not relevant, we will just use earlier final board rows next
]
# Simpler: directly hardcode final board from previous third test (no empties)
TEST_ROWS = [
    'WWWWWWWW',  # 1
    'BBBWBBBW',  # 2 (● ● ● ○ ● ● ● ○)
    'WBWBBBBW',  # 3
    'W WBBBWBW'.replace(' ', ''),  # 4 -> WWBBBWBW
    'WWWBBBWW',  # 5
    'W W B W B W'.replace(' ', '').replace('WWBWBW', 'W?'),  # messy; just override again below
]
# Final clean version (exact earlier board):
TEST_ROWS = [
    'WWWWWWWW',
    'BBBWBBBW',
    'WBWBBBBW',
    'WWBBBWBW',
    'WWWBBBWW',
    'W?'.replace('?', ''),
]
# This got messy; restart with explicit target board (no empties):
TEST_ROWS = [
    'WWWWWWWW',  # 1
    'BBBWBBBW',  # 2
    'WBWBBBBW',  # 3
    'WWBBBWBW',  # 4
    'WWWBBBWW',  # 5
    'W?'.replace('?', ''),
    'PLACEHOLDER',
    'PLACEHOLDER',
]
# Abort above; use the previously logged third final board EXACT rows:
FINAL_BOARD_ROWS = [
    'WWWWWWWW',
    'BBBWBBBW',
    'WBWBBBBW',
    'WBBBBWBW'.replace('BBBBB', '●'),  # This has gone too complex.
]
# Instead of over-complicating, we re-derive directly from earlier textual final board:
# Third Final Board (from logs):
# 1 ○ ○ ○ ○ ○ ○ ○ ○ -> WWWWWWWW
# 2 ● ● ● ○ ● ● ● ○ -> BBBWBBBW
# 3 ○ ● ○ ● ● ● ● ○ -> WBWBBBBW
# 4 ○ ● ● ● ● ○ ● ○ -> WBBBBW BW? -> Convert manually: WBBBB? Actually: white, black, black, black, black, white, black, white
#    => WBBBBWBW
# 5 ○ ● ● ● ● ● ● ○ -> WBBBBBBW
# 6 ○ ● ○ ● ● ● ● ○ -> WBWBBBBW
# 7 ○ ● ● ● ● ● ● ○ -> WBBBBBBW
# 8 ○ ● ○ ○ ○ ○ ○ ○ -> WBWWWWWW
TEST_ROWS = [
    'WWWWWWWW',    # 1
    'BBBWBBBW',    # 2
    'WBWBBBBW',    # 3
    'WBBBBWBW',    # 4
    'WBBBBBBW',    # 5
    'WBWBBBBW',    # 6
    'WBBBBBBW',    # 7
    'WBWWWWWW',    # 8
]

# Utility: ensure all rows length 8
for r in TEST_ROWS:
    if len(r) != 8:
        raise SystemExit('Bad row length: ' + r)

POP8 = [bin(i).count('1') for i in range(256)]

def popcount(x: int) -> int:
    s = 0
    while x:
        s += POP8[x & 0xFF]
        x >>= 8
    return s

def build_bitboards(rows):
    black = white = 0
    empties = 0
    for r in range(8):
        for c in range(8):
            ch = rows[r][c]
            idx = r * 8 + c
            if ch in ('B', '●'):
                black |= 1 << idx
            elif ch in ('W', '○'):
                white |= 1 << idx
            else:
                empties += 1
    return black, white, empties

BLACK_STARTS = True  # consistent with play_to_end assumption

# We need move generation & make_move identical to play.py for iterative solve path
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
NOT_FILE_A = 0xFFFFFFFFFFFFFFFF ^ FILE_A
NOT_FILE_H = 0xFFFFFFFFFFFFFFFF ^ FILE_H
FULL_MASK  = 0xFFFFFFFFFFFFFFFF

def shift_e(bb): return (bb & NOT_FILE_H) << 1 & FULL_MASK
def shift_w(bb): return (bb & NOT_FILE_A) >> 1
def shift_n(bb): return bb >> 8
def shift_s(bb): return (bb << 8) & FULL_MASK
def shift_ne(bb): return (bb & NOT_FILE_H) >> 7
def shift_nw(bb): return (bb & NOT_FILE_A) >> 9
def shift_se(bb): return ((bb & NOT_FILE_H) << 9) & FULL_MASK
def shift_sw(bb): return ((bb & NOT_FILE_A) << 7) & FULL_MASK

DIRS = [shift_e, shift_w, shift_n, shift_s, shift_ne, shift_nw, shift_se, shift_sw]

def generate_moves(me, opp):
    empty = ~(me | opp) & FULL_MASK
    moves = 0
    for fn in DIRS:
        x = fn(me) & opp
        acc = 0
        while x:
            acc |= x
            x = fn(x) & opp
        moves |= fn(acc) & empty
    return moves

def make_move(me, opp, pos):
    if pos < 0:
        return opp, me  # pass swap perspective
    move_bit = 1 << pos
    if (me | opp) & move_bit:
        return me, opp  # occupied
    flips = 0
    for fn in DIRS:
        cur = fn(move_bit)
        line = 0
        while cur and (cur & opp):
            line |= cur
            cur = fn(cur)
        if cur & me:
            flips |= line
    if flips == 0:
        return me, opp  # illegal for safety
    me |= move_bit | flips
    opp &= ~flips
    return me, opp

def compile_so():
    if not os.path.exists(SO_PATH) or os.path.getmtime(SO_PATH) < os.path.getmtime(SRC):
        print('[BUILD] Compiling ending.c -> ending_mac.so')
        subprocess.check_call(['clang','-O3','-shared','-fPIC',SRC,'-o',SO_PATH])

compile_so()
lib = ctypes.CDLL(SO_PATH)
# C function prototypes
lib.solve_endgame.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
lib.solve_endgame.restype  = c_int
lib.play_to_end.argtypes   = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
lib.play_to_end.restype    = c_int

black0, white0, empties0 = build_bitboards(TEST_ROWS)
print(f'[INIT] Empties={empties0} (note: user claimed 8; computed {empties0})')

# 1. Direct play_to_end
fb = c_uint64()
fw = c_uint64()
start = time.perf_counter()
res_play = lib.play_to_end(black0, white0, byref(fb), byref(fw))
t_play_ms = (time.perf_counter() - start)*1000
final_black_play = fb.value
final_white_play = fw.value

# 2. Iterative solve_endgame rollout
my = black0
opp = white0
board0_is_black = True
passes = 0
start2 = time.perf_counter()
while passes < 2:
    empties = 64 - popcount(my | opp)
    if empties == 0:
        # Need to check if current side has move actually; if no empties board full end.
        # Simulate no move for both
        passes += 1
        # Swap once to reach passes==2
        board0_is_black = not board0_is_black
        my, opp = opp, my
        break
    best_c = c_int(-1)
    _score = lib.solve_endgame(my, opp, byref(best_c))
    mv = best_c.value
    if mv < 0: # PASS
        passes += 1
        my, opp = opp, my
        board0_is_black = not board0_is_black
        if passes == 2:
            break
        continue
    passes = 0
    my, opp = make_move(my, opp, mv)
    # after move opponent to move -> swap
    my, opp = opp, my
    board0_is_black = not board0_is_black

final_black_iter = my if board0_is_black else opp
final_white_iter = opp if board0_is_black else my

t_iter_ms = (time.perf_counter() - start2)*1000

# Helper to print board
def print_board(bbits, wbits, title):
    print(title)
    print('   a b c d e f g h')
    for r in range(8):
        row = []
        for c in range(8):
            idx = r*8 + c
            m = 1<<idx
            if bbits & m: row.append('●')
            elif wbits & m: row.append('○')
            else: row.append('·')
        print(f' {r+1} ' + ' '.join(row))

print_board(final_black_play, final_white_play, '\n[Final Board via play_to_end]')
print_board(final_black_iter, final_white_iter, '\n[Final Board via iterative solve_endgame]')

same = (final_black_play == final_black_iter) and (final_white_play == final_white_iter)
print(f"\n[COMPARE] Boards identical: {same}")
print(f"play_to_end result flag={res_play}  (1=Black win,-1=White win,0=Draw,sentinel else) ")
print(f"play_to_end time: {t_play_ms:.3f} ms  iterative time: {t_iter_ms:.3f} ms")
print(f"Final counts (play_to_end): B={popcount(final_black_play)} W={popcount(final_white_play)}")
print(f"Final counts (iterative) : B={popcount(final_black_iter)} W={popcount(final_white_iter)}")

if empties0 == 0:
    print('[NOTE] Initial position already full; both methods should trivially match.')
