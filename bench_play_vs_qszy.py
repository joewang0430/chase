import sys, os, time, random, statistics
from typing import List, Tuple

# Add botzone dir for play.py
BOTZONE_DIR = os.path.join(os.path.dirname(__file__), 'botzone')
if BOTZONE_DIR not in sys.path:
    sys.path.append(BOTZONE_DIR)

# Import fast pure python engine
aimport = __import__
play_mod = aimport('play')
OthelloAI = play_mod.OthelloAI

import numpy as np

# --- QSZYLite (torch-free) ---
# Extracted core bitboard algorithm from qszy.py (move_dirs + move gen + flipping)
move_dirs = [
    (~np.uint64(0x0101010101010101), np.uint64(1),  np.uint64.__rshift__),
    (~np.uint64(0x80808080808080FF), np.uint64(7),  np.uint64.__rshift__),
    (~np.uint64(0x00000000000000FF), np.uint64(8),  np.uint64.__rshift__),
    (~np.uint64(0x01010101010101FF), np.uint64(9),  np.uint64.__rshift__),
    (~np.uint64(0x8080808080808080), np.uint64(1),  np.uint64.__lshift__),
    (~np.uint64(0xFF01010101010101), np.uint64(7),  np.uint64.__lshift__),
    (~np.uint64(0xFF00000000000000), np.uint64(8),  np.uint64.__lshift__),
    (~np.uint64(0xFF80808080808080), np.uint64(9),  np.uint64.__lshift__),
]

class QSZYLite:
    @staticmethod
    def valid_move_mask(b: int, w: int) -> int:
        bb = np.uint64(b)
        ww = np.uint64(w)
        avail = np.uint64(0)
        for mask, move, op in move_dirs:
            alive = bb
            while alive:
                alive = op((alive & mask), move) & ww
                avail |= op((alive & mask), move) & ~ww & ~bb
        return int(avail)

    @staticmethod
    def make_move(b: int, w: int, pos: int) -> Tuple[int,int]:
        # Return new (b,w) after current player (b) plays pos
        stone = np.uint64(1) << np.uint64(pos)
        if ( (b | w) & (1 << pos) ) != 0:
            return b, w
        bb = np.uint64(b)
        ww = np.uint64(w)
        flips = np.uint64(0)
        for mask, move, op in move_dirs:
            alive = op((stone & mask), move) & ww
            captured = np.uint64(0)
            while alive:
                captured |= alive
                nxt = op((alive & mask), move)
                if nxt & bb:
                    flips |= captured
                    break
                alive = nxt & ww
        new_b = b | (1 << pos) | int(flips)
        new_w = w & ~int(flips)
        return new_b, new_w

# --- Helpers ---
random.seed(2024)

def lsb_index(bb: int) -> int:
    lsb = bb & -bb
    return lsb.bit_length() - 1

def bitboard_to_moves_list(bb: int) -> List[int]:
    res = []
    while bb:
        l = bb & -bb
        pos = l.bit_length() - 1
        res.append(pos)
        bb ^= l
    return res

# Generate diverse legal midgame positions using fast engine (play.py)

def generate_positions(n: int = 6000, plies_range=(6, 30)) -> List[Tuple[int,int]]:
    eng = OthelloAI()
    positions: List[Tuple[int,int]] = []
    for _ in range(n):
        eng.init_standard_board(1)
        my, opp = eng.my_pieces, eng.opp_pieces
        for _ply in range(random.randint(*plies_range)):
            moves = eng.generate_moves_fast(my, opp)
            if moves == 0:
                opp_moves = eng.generate_moves_fast(opp, my)
                if opp_moves == 0:
                    break
                my, opp = opp, my
                continue
            mv_list = bitboard_to_moves_list(moves)
            pos = random.choice(mv_list)
            my, opp = eng.fast_make_move(my, opp, pos)
            my, opp = opp, my  # swap turn
        positions.append((my, opp))
    return positions

print("Generating positions...")
positions = generate_positions()
print(f"Generated {len(positions)} positions")

# --- Bench: move generation ---

def bench_generate_play(pos_list):
    eng = OthelloAI()
    func = eng.generate_moves_fast
    for my, opp in pos_list[:200]:
        func(my, opp)
    t0 = time.perf_counter()
    for my, opp in pos_list:
        func(my, opp)
    t1 = time.perf_counter()
    return t1 - t0

def bench_generate_qszy(pos_list):
    for my, opp in pos_list[:200]:
        QSZYLite.valid_move_mask(my, opp)
    t0 = time.perf_counter()
    for my, opp in pos_list:
        QSZYLite.valid_move_mask(my, opp)
    t1 = time.perf_counter()
    return t1 - t0

# --- Bench: make first legal move ---

def bench_make_play(pos_list):
    eng = OthelloAI()
    times = []
    for my, opp in pos_list:
        mv_bb = eng.generate_moves_fast(my, opp)
        if mv_bb == 0:
            continue
        pos = lsb_index(mv_bb)
        t0 = time.perf_counter_ns()
        eng.fast_make_move(my, opp, pos)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    if not times: return 0,0,0
    return statistics.mean(times), statistics.median(times), max(times)

def bench_make_qszy(pos_list):
    times = []
    for my, opp in pos_list:
        mv_bb = QSZYLite.valid_move_mask(my, opp)
        if mv_bb == 0:
            continue
        pos = lsb_index(mv_bb)
        t0 = time.perf_counter_ns()
        QSZYLite.make_move(my, opp, pos)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    if not times: return 0,0,0
    return statistics.mean(times), statistics.median(times), max(times)

print("Benchmarking move generation...")
play_gen = bench_generate_play(positions)
qszy_gen = bench_generate_qszy(positions)
print(f"generate_moves  play.py: {play_gen:.4f}s | qszy-lite: {qszy_gen:.4f}s (positions={len(positions)})")
if qszy_gen>0:
    print(f"  Ratio (qszy-lite / play) = {qszy_gen/play_gen:.2f}x")

print("Benchmarking make_move (first legal)...")
play_mean, play_med, play_max = bench_make_play(positions)
qszy_mean, qszy_med, qszy_max = bench_make_qszy(positions)
print("fast_make_move ns per op:")
print(f"  play.py    mean={play_mean:.1f}  median={play_med:.1f}  max={play_max:.1f}")
print(f"  qszy-lite  mean={qszy_mean:.1f}  median={qszy_med:.1f}  max={qszy_max:.1f}")
if qszy_mean>0:
    print(f"  Mean ratio (qszy-lite / play) = {qszy_mean/play_mean:.2f}x")
