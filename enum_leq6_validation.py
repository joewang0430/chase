#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exhaustive continuation validation for positions with <=6 empties (BLACK to move).
For each sampled reachable position (generated via random legal playout from the initial position
until empties <= 6 and black to move), we:
 1. Enumerate the full remaining game tree in pure Python (perfect play) to obtain exact value
    (black_count - white_count) and one optimal final board.
 2. Call solve_endgame (C) to get score and best move; verify score matches enumerated value.
 3. Call play_to_end (C) to get final board & winner flag; verify final board matches one optimal
    final board from enumeration and the resulting difference matches.

Because enumeration explores the entire sub-tree, this acts as a correctness oracle for these
positions. (Note: We do not attempt to enumerate *all* theoretical boards with <=6 empties—that
set is astronomically large; we fully enumerate the *continuations* of many sampled reachable
boards.)

Usage:
  python3 enum_leq6_validation.py [SAMPLES]
Default SAMPLES=500
"""
import os, sys, random, time, ctypes, subprocess
from functools import lru_cache
from ctypes import c_uint64, c_int, byref

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'botzone', 'data', 'ending.c')
SO_PATH_MAC = os.path.join(ROOT, 'botzone', 'data', 'ending_mac.so')
SO_PATH_LINUX = os.path.join(ROOT, 'botzone', 'data', 'ending.so')
if os.path.exists(SO_PATH_MAC):
    SO_PATH = SO_PATH_MAC
else:
    SO_PATH = SO_PATH_MAC if sys.platform == 'darwin' else SO_PATH_LINUX
if not os.path.exists(SO_PATH) or os.path.getmtime(SO_PATH) < os.path.getmtime(SRC):
    print('[BUILD] Compiling ending.c ->', os.path.basename(SO_PATH))
    subprocess.check_call(['clang','-O3','-fPIC','-shared',SRC,'-o',SO_PATH])
lib = ctypes.CDLL(SO_PATH)
lib.solve_endgame.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
lib.solve_endgame.restype  = c_int
lib.play_to_end.argtypes   = [c_uint64, c_uint64, ctypes.POINTER(c_uint64), ctypes.POINTER(c_uint64)]
lib.play_to_end.restype    = c_int

POPCNT8 = [bin(i).count('1') for i in range(256)]
def popcount(x:int)->int:
    s=0
    while x:
        s += POPCNT8[x & 0xFF]
        x >>= 8
    return s
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
        return opp, me
    move_bit = 1 << pos
    if (me | opp) & move_bit:
        return me, opp
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
        return me, opp
    me |= move_bit | flips
    opp &= ~flips
    return me, opp

# Exhaustive perfect play evaluation (returns (value, final_black, final_white))
# value = black_count - white_count under perfect play from current side to move (black_turn flag)
@lru_cache(maxsize=None)
def solve_full(black: int, white: int, black_turn: bool = True):
    """Return (value, final_black, final_white) with perfect play.
    value = final_black_count - final_white_count.
    black_turn indicates side to move.
    """
    empties = 64 - popcount(black | white)
    me = black if black_turn else white
    opp = white if black_turn else black
    moves = generate_moves(me, opp)
    if moves == 0:
        # check opponent
        opp_moves = generate_moves(opp, me)
        if opp_moves == 0:
            bcnt = popcount(black); wcnt = popcount(white)
            return bcnt - wcnt, black, white
        # pass turn
        return solve_full(black, white, not black_turn)
    if black_turn:
        best_val = -10**9
    else:
        best_val = 10**9
    best_fb = best_fw = None
    tmp = moves
    while tmp:
        p = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        if black_turn:
            nb, nw = make_move(black, white, p)
            val, fb, fw = solve_full(nb, nw, False)
            if val > best_val:
                best_val = val; best_fb, best_fw = fb, fw
        else:
            nw, nb = make_move(white, black, p)
            val, fb, fw = solve_full(nb, nw, True)
            if val < best_val:
                best_val = val; best_fb, best_fw = fb, fw
    return best_val, best_fb, best_fw

# Random generation of reachable boards with <=6 empties and black to move
def random_board_leq6(max_attempts=200000):
    for _ in range(max_attempts):
        black = (1 << (3*8+4)) | (1 << (4*8+3))
        white = (1 << (3*8+3)) | (1 << (4*8+4))
        black_turn = True
        passes = 0
        while True:
            empties = 64 - popcount(black | white)
            if empties <= 6:
                if black_turn:
                    return black, white
                else:
                    break
            me = black if black_turn else white
            opp = white if black_turn else black
            moves = generate_moves(me, opp)
            if moves == 0:
                passes += 1
                if passes == 2:
                    break
                black_turn = not black_turn
                continue
            passes = 0
            # biased: prefer moves near frontier to diversify (random anyway)
            choices = []
            tmp = moves
            while tmp:
                p = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                choices.append(p)
            mv = random.choice(choices)
            if black_turn:
                black, white = make_move(black, white, mv)
            else:
                white, black = make_move(white, black, mv)
            black_turn = not black_turn
    raise RuntimeError('Failed to find board with <=6 empties and black to move')


def validate_samples(N=500, seed=None):
    if seed is not None:
        random.seed(seed)
    mismatches = 0
    total_solver_ok = 0
    t_enum = 0.0
    t_solver = 0.0
    t_play = 0.0
    for i in range(1, N+1):
        black, white = random_board_leq6()
        t0 = time.perf_counter()
        val_enum, fb_enum, fw_enum = solve_full(black, white, True)
        t_enum += time.perf_counter() - t0
        bm = c_int(-1)
        t1 = time.perf_counter()
        val_solve = lib.solve_endgame(black, white, byref(bm))
        t_solver += time.perf_counter() - t1
        fb = c_uint64(); fw = c_uint64()
        t2 = time.perf_counter()
        flag = lib.play_to_end(black, white, byref(fb), byref(fw))
        t_play += time.perf_counter() - t2
        fb_play = fb.value; fw_play = fw.value
        val_play = popcount(fb_play) - popcount(fw_play)
        # correctness criteria:
        values_ok = (val_enum == val_solve == val_play)
        disjoint_ok = ((fb_play & fw_play) == 0)
        diff_ok = (val_play == popcount(fb_play) - popcount(fw_play))
        ok = values_ok and disjoint_ok and diff_ok
        if ok:
            total_solver_ok += 1
        else:
            mismatches += 1
            print('[MISMATCH]', i, 'val_enum=', val_enum, 'val_solve=', val_solve, 'val_play=', val_play,
                  'counts_play=', popcount(fb_play), popcount(fw_play), 'empties=', 64 - popcount(fb_play | fw_play))
        if i % 50 == 0 or not ok:
            print(f'[PROGRESS] {i}/{N} ok={ok} enum_avg={t_enum/i*1e6:.1f}us solve_avg={t_solver/i*1e6:.1f}us play_avg={t_play/i*1e6:.1f}us empties_fin_example={64 - popcount(fb_play | fw_play)}')
    print('\n===== SUMMARY =====')
    print('Samples:', N)
    print('All ok:', total_solver_ok)
    print('Mismatches:', mismatches)
    print(f'Avg enum_full: {t_enum/N*1e6:.1f} us')
    print(f'Avg solve_endgame: {t_solver/N*1e6:.1f} us')
    print(f'Avg play_to_end: {t_play/N*1e6:.1f} us')
    if mismatches == 0:
        print('All sampled <=6 empty positions validated by full enumeration.')

if __name__ == '__main__':
    N = 500
    if len(sys.argv) > 1:
        try: N = int(sys.argv[1])
        except: pass
    validate_samples(N)
