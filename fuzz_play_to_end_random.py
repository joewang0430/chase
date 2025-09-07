#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random fuzz test for play_to_end vs iterative solve_endgame equivalence.
Generates reachable Reversi positions with exactly 8 empties (56 discs) and BLACK to move
(by simulating random legal games from the initial position) then:
 1. Calls play_to_end (C) directly
 2. Calls solve_endgame repeatedly applying best moves until terminal
Compares final bitboards. Reports any mismatches (should be zero).

Usage:
  python3 fuzz_play_to_end_random.py [N]
Default N=100
"""
import os, sys, random, time, ctypes, subprocess
from ctypes import c_uint64, c_int, byref

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'botzone', 'data', 'ending.c')
SO_PATH_MAC = os.path.join(ROOT, 'botzone', 'data', 'ending_mac.so')
SO_PATH_LINUX = os.path.join(ROOT, 'botzone', 'data', 'ending.so')  # in case running in container

# Decide which .so to use (prefer mac build if exists, else linux)
if os.path.exists(SO_PATH_MAC):
    SO_PATH = SO_PATH_MAC
else:
    SO_PATH = SO_PATH_MAC if sys.platform == 'darwin' else SO_PATH_LINUX

# Build shared object if needed (mac path)
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

def random_reachable_board_with_empties(target_empties=8, max_attempts=10000):
    """Play random legal moves from initial position until exactly target_empties and BLACK to move.
    Return (black, white) bitboards.
    """
    for attempt in range(max_attempts):
        # Initial position: black to move
        black = (1 << (3*8+4)) | (1 << (4*8+3))  # e4 & d5 -> (3,4),(4,3)
        white = (1 << (3*8+3)) | (1 << (4*8+4))  # d4 & e5 -> (3,3),(4,4)
        black_to_move = True
        passes = 0
        # Random play until empties <= target
        while True:
            empties = 64 - popcount(black | white)
            if empties <= target_empties:
                break
            me = black if black_to_move else white
            opp = white if black_to_move else black
            moves = generate_moves(me, opp)
            if moves == 0:
                passes += 1
                if passes == 2:
                    break  # game ended prematurely
                black_to_move = not black_to_move
                continue
            passes = 0
            # choose random move
            # collect list
            tmp = moves
            choices = []
            while tmp:
                p = (tmp & -tmp).bit_length() - 1
                choices.append(p)
                tmp &= tmp - 1
            mv = random.choice(choices)
            if black_to_move:
                black, white = make_move(black, white, mv)
            else:
                white, black = make_move(white, black, mv)
            black_to_move = not black_to_move
        empties = 64 - popcount(black | white)
        if empties == target_empties and black_to_move:  # BLACK to move
            return black, white
    raise RuntimeError(f"Failed to generate board with {target_empties} empties and black to move after {max_attempts} attempts")


def iterative_final(black0, white0):
    my = black0; opp = white0; board0_is_black = True; passes = 0
    while passes < 2:
        empties = 64 - popcount(my | opp)
        if empties == 0:
            passes = 2
            break
        best_c = c_int(-1)
        _score = lib.solve_endgame(my, opp, byref(best_c))
        mv = best_c.value
        if mv < 0:  # PASS
            passes += 1
            my, opp = opp, my
            board0_is_black = not board0_is_black
            continue
        passes = 0
        my, opp = make_move(my, opp, mv)
        my, opp = opp, my
        board0_is_black = not board0_is_black
    final_black = my if board0_is_black else opp
    final_white = opp if board0_is_black else my
    return final_black, final_white


def run_fuzz(N=100, seed=None):
    if seed is not None:
        random.seed(seed)
    mismatches = 0
    mismatch_examples = []
    total_time_play = 0.0
    total_time_iter = 0.0
    for i in range(1, N+1):
        black, white = random_reachable_board_with_empties(8)
        fb = c_uint64(); fw = c_uint64()
        t0 = time.perf_counter()
        flag = lib.play_to_end(black, white, byref(fb), byref(fw))
        t1 = time.perf_counter()
        final_black_play, final_white_play = fb.value, fw.value
        t2 = time.perf_counter()
        final_black_iter, final_white_iter = iterative_final(black, white)
        t3 = time.perf_counter()
        dt_play = (t1 - t0) * 1e6
        dt_iter = (t3 - t2) * 1e6
        total_time_play += dt_play
        total_time_iter += dt_iter
        same = (final_black_play == final_black_iter) and (final_white_play == final_white_iter)
        if not same:
            mismatches += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((black, white, final_black_play, final_white_play, final_black_iter, final_white_iter))
        # Progress output every 100 samples (or immediately on mismatch)
        if (i % 100 == 0) or (not same):
            print(f'[PROGRESS] {i}/{N} same={same} avg_play={total_time_play/i:.1f}us avg_iter={total_time_iter/i:.1f}us')
    print('\n===== SUMMARY =====')
    print(f'Total samples: {N}')
    print(f'Mismatches: {mismatches}')
    print(f'Average play_to_end time: {total_time_play/N:.1f} us')
    print(f'Average iterative   time: {total_time_iter/N:.1f} us')
    if mismatches:
        print('First mismatches:')
        for m in mismatch_examples:
            print(m)
    else:
        print('All positions matched.')

if __name__ == '__main__':
    N = 100
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except: pass
    run_fuzz(N)
