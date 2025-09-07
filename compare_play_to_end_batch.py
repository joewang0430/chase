#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch comparison:
For each provided <=8-empties position (black to move), compare:
 1. Direct play_to_end
 2. Iterative sequence of solve_endgame calls applying the chosen move each ply
Verify final boards (bitboards) match. Print timings.
"""
import os, ctypes, time, subprocess
from ctypes import c_uint64, c_int, byref, POINTER

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'botzone', 'data', 'ending.c')
SO_PATH = os.path.join(ROOT, 'botzone', 'data', 'ending_mac.so')

# Boards (list of 8-row ASCII strings). '.' empty, 'B' black, 'W' white.
# Board A (8 empties)
BOARD_A = [
    '.BBBBBBW',
    'BBBBBBW.',
    'BBBWBWBW',
    'B.BWWBB.',
    '.WBWBWBB',
    '.WBBWWBB',
    'WWWWBWBB',
    'BWW.WWW.',
]
# Board B (8 empties)
BOARD_B = [
    'BBBBBBBB',
    'BBBBWWBB',
    'BWBWBWWB',
    'BWWBBWWB',
    'WWBBBWWB',
    'WWBWWWWB',
    '.WWW...B',
    '.WWW...B',
]
# Board C (8 empties)
BOARD_C = [
    '.WWWWWWW',
    'WWWWWWWW',
    '.WWBBWWW',
    'BWWWBBWW',
    'BWWBWWB.',
    'B.BWWWB.',
    '.BBBBBBW',
    '..WWWWBW',
]
ALL_BOARDS = [("A", BOARD_A), ("B", BOARD_B), ("C", BOARD_C)]

# Popcount
POPCNT8 = [bin(i).count('1') for i in range(256)]
def popcount(x: int) -> int:
    s = 0
    while x:
        s += POPCNT8[x & 0xFF]
        x >>= 8
    return s

# Compile shared object if needed
def build_so():
    if (not os.path.exists(SO_PATH)) or os.path.getmtime(SO_PATH) < os.path.getmtime(SRC):
        print('[BUILD] Compiling ending.c -> ending_mac.so')
        subprocess.check_call(['clang','-O3','-fPIC','-shared',SRC,'-o',SO_PATH])

build_so()
lib = ctypes.CDLL(SO_PATH)
lib.solve_endgame.argtypes = [c_uint64, c_uint64, POINTER(c_int)]
lib.solve_endgame.restype  = c_int
lib.play_to_end.argtypes   = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
lib.play_to_end.restype    = c_int

# Bitboard helpers
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

def rows_to_bitboards(rows):
    black = white = 0
    empties = 0
    for r in range(8):
        row = rows[r]
        if len(row) != 8:
            raise ValueError('Row length not 8: '+row)
        for c in range(8):
            ch = row[c]
            idx = r*8 + c
            if ch == 'B':
                black |= 1 << idx
            elif ch == 'W':
                white |= 1 << idx
            else:
                empties += 1
    return black, white, empties

def print_board(tag, bbits, wbits):
    print(tag)
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

for name, rows in ALL_BOARDS:
    print(f'\n===== Board {name} =====')
    black0, white0, empties0 = rows_to_bitboards(rows)
    print(f'[START] Empties={empties0}')
    print_board('[START BOARD]', black0, white0)

    # Direct play_to_end
    fb = c_uint64(); fw = c_uint64()
    t0 = time.perf_counter()
    res_play = lib.play_to_end(black0, white0, byref(fb), byref(fw))
    t1 = (time.perf_counter() - t0)*1000
    final_black_play = fb.value; final_white_play = fw.value

    # Iterative solve_endgame
    my = black0; opp = white0; board0_is_black = True; passes = 0
    t2s = time.perf_counter()
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
    final_black_iter = my if board0_is_black else opp
    final_white_iter = opp if board0_is_black else my
    t2 = (time.perf_counter() - t2s)*1000

    same = (final_black_play == final_black_iter) and (final_white_play == final_white_iter)

    print_board('[Final via play_to_end]', final_black_play, final_white_play)
    print_board('[Final via iterative ]', final_black_iter, final_white_iter)
    print(f'[RESULT] play_to_end flag={res_play}  identical={same}')
    print(f'[TIME] play_to_end={t1:.3f} ms  iterative={t2:.3f} ms')
    print(f'[COUNTS] play_to_end B={popcount(final_black_play)} W={popcount(final_white_play)} | iterative B={popcount(final_black_iter)} W={popcount(final_white_iter)}')
    if not same:
        print('[MISMATCH] Bitboards differ!')
