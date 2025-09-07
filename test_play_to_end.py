#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for play_to_end in ending_mac.so

Creates test positions with <=10 empty squares (black to move),
compiles ending.c as ending_mac.so (to avoid name collision),
invokes play_to_end, prints the final board and result plus timing.
"""
import os
import subprocess
import ctypes
import time
from ctypes import c_uint64, c_int, POINTER, byref

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'botzone', 'data', 'ending.c')
SO_PATH = os.path.join(ROOT, 'botzone', 'data', 'ending_mac.so')

# Compile ending.c -> ending_mac.so (force rebuild if source newer or .so missing)
def build_shared():
    need = (not os.path.exists(SO_PATH) or
            os.path.getmtime(SO_PATH) < os.path.getmtime(SRC))
    if not need:
        return
    print('[Build] Compiling ending.c -> ending_mac.so')
    cmd = ['clang', '-shared', '-O3', '-fPIC', SRC, '-o', SO_PATH]
    try:
        subprocess.check_call(cmd, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        print('Compilation failed:', e)
        raise SystemExit(1)

# Board representation (row 1 at top). B=black, W=white, .=empty
#   a b c d e f g h
# 1 B B B B B B . .
# 2 . W W W B B . .
# 3 W W W W W W . .
# 4 W W B W B B B B
# 5 W W B W W W W W
# 6 . W W B W W W .
# 7 . W B W W W W W
# 8 W B B W B B B B
initial_rows = [
    'BBBBBB..',
    '.WWWBB..',
    'WWWWWW..',
    'WWBWBBBB',
    'WWBWWWWW',
    '.WWBWWW.',
    '.WBWWWWW',
    'WBBWBBBB'
]
# Sanity check length
for r in initial_rows:
    if len(r) != 8:
        raise ValueError('Row length not 8: ' + r)

# Build bitboards
black_bb = 0
white_bb = 0
empties = 0
for r in range(8):
    for c in range(8):
        ch = initial_rows[r][c]
        idx = r * 8 + c
        if ch == 'B':
            black_bb |= 1 << idx
        elif ch == 'W':
            white_bb |= 1 << idx
        else:
            empties += 1

if empties != 10:
    print('[Warn] Expected 10 empties, got', empties)

print('[Init] Black bitboard: 0x%016X' % black_bb)
print('[Init] White bitboard: 0x%016X' % white_bb)
print('[Init] Empties:', empties)

build_shared()
lib = ctypes.CDLL(SO_PATH)
# int play_to_end(uint64_t black, uint64_t white, uint64_t *final_black, uint64_t *final_white)
lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
lib.play_to_end.restype = c_int

final_black = c_uint64()
final_white = c_uint64()
SENTINEL = - (2**31) // 2  # matches (INT_MIN/2) in C (-1073741824)

start = time.perf_counter()
result = lib.play_to_end(black_bb, white_bb, byref(final_black), byref(final_white))
elapsed = (time.perf_counter() - start) * 1000.0

if result == SENTINEL:
    print('[Result] Sentinel returned (position has >8 empties) - skip first board output.')
else:
    fb = final_black.value
    fw = final_white.value
    def print_board(black_bits: int, white_bits: int):
        print('   a b c d e f g h')
        for r in range(8):
            row_chars = []
            for c in range(8):
                idx = r * 8 + c
                mask = 1 << idx
                if black_bits & mask:
                    row_chars.append('●')
                elif white_bits & mask:
                    row_chars.append('○')
                else:
                    row_chars.append('·')
            print(f' {r+1} ' + ' '.join(row_chars))
    print('\n[Final Board]')
    print_board(fb, fw)
    black_count = bin(fb).count('1')
    white_count = bin(fw).count('1')
    print(f'Black count: {black_count}, White count: {white_count}')
    print(f'play_to_end return value (1=Black win, -1=White win, 0=Draw): {result}')
    print(f'Time: {elapsed:.3f} ms')
    if black_count + white_count != 64:
        print('[Note] Board not full; implies double pass end. (Total pieces =', black_count + white_count, ')')

# ---------------- Second Test: 8 empties position ----------------
print('\n================ Second Test: 8 Empties Position ================')
second_rows = [
    'BBBBBB..',  # 1
    'BBBBBB..',  # 2
    'WBWWWW.W',  # 3
    'WWBWBBWW',  # 4
    'WWBWWWWW',  # 5
    '.WWBWWW.',  # 6
    '.WBWWWWW',  # 7
    'WBBWBBBB',  # 8
]

for r in second_rows:
    if len(r) != 8:
        raise ValueError('[Second] Row length not 8: ' + r)

black2 = 0
white2 = 0
empties2 = 0
for r in range(8):
    for c in range(8):
        ch = second_rows[r][c]
        idx = r * 8 + c
        if ch == 'B':
            black2 |= 1 << idx
        elif ch == 'W':
            white2 |= 1 << idx
        else:
            empties2 += 1

print('[Second] Black bitboard: 0x%016X' % black2)
print('[Second] White bitboard: 0x%016X' % white2)
print('[Second] Empties:', empties2)

if empties2 != 8:
    print('[Second][Warn] Expected 8 empties, got', empties2)

final_black2 = c_uint64()
final_white2 = c_uint64()
start2 = time.perf_counter()
result2 = lib.play_to_end(black2, white2, byref(final_black2), byref(final_white2))
elapsed2 = (time.perf_counter() - start2) * 1000.0

if result2 == SENTINEL:
    print('[Second][Error] Unexpected sentinel for 8-empties position')
else:
    def print_board2(black_bits: int, white_bits: int):
        print('   a b c d e f g h')
        for r in range(8):
            row_chars = []
            for c in range(8):
                idx = r * 8 + c
                mask = 1 << idx
                if black_bits & mask:
                    row_chars.append('●')
                elif white_bits & mask:
                    row_chars.append('○')
                else:
                    row_chars.append('·')
            print(f' {r+1} ' + ' '.join(row_chars))
    print('\n[Second Final Board]')
    print_board2(final_black2.value, final_white2.value)
    bcnt2 = bin(final_black2.value).count('1')
    wcnt2 = bin(final_white2.value).count('1')
    print(f'[Second] Black count: {bcnt2}, White count: {wcnt2}')
    print(f'[Second] play_to_end return value: {result2}')
    print(f'[Second] Time: {elapsed2:.3f} ms')
    if bcnt2 + wcnt2 != 64:
        print('[Second][Note] Board not full; implies double pass end. Total pieces =', bcnt2 + wcnt2)
# -----------------------------------------------------------------
# Third test: user-provided 8-empties position
print('\n================ Third Test: User 8 Empties Position ================')
third_rows = [
    'WWWWWWWW',  # 1
    'BBBWBBBW',  # 2
    'WBWBBBBW',  # 3
    'WWBBBWBW',  # 4
    'WWWBBBWW',  # 5
    '.WWWBWBW',  # 6
    '.WWWW..W',  # 7
    'W.WW...W',  # 8
]
for r in third_rows:
    if len(r) != 8:
        raise ValueError('[Third] Row length not 8: ' + r)
black3 = 0
white3 = 0
empties3 = 0
for r in range(8):
    for c in range(8):
        ch = third_rows[r][c]
        idx = r * 8 + c
        if ch == 'B':
            black3 |= 1 << idx
        elif ch == 'W':
            white3 |= 1 << idx
        else:
            empties3 += 1
print('[Third] Black bitboard: 0x%016X' % black3)
print('[Third] White bitboard: 0x%016X' % white3)
print('[Third] Empties:', empties3)
if empties3 != 8:
    print('[Third][Warn] Expected 8 empties, got', empties3)
final_black3 = c_uint64()
final_white3 = c_uint64()
start3 = time.perf_counter()
result3 = lib.play_to_end(black3, white3, byref(final_black3), byref(final_white3))
elapsed3 = (time.perf_counter() - start3) * 1000.0
if result3 == SENTINEL:
    print('[Third][Error] Unexpected sentinel for this 8-empties position')
else:
    def print_board3(black_bits: int, white_bits: int):
        print('   a b c d e f g h')
        for r in range(8):
            row_chars = []
            for c in range(8):
                idx = r * 8 + c
                mask = 1 << idx
                if black_bits & mask:
                    row_chars.append('●')
                elif white_bits & mask:
                    row_chars.append('○')
                else:
                    row_chars.append('·')
            print(f' {r+1} ' + ' '.join(row_chars))
    print('\n[Third Final Board]')
    print_board3(final_black3.value, final_white3.value)
    bcnt3 = bin(final_black3.value).count('1')
    wcnt3 = bin(final_white3.value).count('1')
    print(f'[Third] Black count: {bcnt3}, White count: {wcnt3}')
    print(f'[Third] play_to_end return value: {result3}')
    print(f'[Third] Time: {elapsed3:.3f} ms')
    if bcnt3 + wcnt3 != 64:
        print('[Third][Note] Board not full; implies double pass end. Total pieces =', bcnt3 + wcnt3)
# -----------------------------------------------------------------
