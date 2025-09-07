#!/usr/bin/env python3
"""Minimal sanity validation for current botzone/play.py (without legacy aliases).

Tests performed:
1. Initial board setup (black & white)
2. Initial legal moves for black
3. Single move application correctness (black plays (2,3))
4. Pass move semantics (swap bitboards)
5. Random near-endgame position: endgame solver move legality (if ending.so present)

Exit code 0 on success, 1 on failure.
"""
import os, sys, random, ctypes
from ctypes import c_uint64, c_int, byref

BASE = os.path.abspath(os.path.dirname(__file__))
BOTZONE = os.path.join(BASE, 'botzone')
if BOTZONE not in sys.path:
    sys.path.append(BOTZONE)

from play import OthelloAI

FAIL = 0
TOTAL = 0

def require(cond, msg):
    global FAIL, TOTAL
    TOTAL += 1
    if cond:
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")
        FAIL += 1

def bit_positions(bb:int):
    res = []
    while bb:
        lsb = bb & -bb
        pos = lsb.bit_length() - 1
        res.append(pos)
        bb ^= lsb
    return res

def pos_to_xy(pos):
    return divmod(pos, 8)  # (x,y)

# 1. Initial board setup
ai = OthelloAI()
ai.init_standard_board(1)
# Black expected pieces at (3,4) and (4,3); white at (3,3) and (4,4)
black_expected = {(3,4), (4,3)}
white_expected = {(3,3), (4,4)}
black_actual = {pos_to_xy(p) for p in bit_positions(ai.my_pieces)}
white_actual = {pos_to_xy(p) for p in bit_positions(ai.opp_pieces)}
require(black_actual == black_expected and white_actual == white_expected, "Initial board (black) correct")

ai2 = OthelloAI()
ai2.init_standard_board(-1)
# When my_color = -1, my_pieces should be white discs
white_actual2 = {pos_to_xy(p) for p in bit_positions(ai2.my_pieces)}
black_actual2 = {pos_to_xy(p) for p in bit_positions(ai2.opp_pieces)}
require(white_actual2 == white_expected and black_actual2 == black_expected, "Initial board (white) correct")

# 2. Initial legal moves
ai.init_standard_board(1)
legal_bb = ai.get_legal_moves(ai.my_pieces, ai.opp_pieces)
legal_moves = {pos_to_xy(p) for p in bit_positions(legal_bb)}
expected_legal = {(2,3),(3,2),(4,5),(5,4)}
require(legal_moves == expected_legal, f"Initial legal moves match {expected_legal}")

# 3. Single move application (black plays (2,3))
move_pos = 2*8 + 3
new_my, new_opp = ai.make_move(ai.my_pieces, ai.opp_pieces, move_pos)
ai.my_pieces, ai.opp_pieces = new_my, new_opp
# After move, (2,3) should be black, (3,3) flipped to black
black_set = {pos_to_xy(p) for p in bit_positions(ai.my_pieces)}
require((2,3) in black_set and (3,3) in black_set, "Move (2,3) applied & flip (3,3)")
# Counts: my should have 4, opp 1
require(len(black_set) == 4 and len({pos_to_xy(p) for p in bit_positions(ai.opp_pieces)}) == 1, "Piece counts after first move correct")

# 4. Pass semantics
before_my, before_opp = ai.my_pieces, ai.opp_pieces
p_my, p_opp = ai.make_move(ai.my_pieces, ai.opp_pieces, -1)  # pass
require(p_my == before_opp and p_opp == before_my, "PASS returns swapped (opp,my)")

# 5. Endgame solver legality (if available)
ai_end = OthelloAI()
# Build random near-endgame position by random playout
random.seed(2025)
ai_end.init_standard_board(1)
me = ai_end.my_pieces; opp = ai_end.opp_pieces
turn_black = True
while True:
    empties = 64 - ((me | opp).bit_count())
    if empties <= 16:
        break
    moves = ai_end.get_legal_moves(me, opp)
    if moves == 0:
        # pass
        moves_opp = ai_end.get_legal_moves(opp, me)
        if moves_opp == 0:
            break
        me, opp = opp, me
        turn_black = not turn_black
        continue
    # pick first move
    lsb = moves & -moves
    pos = lsb.bit_length() - 1
    me, opp = ai_end.make_move(me, opp, pos)
    me, opp = opp, me
    turn_black = not turn_black

ai_end.my_pieces = me
ai_end.opp_pieces = opp
ai_end.my_color = 1 if turn_black else -1  # current side to move as 'my'
# attempt endgame search
res = ai_end.endgame_search()
if res is None:
    print("[INFO] ending.so not available; skipping endgame legality test")
else:
    x,y,val = res
    if x < 0:
        # pass expected only if no legal moves
        legal_now = ai_end.get_legal_moves(ai_end.my_pieces, ai_end.opp_pieces)
        require(legal_now == 0, "Endgame solver PASS only when no legal moves")
    else:
        pos = x*8 + y
        legal_now = ai_end.get_legal_moves(ai_end.my_pieces, ai_end.opp_pieces)
        require(((legal_now >> pos) & 1) == 1, "Endgame solver move is legal")

print(f"\nSummary: {TOTAL-FAIL} passed / {TOTAL} total; failures={FAIL}")
if FAIL:
    sys.exit(1)
