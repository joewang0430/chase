import os, sys, time, random, ctypes
from ctypes import c_uint64, c_int, byref

# Ensure botzone path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOTZONE = os.path.join(ROOT, 'botzone')
if BOTZONE not in sys.path:
    sys.path.append(BOTZONE)

from play import OthelloAI

# Load solver
SOLIB = os.path.join(BOTZONE, 'data', 'ending_mac.so')
if not os.path.exists(SOLIB):
    SOLIB = os.path.join(BOTZONE, 'data', 'ending.so')
lib = ctypes.CDLL(SOLIB)
lib.solve_endgame.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
lib.solve_endgame.restype = c_int
print(f"[INFO] Using solver: {SOLIB}")

ai = OthelloAI()

# ---------------- Manual board from user ----------------
# Legend: '●' black, '○' white, '·' empty
manual_board_rows = [
"· · ○ · ● · · ·",  # row1
"○ ○ ○ ○ ● ● ● ·",  # row2
"· ○ ○ ○ ○ ● ● ○",  # row3
"· ● · ● ● ○ ● ●",  # row4
"● · ● ● ○ ○ ● ●",  # row5
"● ● ● ● ○ ● ● ○",  # row6
"● ● ● · ● ● · ·",  # row7
"● ○ ○ · ○ ● ● ·",  # row8
]
BLACK='●'; WHITE='○'; EMPTY='·'
black_bb = 0
white_bb = 0
for r,line in enumerate(manual_board_rows):
    cells = line.split()
    for c,ch in enumerate(cells):
        idx = r*8 + c
        if ch == BLACK:
            black_bb |= 1<<idx
        elif ch == WHITE:
            white_bb |= 1<<idx
empties_manual = 64 - (black_bb | white_bb).bit_count()
print(f"[MANUAL] black bits={black_bb:#018x} white bits={white_bb:#018x} empties={empties_manual}")

# Helper: list moves from Python engine

def python_moves(me, opp):
    return ai.generate_moves_fast(me, opp)

# Test manual board
def test_manual():
    results = []
    for side,label in [(black_bb, 'BLACK_TO_MOVE'), (white_bb, 'WHITE_TO_MOVE')]:
        me = side
        opp = white_bb if side==black_bb else black_bb
        moves_py = python_moves(me, opp)
        best_c = c_int(-1)
        score = lib.solve_endgame(me, opp, byref(best_c))
        best = best_c.value
        legal = (best < 0 and moves_py == 0) or (best >=0 and ((moves_py>>best)&1))
        results.append((label, moves_py, best, legal, score))
    return results

manual_results = test_manual()
for label, mv, best, legal, score in manual_results:
    mv_list=[]; tmp=mv
    while tmp:
        lsb= tmp & -tmp
        mv_list.append(lsb.bit_length()-1)
        tmp ^= lsb
    print(f"[MANUAL-{label}] moves={mv_list} best={best} legal={legal} score={score}")

# ---------------- Improved legal playout generator (avoid pathological random fills) ----------------

def playout_endgame(target_empties=16):
    ai.init_standard_board(1)  # black starts
    me = ai.my_pieces; opp = ai.opp_pieces
    color = 1  # current side = me
    passes = 0
    while True:
        empties = 64 - (me | opp).bit_count()
        if empties <= target_empties:
            return me, opp  # me to move
        moves = python_moves(me, opp)
        if moves == 0:
            # pass
            passes += 1
            if passes == 2:
                return me, opp
            me, opp = opp, me
            color = -color
            continue
        passes = 0
        # collect moves list
        mv_list = []
        tmp = moves
        while tmp:
            lsb = tmp & -tmp
            mv_list.append(lsb.bit_length()-1)
            tmp ^= lsb
        pos = random.choice(mv_list)
        me, opp = ai.fast_make_move(me, opp, pos)
        # swap sides
        me, opp = opp, me
        color = -color

# ---------------- Random endgame positions (Test A) ----------------
random.seed(2025)
N = 50  # reduced to avoid long runtime
TARGET_EMPTIES = 14  # tighter for faster solves
illegal=0
passes_ok=0
solve_times=[]
for i in range(N):
    me, opp = playout_endgame(TARGET_EMPTIES)
    moves_py = python_moves(me, opp)
    best_c = c_int(-1)
    t0=time.perf_counter()
    score = lib.solve_endgame(me, opp, byref(best_c))
    t1=time.perf_counter()
    solve_times.append(t1-t0)
    bm = best_c.value
    if moves_py==0:
        if bm<0: passes_ok+=1
        else: illegal+=1
    else:
        if bm<0 or not ((moves_py>>bm)&1):
            illegal+=1
            print(f"[ILLEGAL] idx={i} bm={bm} moves_py={moves_py:#018x}")
            break
    if (i+1)%10==0:
        avg_ms = sum(solve_times)/len(solve_times)*1000
        print(f"[PROGRESS] {i+1}/{N} avg_ms={avg_ms:.2f}")
print(f"[RANDOM] total={N} illegal={illegal} pass_ok={passes_ok}")
if solve_times:
    print(f"[RANDOM] overall_avg_ms={sum(solve_times)/len(solve_times)*1000:.2f}")

# ---------------- PASS Scenario (Test B) ----------------

def find_pass_position(max_attempts=2000):
    for _ in range(max_attempts):
        me, opp = playout_endgame(16)
        if python_moves(me, opp)==0 and python_moves(opp, me)!=0:
            return me, opp
    return None

pass_pos = find_pass_position()
if pass_pos:
    me, opp = pass_pos
    best_c = c_int(-1)
    score = lib.solve_endgame(me, opp, byref(best_c))
    print(f"[PASS] best_move={best_c.value} expected -1; score={score}")
else:
    print("[PASS] Not found in attempts")

# ---------------- Performance (Test C) ----------------
M=40
cases=[playout_endgame(14) for _ in range(M)]
start=time.perf_counter()
for me, opp in cases:
    best_c=c_int(-1)
    lib.solve_endgame(me, opp, byref(best_c))
end=time.perf_counter()
print(f"[PERF] {M} solves avg_ms={(end-start)/M*1000:.2f}")
print("DONE")
