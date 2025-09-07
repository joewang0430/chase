import os, sys, random, ctypes
from ctypes import c_uint64, c_int, byref

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
BOTZONE = os.path.join(ROOT, 'botzone')
DATA = os.path.join(BOTZONE, 'data')
if BOTZONE not in sys.path:
    sys.path.append(BOTZONE)

from play import OthelloAI

# Byte mirror table for horizontal flip (reverse bits inside each 8-bit row)
_REV_BYTE = bytes.maketrans(
    bytes(range(256)),
    bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))
)

def mirror_horizontal(bb: int) -> int:
    res = 0
    for r in range(8):
        row = (bb >> (r*8)) & 0xFF
        rev = _REV_BYTE[row]
        res |= rev << (r*8)
    return res

def load_solver():
    # Prefer mac build; fallback to generic ending.so
    cand = [os.path.join(DATA, 'ending_mac.so'), os.path.join(DATA, 'ending.so')]
    for p in cand:
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(p)
                fn = lib.solve_endgame
                fn.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
                fn.restype = c_int
                return fn, p
            except Exception:
                continue
    return None, None

solve_endgame, lib_path = load_solver()
if solve_endgame is None:
    print('NO_SOLVER_LIBRARY_FOUND')
    sys.exit(0)
print(f'[INFO] Loaded solver: {lib_path}')

ENG = OthelloAI()
random.seed(2025)
positions = []
MAX_SAMPLES = 120
TRIES = 8000
for _ in range(TRIES):
    ENG.init_standard_board(1)
    my, opp = ENG.my_pieces, ENG.opp_pieces
    while True:
        empties = 64 - (my | opp).bit_count()
        if empties <= 16:
            positions.append((my, opp))
            break
        mv_bb = ENG.generate_moves_fast(my, opp)
        if mv_bb == 0:
            mv_opp = ENG.generate_moves_fast(opp, my)
            if mv_opp == 0:
                break
            my, opp = opp, my
            continue
        # pick a random legal move
        legal_list = []
        tmp = mv_bb
        while tmp:
            lsb = tmp & -tmp
            legal_list.append(lsb.bit_length() - 1)
            tmp ^= lsb
        my, opp = ENG.fast_make_move(my, opp, random.choice(legal_list))
        my, opp = opp, my
    if len(positions) >= MAX_SAMPLES:
        break

print(f'[INFO] Generated {len(positions)} positions with empties <=16')
if not positions:
    sys.exit(0)

ok_direct = ok_mirror = pass_ok = fail = 0
for (my, opp) in positions:
    moves = ENG.generate_moves_fast(my, opp)
    best_c = c_int(-1)
    _score = solve_endgame(c_uint64(my), c_uint64(opp), byref(best_c))
    best = best_c.value
    if moves == 0:
        if best < 0:
            pass_ok += 1
        else:
            fail += 1
        continue
    if best >= 0 and ((moves >> best) & 1):
        ok_direct += 1
        continue
    # try horizontal mirror mapping of result index
    if best >= 0:
        r, c = divmod(best, 8)
        mirror_idx = r*8 + (7-c)
        if (moves >> mirror_idx) & 1:
            ok_mirror += 1
            continue
    # attempt solving mirrored board
    my_m = mirror_horizontal(my)
    opp_m = mirror_horizontal(opp)
    best_c2 = c_int(-1)
    _ = solve_endgame(c_uint64(my_m), c_uint64(opp_m), byref(best_c2))
    b2 = best_c2.value
    if b2 >= 0:
        r2, c2 = divmod(b2, 8)
        orig_idx = r2*8 + (7-c2)
        if (moves >> orig_idx) & 1:
            ok_mirror += 1
            continue
    fail += 1

print('=== ORIENTATION SUMMARY ===')
print(f'Direct legal: {ok_direct}')
print(f'Mirror legal: {ok_mirror}')
print(f'Pass correct: {pass_ok}')
print(f'Failures: {fail}')

if fail == 0:
    if ok_direct and not ok_mirror:
        print('RESULT: No mirroring needed.')
    elif ok_mirror and not ok_direct:
        print('RESULT: Horizontal mirroring required (shift directions opposite).')
    else:
        print('RESULT: Mixed but all consistent after considering mirror.')
else:
    print('RESULT: Some failures remain; investigate further.')
