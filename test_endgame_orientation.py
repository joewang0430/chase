import os, sys, random, ctypes
from ctypes import c_uint64, c_int, byref

# Ensure botzone path available
BASE = os.path.dirname(os.path.abspath(__file__))
BOTZONE = os.path.join(BASE, 'botzone')
if BOTZONE not in sys.path:
    sys.path.append(BOTZONE)

try:
    from play import OthelloAI
except Exception as e:
    print(f"IMPORT_FAIL play.py: {e}")
    sys.exit(1)

# ---- Utility: horizontal mirror (reverse bits within each 8-bit row) ----
_REV_BYTE = bytes.maketrans(
    bytes(range(256)),
    bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))
)

def mirror_horizontal(bb: int) -> int:
    # process row by row (8 rows, each 8 bits)
    res = 0
    for r in range(8):
        row = (bb >> (r*8)) & 0xFF
        rev = _REV_BYTE[row]
        res |= rev << (r*8)
    return res

# ---- Load ending.so ----
so_path = os.path.join(BOTZONE, 'data', 'ending.so')
if not os.path.exists(so_path):
    print(f"MISSING ending.so at {so_path}")
    sys.exit(0)
try:
    lib = ctypes.CDLL(so_path)
    solve_endgame = lib.solve_endgame
    solve_endgame.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
    solve_endgame.restype = c_int
except AttributeError:
    print("solve_endgame symbol not found in ending.so")
    sys.exit(0)

# ---- Generate near-endgame positions (empties <= 16) ----
ENG = OthelloAI()
random.seed(2024)
positions = []
MAX_SAMPLES = 80
TRIES = 4000
for _ in range(TRIES):
    ENG.init_standard_board(1)
    my, opp = ENG.my_pieces, ENG.opp_pieces
    # random playout until empties <=16 or terminal
    while True:
        empties = 64 - (my | opp).bit_count()
        if empties <= 16:
            positions.append((my, opp))
            break
        mv_bb = ENG.generate_moves_fast(my, opp)
        if mv_bb == 0:
            mv_bb_opp = ENG.generate_moves_fast(opp, my)
            if mv_bb_opp == 0:
                break
            my, opp = opp, my  # pass
            continue
        # choose random legal
        lsb = mv_bb & -mv_bb
        pos = lsb.bit_length() - 1 if random.random() < 0.5 else random.choice([
            ( (tmp & -tmp).bit_length() - 1) for tmp in [mv_bb]  # keep simple
        ])
        my, opp = ENG.fast_make_move(my, opp, pos)
        my, opp = opp, my  # switch side
    if len(positions) >= MAX_SAMPLES:
        break

if not positions:
    print("No positions generated with empties <=16")
    sys.exit(0)

# ---- Test orientation ----

def legal_moves_internal(my, opp):
    return ENG.generate_moves_fast(my, opp)

ok_direct = 0
ok_mirror = 0
fail = 0
sample_details = []

for idx, (my, opp) in enumerate(positions):
    moves_bb = legal_moves_internal(my, opp)
    best_c = c_int(-1)
    score = solve_endgame(c_uint64(my), c_uint64(opp), byref(best_c))
    best = best_c.value
    empties = 64 - (my | opp).bit_count()
    if moves_bb == 0:
        # expect pass
        if best < 0:
            ok_direct += 1
        else:
            fail += 1
        continue
    # we have legal moves
    if best >= 0:
        if (moves_bb >> best) & 1:
            ok_direct += 1
            continue
        # try mirror horizontal mapping of C result back to internal
        # Mirror index within row: col' = 7 - col
        row, col = divmod(best, 8)
        mirror_index = row*8 + (7 - col)
        if (moves_bb >> mirror_index) & 1:
            ok_mirror += 1
            sample_details.append((idx, 'MIRROR_USED', best, mirror_index))
            continue
        # else attempt full board mirror call
    # Attempt mirrored call path
    my_m = mirror_horizontal(my)
    opp_m = mirror_horizontal(opp)
    best_c2 = c_int(-1)
    _ = solve_endgame(c_uint64(my_m), c_uint64(opp_m), byref(best_c2))
    best2 = best_c2.value
    if best2 >= 0:
        # Convert mirrored best2 back: row same, col mirrored
        r2, c2 = divmod(best2, 8)
        orig_index = r2*8 + (7 - c2)
        if (moves_bb >> orig_index) & 1:
            ok_mirror += 1
            sample_details.append((idx, 'FULL_MIRROR_SOLVE', best, orig_index))
            continue
    fail += 1
    sample_details.append((idx, 'FAIL', best, -1))

print("=== Endgame Orientation Check ===")
print(f"Samples: {len(positions)}")
print(f"Direct matches: {ok_direct}")
print(f"Mirror matches: {ok_mirror}")
print(f"Failures: {fail}")
if fail == 0:
    if ok_mirror > 0 and ok_direct == 0:
        print("RESULT: Orientation requires horizontal mirroring.")
    elif ok_direct > 0 and ok_mirror == 0:
        print("RESULT: Orientation consistent (no mirror needed).")
    else:
        print("RESULT: Mixed but all accounted; prefer direct if majority direct.")
else:
    print("Some failures remain; inspect sample_details for debugging.")

# Optional detail output (limit 10)
for rec in sample_details[:10]:
    print("DETAIL:", rec)
