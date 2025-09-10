import os
import sys
import json
import argparse
import random
import ctypes
from ctypes import c_uint64, c_int, POINTER, byref

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from botzone.play import OthelloAI, popcount
from utils.greedy import choose_move as greedy_choose

PASS_INDEX = 64
BOARD_MASK = (1 << 64) - 1
INIT_BLACK = 0x0000000810000000
INIT_WHITE = 0x0000001008000000
PLAY_TO_END_SENTINEL = -1073741824  # INT_MIN/2

# Endgame lib wrapper
class EndgameLib:
    def __init__(self):
        self.lib = None
        self.play_to_end = None
        self.solve_endgame = None
        self._load()

    def _load(self):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'botzone', 'data')
        so_path = os.path.abspath(os.path.join(base_dir, 'ending_mac.so'))
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library not found: {so_path}. Please compile ending.c into ending_mac.so")
        lib = ctypes.CDLL(so_path)
        lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
        lib.play_to_end.restype = c_int
        lib.solve_endgame.argtypes = [c_uint64, c_uint64, POINTER(c_int)]
        lib.solve_endgame.restype = c_int
        self.lib = lib
        self.play_to_end = lib.play_to_end
        self.solve_endgame = lib.solve_endgame

    def solve_myopp(self, my: int, opp: int):
        out_my = c_uint64(0)
        out_opp = c_uint64(0)
        res = self.play_to_end(c_uint64(my), c_uint64(opp), byref(out_my), byref(out_opp))
        ok = (res != PLAY_TO_END_SENTINEL)
        return ok, out_my.value, out_opp.value

    def solve_one_best_move(self, my: int, opp: int):
        out_mv = c_int(-1)
        score = int(self.solve_endgame(c_uint64(my), c_uint64(opp), byref(out_mv)))
        return int(out_mv.value), score

END_LIB = EndgameLib()

# Helpers

def enumerate_moves(bb: int):
    res = []
    m = bb
    while m:
        l = m & -m
        res.append(l.bit_length() - 1)
        m ^= l
    return res

class RandomBaseline:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.ai = OthelloAI()
    def choose_move(self, my: int, opp: int) -> int:
        legal = self.ai.get_legal_moves(my, opp)
        if legal == 0:
            return PASS_INDEX
        return self.rng.choice(enumerate_moves(legal))


def finish_with_endgame_myopp(black: int, white: int, side_black_to_move: bool):
    if side_black_to_move:
        ok, fmy, fopp = END_LIB.solve_myopp(black, white)
        return ok, fmy, fopp
    else:
        ok, fmy, fopp = END_LIB.solve_myopp(white, black)
        return ok, fopp, fmy


def endgame_by_per_move_solve(black: int, white: int, side_black_to_move: bool):
    ai = OthelloAI()
    b, w = black, white
    pass_streak = 0
    while True:
        my = b if side_black_to_move else w
        opp = w if side_black_to_move else b
        occ = (my | opp) & BOARD_MASK
        empties = 64 - popcount(occ)
        if empties == 0:
            break
        legal = ai.get_legal_moves(my, opp)
        if legal == 0:
            pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            continue
        pass_streak = 0
        mv, _ = END_LIB.solve_one_best_move(my, opp)
        if mv < 0:
            pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            continue
        nmy, nopp = ai.make_move(my, opp, mv)
        if side_black_to_move:
            b, w = nmy, nopp
        else:
            w, b = nmy, nopp
        side_black_to_move = not side_black_to_move
    return b, w


def true_score_for_side(side_is_black: bool, final_black: int, final_white: int) -> int:
    b = popcount(final_black); w = popcount(final_white)
    empties = 64 - b - w
    if b > w:
        return (b + empties - w) if side_is_black else (w - (b + empties))
    elif w > b:
        return (w + empties - b) if not side_is_black else (b - (w + empties))
    else:
        return 0

class Stats:
    def __init__(self):
        self.w = 0; self.d = 0; self.l = 0
        self.sum_true = 0.0
        self.n = 0
    def add(self, final_black: int, final_white: int, side_is_black: bool):
        b = popcount(final_black); w = popcount(final_white)
        if b == w:
            self.d += 1
        elif (b > w) == side_is_black:
            self.w += 1
        else:
            self.l += 1
        self.sum_true += true_score_for_side(side_is_black, final_black, final_white)
        self.n += 1
    def avg_true(self):
        return (self.sum_true / self.n) if self.n else 0.0


def play_one_game(greedy_is_black: bool, rng: random.Random, dual_counter: dict):
    ai = OthelloAI()
    rnd = RandomBaseline(rng)
    black = INIT_BLACK
    white = INIT_WHITE
    side_black_to_move = True
    pass_streak = 0

    while True:
        my = black if side_black_to_move else white
        opp = white if side_black_to_move else black
        occ = (my | opp) & BOARD_MASK
        empties = 64 - popcount(occ)
        if empties == 0:
            break
        legal = ai.get_legal_moves(my, opp)
        if legal == 0:
            pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            continue

        if empties <= 8:
            b0, w0, s0 = black, white, side_black_to_move
            ok, fb1, fw1 = finish_with_endgame_myopp(b0, w0, s0)
            if ok:
                fb2, fw2 = endgame_by_per_move_solve(b0, w0, s0)
                dual_counter['samples'] = dual_counter.get('samples', 0) + 1
                if (fb1 & BOARD_MASK) != (fb2 & BOARD_MASK) or (fw1 & BOARD_MASK) != (fw2 & BOARD_MASK):
                    dual_counter['mismatch'] = dual_counter.get('mismatch', 0) + 1
                black, white = fb1, fw1
                break
            # if not ok, fallthrough to play normally

        # Midgame move
        if side_black_to_move:
            mv = greedy_choose(my, opp) if greedy_is_black else rnd.choose_move(my, opp)
        else:
            mv = rnd.choose_move(my, opp) if greedy_is_black else greedy_choose(my, opp)

        if mv == PASS_INDEX:
            pass_streak += 1
        else:
            pass_streak = 0
            new_my, new_opp = ai.make_move(my, opp, mv)
            if side_black_to_move:
                black, white = new_my, new_opp
            else:
                white, black = new_my, new_opp

        if pass_streak == 2:
            break
        side_black_to_move = not side_black_to_move

    return black, white


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--games', type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    os.environ['CHASE_SEED'] = str(args.seed)

    games = args.games
    if games % 2 != 0:
        games += 1  # ensure even to split equally

    g_black = Stats()
    g_white = Stats()
    dual = {'samples': 0, 'mismatch': 0}

    # Greedy as Black
    for _ in range(games // 2):
        fb, fw = play_one_game(greedy_is_black=True, rng=random, dual_counter=dual)
        g_black.add(fb, fw, side_is_black=True)
    # Greedy as White
    for _ in range(games // 2):
        fb, fw = play_one_game(greedy_is_black=False, rng=random, dual_counter=dual)
        g_white.add(fb, fw, side_is_black=False)

    # Aggregate
    g_all = Stats()
    g_all.w = g_black.w + g_white.w
    g_all.d = g_black.d + g_white.d
    g_all.l = g_black.l + g_white.l
    g_all.sum_true = g_black.sum_true + g_white.sum_true
    g_all.n = g_black.n + g_white.n

    print("=== greedy vs random ({} games) ===".format(g_all.n))
    print(f"[dual-endgame consistency] samples={dual.get('samples',0)}, mismatches={dual.get('mismatch',0)}")
    print(f"Greedy as Black: W/D/L = {g_black.w}/{g_black.d}/{g_black.l} out of {g_black.n}, avg true = {g_black.avg_true():.3f}")
    print(f"Greedy as White: W/D/L = {g_white.w}/{g_white.d}/{g_white.l} out of {g_white.n}, avg true = {g_white.avg_true():.3f}")
    print(f"Overall:         W/D/L = {g_all.w}/{g_all.d}/{g_all.l} out of {g_all.n}, avg true = {g_all.avg_true():.3f}")

    summary = {
        'dual_endgame_consistency': dual,
        'greedy': {
            'black': {'W': g_black.w, 'D': g_black.d, 'L': g_black.l, 'avg_true': round(g_black.avg_true(), 3)},
            'white': {'W': g_white.w, 'D': g_white.d, 'L': g_white.l, 'avg_true': round(g_white.avg_true(), 3)},
            'overall': {'W': g_all.w, 'D': g_all.d, 'L': g_all.l, 'avg_true': round(g_all.avg_true(), 3)}
        }
    }
    print('\nSUMMARY JSON:')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
