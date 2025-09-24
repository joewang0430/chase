# PYTHONPATH=. python3 nn_tests/test_policy.py --model models/chase.pt --seed 42 --games-per-opponent 200

import os
import sys
import argparse
import random
import ctypes
from ctypes import c_uint64, c_int, POINTER, byref
import json

import torch

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from botzone.play import OthelloAI, popcount
from nn.model_def import Net, build_input_planes
from utils.greedy import choose_move as greedy_choose


PASS_INDEX = 64
BOARD_MASK = (1 << 64) - 1
INIT_BLACK = 0x0000000810000000
INIT_WHITE = 0x0000001008000000
PLAY_TO_END_SENTINEL = -1073741824  # INT_MIN/2
CENTER_FIRST_MOVES = {19, 26, 37, 44}  # d3, c4, f5, e6


def enumerate_moves(bb: int):
    res = []
    m = bb
    while m:
        l = m & -m
        res.append(l.bit_length() - 1)
        m ^= l
    return res


class EndgameLib:
    def __init__(self):
        self.lib = None
        self.play_to_end = None
        self._load()

    def _load(self):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'botzone', 'data')
        so_path = os.path.abspath(os.path.join(base_dir, 'ending_mac.so'))
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library not found: {so_path}. Please compile ending.c into ending_mac.so")
        lib = ctypes.CDLL(so_path)
        lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
        lib.play_to_end.restype = c_int
        self.lib = lib
        self.play_to_end = lib.play_to_end

    def finish_with_endgame_myopp(self, black: int, white: int, side_black_to_move: bool):
        out_my = c_uint64(0)
        out_opp = c_uint64(0)
        if side_black_to_move:
            res = self.play_to_end(c_uint64(black), c_uint64(white), byref(out_my), byref(out_opp))
            ok = (res != PLAY_TO_END_SENTINEL)
            return ok, out_my.value, out_opp.value
        else:
            res = self.play_to_end(c_uint64(white), c_uint64(black), byref(out_my), byref(out_opp))
            ok = (res != PLAY_TO_END_SENTINEL)
            return ok, out_opp.value, out_my.value


END_LIB = EndgameLib()


class NNPolicy:
    def __init__(self, model_path: str):
        self.ai = OthelloAI()
        self.net = Net().cpu().eval()
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            state = torch.load(model_path, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.net.load_state_dict(state, strict=False)

    def choose_move(self, my: int, opp: int) -> int:
        legal = self.ai.get_legal_moves(my, opp)
        if legal == 0:
            return PASS_INDEX
        x = build_input_planes(my, opp, legal)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits.squeeze(0)  # (65,)
        # mask illegal and forbid pass while legal exists
        mask = torch.full((65,), float('-inf'))
        for idx in enumerate_moves(legal):
            mask[idx] = 0.0
        move = int(torch.argmax(logits + mask).item())
        if move == PASS_INDEX or (((legal >> move) & 1) == 0):
            move = enumerate_moves(legal)[0]
        return move


class CEngine:
    def __init__(self, name: str, c_path: str, dylib_path: str):
        self.name = name
        self.c_path = c_path
        self.dylib_path = dylib_path
        self.lib = None

    def ensure_built_and_load(self):
        if not os.path.exists(self.dylib_path):
            # Try to build the shared library
            cmd = f"clang -O3 -std=c11 -shared -fPIC -o '{self.dylib_path}' '{self.c_path}'"
            ret = os.system(cmd)
            if ret != 0 or (not os.path.exists(self.dylib_path)):
                raise RuntimeError(f"Failed to build {self.name} from {self.c_path}")
        lib = ctypes.CDLL(self.dylib_path)
        lib.choose_move.argtypes = [c_uint64, c_uint64]
        lib.choose_move.restype = c_int
        self.lib = lib

    def choose_move(self, my: int, opp: int) -> int:
        mv = int(self.lib.choose_move(c_uint64(my), c_uint64(opp)))
        return mv if mv >= 0 else PASS_INDEX


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
    def add(self, final_black: int, final_white: int, nn_is_black: bool):
        b = popcount(final_black); w = popcount(final_white)
        if b == w: self.d += 1
        elif (b > w) == nn_is_black: self.w += 1
        else: self.l += 1
        self.sum_true += true_score_for_side(nn_is_black, final_black, final_white)
        self.n += 1
    def avg_true(self):
        return (self.sum_true / self.n) if self.n else 0.0


def opening_center_random_if_first_move(ai: OthelloAI, black: int, white: int, side_black_to_move: bool, rng: random.Random):
    # If at the very start (standard init, 4 stones, black to move), choose randomly among the 4 center moves
    if side_black_to_move and black == INIT_BLACK and white == INIT_WHITE:
        legal = ai.get_legal_moves(black, white)
        centers = [m for m in enumerate_moves(legal) if m in CENTER_FIRST_MOVES]
        if centers:
            return rng.choice(centers)
    return None


def play_one_game_vs_engine(nn_policy: NNPolicy, engine: CEngine, nn_is_black: bool, rng: random.Random):
    ai = OthelloAI()
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

        # First move special randomness for black
        if empties == 60 and side_black_to_move:
            mv = opening_center_random_if_first_move(ai, black, white, side_black_to_move, rng)
            if mv is not None:
                new_my, new_opp = ai.make_move(my, opp, mv)
                black, white = (new_my, new_opp) if side_black_to_move else (white, black)
                side_black_to_move = not side_black_to_move
                pass_streak = 0
                continue

        if empties <= 8:
            ok, fb, fw = END_LIB.finish_with_endgame_myopp(black, white, side_black_to_move)
            if ok:
                black, white = fb, fw
                break

        # Midgame selection
        if side_black_to_move:
            mv = nn_policy.choose_move(my, opp) if nn_is_black else engine.choose_move(my, opp)
        else:
            mv = engine.choose_move(my, opp) if nn_is_black else nn_policy.choose_move(my, opp)

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


def play_one_game_vs_greedy(nn_policy: NNPolicy, nn_is_black: bool, rng: random.Random):
    ai = OthelloAI()
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

        # First 6 plies random (total pieces <= 10)
        total_pieces = popcount((black | white) & BOARD_MASK)
        if total_pieces <= 10:
            mv = rng.choice(enumerate_moves(legal))
            if mv != PASS_INDEX:
                new_my, new_opp = ai.make_move(my, opp, mv)
                if side_black_to_move:
                    black, white = new_my, new_opp
                else:
                    white, black = new_my, new_opp
                pass_streak = 0
            else:
                pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            continue

        if empties <= 8:
            ok, fb, fw = END_LIB.finish_with_endgame_myopp(black, white, side_black_to_move)
            if ok:
                black, white = fb, fw
                break

        # Midgame NN vs Greedy
        if side_black_to_move:
            mv = nn_policy.choose_move(my, opp) if nn_is_black else greedy_choose(my, opp)
        else:
            mv = greedy_choose(my, opp) if nn_is_black else nn_policy.choose_move(my, opp)

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
    ap.add_argument('--model', type=str, default=os.path.join('models', 'value_only_best.pt'), help='NN 权重路径')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--games-per-opponent', type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CHASE_SEED'] = str(args.seed)

    nn_policy = NNPolicy(args.model)

    # Four C engines
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cdir = os.path.join(root, 'c')
    engines = [
        ('ajex_noise_3', CEngine('ajex_noise_3', os.path.join(cdir, 'ajex_noise_3.c'), os.path.join(cdir, 'ajex_noise_3.dylib'))),
        ('ajex_noise_5', CEngine('ajex_noise_5', os.path.join(cdir, 'ajex_noise_5.c'), os.path.join(cdir, 'ajex_noise_5.dylib'))),
        ('eva_noise_3',  CEngine('eva_noise_3',  os.path.join(cdir, 'eva_noise_3.c'),  os.path.join(cdir, 'eva_noise_3.dylib'))),
        ('eva_noise_5',  CEngine('eva_noise_5',  os.path.join(cdir, 'eva_noise_5.c'),  os.path.join(cdir, 'eva_noise_5.dylib'))),
    ]
    for _, eng in engines:
        eng.ensure_built_and_load()

    total_stats = Stats()

    for name, eng in engines:
        print(f"\n=== VS {name} ===")
        st_black = Stats()
        st_white = Stats()

        # NN as Black (100)
        for _ in range(args.games_per_opponent // 2):
            fb, fw = play_one_game_vs_engine(nn_policy, eng, nn_is_black=True, rng=random)
            st_black.add(fb, fw, nn_is_black=True)
        # NN as White (100)
        for _ in range(args.games_per_opponent // 2):
            fb, fw = play_one_game_vs_engine(nn_policy, eng, nn_is_black=False, rng=random)
            st_white.add(fb, fw, nn_is_black=False)

        st_all = Stats()
        st_all.w = st_black.w + st_white.w
        st_all.d = st_black.d + st_white.d
        st_all.l = st_black.l + st_white.l
        st_all.sum_true = st_black.sum_true + st_white.sum_true
        st_all.n = st_black.n + st_white.n

        # Print per engine results
        print(f"NN as Black:  W/D/L = {st_black.w}/{st_black.d}/{st_black.l} out of {st_black.n}, avg true = {st_black.avg_true():.3f}")
        print(f"NN as White:  W/D/L = {st_white.w}/{st_white.d}/{st_white.l} out of {st_white.n}, avg true = {st_white.avg_true():.3f}")
        print(f"Overall 200:  W/D/L = {st_all.w}/{st_all.d}/{st_all.l} out of {st_all.n}, avg true = {st_all.avg_true():.3f}")

        # Accumulate to totals
        total_stats.w += st_all.w
        total_stats.d += st_all.d
        total_stats.l += st_all.l
        total_stats.sum_true += st_all.sum_true
        total_stats.n += st_all.n

    print("\n=== OVERALL (4 opponents) ===")
    print(f"Total 800: W/D/L = {total_stats.w}/{total_stats.d}/{total_stats.l} out of {total_stats.n}, avg true = {total_stats.avg_true():.3f}")

    # Extra: vs greedy (30 games, not counted in totals)
    print("\n=== VS greedy (30 games) ===")
    greedy_games = 30
    g_black = Stats(); g_white = Stats()
    for _ in range(greedy_games // 2):
        fb, fw = play_one_game_vs_greedy(nn_policy, nn_is_black=True, rng=random)
        g_black.add(fb, fw, nn_is_black=True)
    for _ in range(greedy_games // 2):
        fb, fw = play_one_game_vs_greedy(nn_policy, nn_is_black=False, rng=random)
        g_white.add(fb, fw, nn_is_black=False)

    g_all = Stats()
    g_all.w = g_black.w + g_white.w
    g_all.d = g_black.d + g_white.d
    g_all.l = g_black.l + g_white.l
    g_all.sum_true = g_black.sum_true + g_white.sum_true
    g_all.n = g_black.n + g_white.n

    print(f"NN as Black:  W/D/L = {g_black.w}/{g_black.d}/{g_black.l} out of {g_black.n}, avg true = {g_black.avg_true():.3f}")
    print(f"NN as White:  W/D/L = {g_white.w}/{g_white.d}/{g_white.l} out of {g_white.n}, avg true = {g_white.avg_true():.3f}")
    print(f"Overall 30:  W/D/L = {g_all.w}/{g_all.d}/{g_all.l} out of {g_all.n}, avg true = {g_all.avg_true():.3f}")


if __name__ == '__main__':
    main()
