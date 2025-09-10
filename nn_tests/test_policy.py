# PYTHONPATH=. python3 nn_tests/test_policy.py --model models/value_only_best.pt --seed 1
# or other model path

import os
import sys
import time
import json
import math
import argparse
import random
import subprocess
import ctypes
from ctypes import c_uint64, c_int, POINTER, byref

import torch
import torch.nn.functional as F

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from botzone.play import OthelloAI, popcount
from nn.model_def import Net, build_input_planes
from utils.greedy import choose_move as greedy_choose

# 常量
PASS_INDEX = 64
BOARD_MASK = (1 << 64) - 1
INIT_BLACK = 0x0000000810000000
INIT_WHITE = 0x0000001008000000

# 终局库加载
class EndgameLib:
    def __init__(self):
        self.lib = None
        self.play_to_end = None
        self._load()

    def _load(self):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'botzone', 'data')
        so_path = os.path.abspath(os.path.join(base_dir, 'ending_mac.so'))
        if not os.path.exists(so_path):
            return
        lib = ctypes.CDLL(so_path)
        lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
        lib.play_to_end.restype = c_int
        self.lib = lib
        self.play_to_end = lib.play_to_end

END_LIB = EndgameLib()

# C 引擎封装（通过 choose_move(my, opp)）
class CEngine:
    def __init__(self, name: str, c_path: str, out_lib: str):
        self.name = name
        self.c_path = c_path
        self.out_lib = out_lib
        self.lib = None
        self.fn = None

    def ensure_built_and_load(self):
        # 若目标 .dylib 不存在或比源文件旧，则编译
        need_build = (not os.path.exists(self.out_lib)) or \
                     (os.path.getmtime(self.out_lib) < os.path.getmtime(self.c_path))
        if need_build:
            cmd = [
                "clang", "-O3", "-std=c11", "-shared", "-fPIC",
                self.c_path, "-o", self.out_lib
            ]
            print(f"[build] {' '.join(cmd)}")
            subprocess.check_call(cmd)
        lib = ctypes.CDLL(self.out_lib)
        fn = lib.choose_move
        fn.argtypes = [c_uint64, c_uint64]
        fn.restype = c_int
        self.lib = lib
        self.fn = fn

    def choose_move(self, my: int, opp: int) -> int:
        if self.fn is None:
            self.ensure_built_and_load()
        r = int(self.fn(c_uint64(my), c_uint64(opp)))
        return PASS_INDEX if r < 0 else r

# 工具
def enumerate_moves(bb: int):
    res = []
    m = bb
    while m:
        l = m & -m
        res.append(l.bit_length() - 1)
        m ^= l
    return res

def random_black_first_move(ai: OthelloAI, black: int, white: int, rng: random.Random) -> int:
    legal = ai.get_legal_moves(black, white)
    moves = enumerate_moves(legal)
    # 标准开局应恰有 4 手；容错：若不为 4，仍然随机从全部合法中选一
    if not moves:
        return PASS_INDEX
    return rng.choice(moves)

def finish_with_endgame(black: int, white: int):
    """调用 ending_mac.so 的 play_to_end 完成残局（empties<=8）。失败则降级返回当前局面。"""
    occ = (black | white) & BOARD_MASK
    empties = 64 - popcount(occ)
    if empties == 0 or END_LIB.play_to_end is None:
        return black, white
    fb = c_uint64(0); fw = c_uint64(0)
    res = END_LIB.play_to_end(c_uint64(black), c_uint64(white), byref(fb), byref(fw))
    # 当库因为阈值返回哨兵或异常，这里仍然使用当前局面
    return (fb.value or black), (fw.value or white)

def true_score_for_nn(nn_is_black: bool, final_black: int, final_white: int) -> int:
    """
    返回得分真值 = 我方(NN)得分 - 对方得分；
    胜方将全部空格计入其子数，负方不计；平局为 0。
    """
    b = popcount(final_black); w = popcount(final_white)
    empties = 64 - b - w
    if b > w:
        # 黑胜
        return (b + empties - w) if nn_is_black else (w - (b + empties))
    elif w > b:
        # 白胜
        return (w + empties - b) if not nn_is_black else (b - (w + empties))
    else:
        # 平局
        return 0

class NNPolicy:
    def __init__(self, model_path: str | None, device: torch.device):
        self.device = device
        self.model = Net().to(device).eval()
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            state = ckpt.get('model_state', ckpt)
            self.model.load_state_dict(state, strict=False)

    def choose_move(self, my: int, opp: int, ai: OthelloAI) -> int:
        legal = ai.get_legal_moves(my, opp)
        if legal == 0:
            return PASS_INDEX
        x = build_input_planes(my, opp, legal).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)  # (1,65)
        logits = logits[0].clone()
        # mask 非法位与 PASS（有合法时禁止 PASS）
        for i in range(64):
            if not ((legal >> i) & 1):
                logits[i] = -1e9
        logits[64] = -1e9
        idx = int(torch.argmax(logits).item())
        return idx

class Stats:
    def __init__(self):
        self.w = 0; self.d = 0; self.l = 0
        self.sum_true = 0.0
        self.n = 0
    def add(self, final_black: int, final_white: int, nn_is_black: bool):
        b = popcount(final_black); w = popcount(final_white)
        if b > w:
            win = 'black'
        elif w > b:
            win = 'white'
        else:
            win = 'draw'
        if win == 'draw':
            self.d += 1
        elif (win == 'black') == nn_is_black:
            self.w += 1
        else:
            self.l += 1
        self.sum_true += true_score_for_nn(nn_is_black, final_black, final_white)
        self.n += 1
    def avg_true(self):
        return (self.sum_true / self.n) if self.n else 0.0

def play_one_game(nn_policy: NNPolicy, engine, nn_is_black: bool, rng: random.Random):
    """
    engine: CEngine 或字符串 'greedy'
    返回 (final_black, final_white)
    """
    ai = OthelloAI()
    black = INIT_BLACK
    white = INIT_WHITE
    side_black_to_move = True
    pass_streak = 0
    raw_ply = 0

    while True:
        my = black if side_black_to_move else white
        opp = white if side_black_to_move else black
        occ = (my | opp) & BOARD_MASK
        empties = 64 - popcount(occ)

        if empties == 0:
            break

        legal = ai.get_legal_moves(my, opp)
        if legal == 0:
            # PASS
            pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            raw_ply += 1
            continue

        # 进入残局：双方交给枚举
        if empties <= 8:
            black, white = finish_with_endgame(black, white)
            break

        # 决策
        if side_black_to_move == nn_is_black:
            # NN 执黑且首步 → 脚本随机 4 选 1（而非用 NN）
            if nn_is_black and raw_ply == 0 and side_black_to_move:
                mv = random_black_first_move(ai, black, white, rng)
            else:
                mv = nn_policy.choose_move(my, opp, ai)
        else:
            if engine == 'greedy':
                mv = greedy_choose(my, opp)
            else:
                mv = engine.choose_move(my, opp)

        # 应用
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
        raw_ply += 1

    return black, white

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None, help='NN 权重路径（可选）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--games-per-opponent', type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CHASE_SEED'] = str(args.seed)  # 供 C 引擎内部 RNG 复现

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    nn_policy = NNPolicy(args.model, device)

    # 四个 C 引擎
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cdir = os.path.join(root, 'c')
    engines = [
        ('ajex_noise_3', CEngine('ajex_noise_3', os.path.join(cdir, 'ajex_noise_3.c'), os.path.join(cdir, 'ajex_noise_3.dylib'))),
        ('ajex_noise_5', CEngine('ajex_noise_5', os.path.join(cdir, 'ajex_noise_5.c'), os.path.join(cdir, 'ajex_noise_5.dylib'))),
        ('eva_noise_3',  CEngine('eva_noise_3',  os.path.join(cdir, 'eva_noise_3.c'),  os.path.join(cdir, 'eva_noise_3.dylib'))),
        ('eva_noise_5',  CEngine('eva_noise_5',  os.path.join(cdir, 'eva_noise_5.c'),  os.path.join(cdir, 'eva_noise_5.dylib'))),
    ]
    # 先加载/编译
    for _, eng in engines:
        eng.ensure_built_and_load()

    total_stats = Stats()
    per_engine_results = {}

    for name, eng in engines:
        print(f'\n=== VS {name} ===')
        st_black = Stats()
        st_white = Stats()
        # 黑方 100
        for i in range(args.games_per_opponent // 2):
            fb, fw = play_one_game(nn_policy, eng, nn_is_black=True, rng=random)
            st_black.add(fb, fw, nn_is_black=True)
        # 白方 100
        for i in range(args.games_per_opponent // 2):
            fb, fw = play_one_game(nn_policy, eng, nn_is_black=False, rng=random)
            st_white.add(fb, fw, nn_is_black=False)

        st_all = Stats()
        st_all.w = st_black.w + st_white.w
        st_all.d = st_black.d + st_white.d
        st_all.l = st_black.l + st_white.l
        st_all.sum_true = st_black.sum_true + st_white.sum_true
        st_all.n = st_black.n + st_white.n

        per_engine_results[name] = (st_black, st_white, st_all)

        # 累计总计
        total_stats.w += st_all.w
        total_stats.d += st_all.d
        total_stats.l += st_all.l
        total_stats.sum_true += st_all.sum_true
        total_stats.n += st_all.n

        print(f'NN as Black:  W/D/L = {st_black.w}/{st_black.d}/{st_black.l} out of {st_black.n}, avg true = {st_black.avg_true():.3f}')
        print(f'NN as White:  W/D/L = {st_white.w}/{st_white.d}/{st_white.l} out of {st_white.n}, avg true = {st_white.avg_true():.3f}')
        print(f'Overall 200:  W/D/L = {st_all.w}/{st_all.d}/{st_all.l} out of {st_all.n}, avg true = {st_all.avg_true():.3f}')

    print('\n=== OVERALL (4 opponents) ===')
    print(f'Total 800: W/D/L = {total_stats.w}/{total_stats.d}/{total_stats.l} out of {total_stats.n}, avg true = {total_stats.avg_true():.3f}')

    # 另外：与 greedy 基线 2 局（不计入总计）
    print('\n=== VS greedy (2 games, report true score only) ===')
    fb, fw = play_one_game(nn_policy, 'greedy', nn_is_black=True, rng=random)
    ts_black = true_score_for_nn(True, fb, fw)
    fb2, fw2 = play_one_game(nn_policy, 'greedy', nn_is_black=False, rng=random)
    ts_white = true_score_for_nn(False, fb2, fw2)
    print(f'NN (Black) vs greedy true score = {ts_black:+d}')
    print(f'NN (White) vs greedy true score = {ts_white:+d}')

    # 详细逐对手汇总（可选 JSON）
    summary = {
        'per_opponent': {
            name: {
                'black': {'W': st_b.w, 'D': st_b.d, 'L': st_b.l, 'avg_true': round(st_b.avg_true(), 3)},
                'white': {'W': st_w.w, 'D': st_w.d, 'L': st_w.l, 'avg_true': round(st_w.avg_true(), 3)},
                'overall': {'W': st_all.w, 'D': st_all.d, 'L': st_all.l, 'avg_true': round(st_all.avg_true(), 3)}
            }
            for name, (st_b, st_w, st_all) in per_engine_results.items()
        },
        'overall_800': {'W': total_stats.w, 'D': total_stats.d, 'L': total_stats.l, 'avg_true': round(total_stats.avg_true(), 3)},
        'greedy': {'nn_black_true': ts_black, 'nn_white_true': ts_white}
    }
    print('\nSUMMARY JSON:')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()