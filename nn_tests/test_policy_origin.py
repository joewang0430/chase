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
# C 端哨兵返回值：仅在空位>8时返回；≤8 时应给出终局
PLAY_TO_END_SENTINEL = -1073741824  # INT_MIN/2

# 终局库加载
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
            return
        lib = ctypes.CDLL(so_path)
        # 新语义：play_to_end(my, opp, &final_my, &final_opp)
        lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
        lib.play_to_end.restype = c_int
        # 单步残局：int solve_endgame(uint64_t my, uint64_t opp, int* best_move)
        lib.solve_endgame.argtypes = [c_uint64, c_uint64, POINTER(c_int)]
        lib.solve_endgame.restype = c_int
        self.lib = lib
        self.play_to_end = lib.play_to_end
        self.solve_endgame = lib.solve_endgame

    def solve_myopp(self, my: int, opp: int):
        """按 my/opp 语义调用收官，返回 (ok, final_my, final_opp)。"""
        if self.play_to_end is None:
            return False, my, opp
        out_my = c_uint64(0)
        out_opp = c_uint64(0)
        res = self.play_to_end(c_uint64(my), c_uint64(opp), byref(out_my), byref(out_opp))
        fmy, fopp = out_my.value, out_opp.value
        # 成功判定：返回值不是哨兵，即接受该终局（即便盘面未占满 64 格）
        ok = (res != PLAY_TO_END_SENTINEL)
        return ok, fmy, fopp

    def solve_one_best_move(self, my: int, opp: int) -> tuple[int, int]:
        """调用 C 端 solve_endgame，返回 (best_move, score)。best_move=-1 表示 PASS/无子可走。"""
        if self.solve_endgame is None:
            return -1, 0
        out_mv = c_int(-1)
        score = int(self.solve_endgame(c_uint64(my), c_uint64(opp), byref(out_mv)))
        return int(out_mv.value), score

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

def finish_with_endgame_naive(black: int, white: int):
    """不区分待走方的原始调用（用于对比检测）。"""
    occ = (black | white) & BOARD_MASK
    empties = 64 - popcount(occ)
    if empties == 0 or END_LIB.play_to_end is None:
        return black, white
    fb = c_uint64(0); fw = c_uint64(0)
    res = END_LIB.play_to_end(c_uint64(black), c_uint64(white), byref(fb), byref(fw))
    return (fb.value or black), (fw.value or white)

def finish_with_endgame_safe(black: int, white: int, side_black_to_move: bool):
    """
    区分待走方的安全调用：
    - 若轮到黑走：按 (black, white) 调用，输出即为最终 (black, white)
    - 若轮到白走：按 (white, black) 调用，再将输出交换回 (black, white)
    返回 (ok, final_black, final_white)
    """
    occ = (black | white) & BOARD_MASK
    empties = 64 - popcount(occ)
    if empties == 0 or END_LIB.play_to_end is None:
        return False, black, white
    if side_black_to_move:
        fb = c_uint64(0); fw = c_uint64(0)
        res = END_LIB.play_to_end(c_uint64(black), c_uint64(white), byref(fb), byref(fw))
        if res:
            return True, fb.value, fw.value
        return False, black, white
    else:
        t1 = c_uint64(0); t2 = c_uint64(0)
        res = END_LIB.play_to_end(c_uint64(white), c_uint64(black), byref(t1), byref(t2))
        if res:
            # t1=最终白，t2=最终黑
            return True, t2.value, t1.value
        return False, black, white

def finish_with_endgame_myopp(black: int, white: int, side_black_to_move: bool):
    """使用 my/opp 语义的统一收官调用。返回 (ok, final_black, final_white)。"""
    if side_black_to_move:
        ok, fmy, fopp = END_LIB.solve_myopp(black, white)
        return ok, fmy, fopp
    else:
        ok, fmy, fopp = END_LIB.solve_myopp(white, black)
        return ok, fopp, fmy

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

def outcome_for_nn(nn_is_black: bool, final_black: int, final_white: int) -> str:
    """
    返回单局结果：'W'（胜）/'D'（平）/'L'（负）
    判定只看棋子数大小（不含空格）。
    """
    b = popcount(final_black); w = popcount(final_white)
    if b == w:
        return 'D'
    return 'W' if ((b > w) == nn_is_black) else 'L'

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

def play_one_game(nn_policy: NNPolicy, engine, nn_is_black: bool, rng: random.Random,
                  check_end_bias: bool = False, end_bias_counters: dict | None = None):
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

        # 进入残局：双方交给枚举（优先用统一的 my/opp 语义；失败则继续对弈）
        if empties <= 8:
            # 统一路径
            ok, fb, fw = finish_with_endgame_myopp(black, white, side_black_to_move)
            if ok:
                black, white = fb, fw
                break

        # 决策
        if side_black_to_move == nn_is_black:
            # NN 执黑且首步 → 脚本随机 4 选 1（而非用 NN）
            if nn_is_black and raw_ply == 0 and side_black_to_move:
                mv = random_black_first_move(ai, black, white, rng)
            else:
                mv = nn_policy.choose_move(my, opp, ai)
        else:
            # 对手：若为整盘第一手（黑方首手）且对手是 greedy，则脚本代为 4 选 1 随机
            if raw_ply == 0 and side_black_to_move and engine == 'greedy':
                mv = random_black_first_move(ai, black, white, rng)
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

def endgame_by_per_move_solve(black: int, white: int, side_black_to_move: bool):
    """从给定残局位置（≤8 空）开始，使用 C 端 solve_endgame 连续逐步对弈至终局，返回 (final_black, final_white)。"""
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
        mv, _score = END_LIB.solve_one_best_move(my, opp)
        if mv < 0:
            # PASS（保险分支，按上方 legal==0 理论上已处理）
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

def play_one_game_greedy_dual_endgame(nn_policy: NNPolicy, nn_is_black: bool, rng: random.Random,
                                      mismatch_counter: dict):
    """与 greedy 对局一盘：当进入 ≤8 空位时，同时：
    - 用 play_to_end 一步收尾
    - 用 solve_endgame 连续逐步对弈收尾
    验证两种方式最终棋盘一致。
    返回 (final_black, final_white)。"""
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
            pass_streak += 1
            if pass_streak == 2:
                break
            side_black_to_move = not side_black_to_move
            raw_ply += 1
            continue

        if empties <= 8:
            # 快速收官
            b0, w0, s0 = black, white, side_black_to_move
            ok, fb1, fw1 = finish_with_endgame_myopp(b0, w0, s0)
            if not ok:
                # 意外失败：退回常规对弈直至终局
                pass
            else:
                # 逐步调用 solve_endgame 收官
                fb2, fw2 = endgame_by_per_move_solve(b0, w0, s0)
                mismatch_counter['samples'] = mismatch_counter.get('samples', 0) + 1
                if (fb1 & BOARD_MASK) != (fb2 & BOARD_MASK) or (fw1 & BOARD_MASK) != (fw2 & BOARD_MASK):
                    mismatch_counter['mismatch'] = mismatch_counter.get('mismatch', 0) + 1
                black, white = fb1, fw1
                break

        # 中盘走子
        if side_black_to_move == nn_is_black:
            if nn_is_black and raw_ply == 0 and side_black_to_move:
                mv = random_black_first_move(ai, black, white, rng)
            else:
                mv = nn_policy.choose_move(my, opp, ai)
        else:
            # greedy 对手
            if raw_ply == 0 and side_black_to_move:
                mv = random_black_first_move(ai, black, white, rng)
            else:
                mv = greedy_choose(my, opp)

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
    ap.add_argument('--check-end-bias', action='store_true', help='检测残局收尾 naive/safe 差异与是否偏向领先一方')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CHASE_SEED'] = str(args.seed)  # 供 C 引擎内部 RNG 复现

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    nn_policy = NNPolicy(args.model, device)
    end_bias_counters = {} if args.check_end_bias else None

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
            fb, fw = play_one_game(nn_policy, eng, nn_is_black=True, rng=random,
                                   check_end_bias=args.check_end_bias, end_bias_counters=end_bias_counters)
            st_black.add(fb, fw, nn_is_black=True)
        # 白方 100
        for i in range(args.games_per_opponent // 2):
            fb, fw = play_one_game(nn_policy, eng, nn_is_black=False, rng=random,
                                   check_end_bias=args.check_end_bias, end_bias_counters=end_bias_counters)
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

    # 另外：与 greedy 基线 30 局（默认；不计入总计），黑白各 15
    print('\n=== VS greedy (30 games) ===')
    greedy_games = 30
    g_black = Stats(); g_white = Stats()
    dual_counter = {'samples': 0, 'mismatch': 0}
    for _ in range(greedy_games // 2):
        fb, fw = play_one_game_greedy_dual_endgame(nn_policy, nn_is_black=True, rng=random,
                                                   mismatch_counter=dual_counter)
        g_black.add(fb, fw, nn_is_black=True)
    for _ in range(greedy_games // 2):
        fb, fw = play_one_game_greedy_dual_endgame(nn_policy, nn_is_black=False, rng=random,
                                                   mismatch_counter=dual_counter)
        g_white.add(fb, fw, nn_is_black=False)
    g_all = Stats()
    g_all.w = g_black.w + g_white.w
    g_all.d = g_black.d + g_white.d
    g_all.l = g_black.l + g_white.l
    g_all.sum_true = g_black.sum_true + g_white.sum_true
    g_all.n = g_black.n + g_white.n

    if args.check_end_bias and end_bias_counters:
        s = end_bias_counters.get('samples', 0)
        m = end_bias_counters.get('mismatch', 0)
        f = end_bias_counters.get('favors_leader', 0)
        print(f'\n[end-bias summary] samples={s}, naive_vs_safe_mismatch={m}, favor_leader_in_mismatch={f}')
    print(f'NN as Black vs greedy: W/D/L = {g_black.w}/{g_black.d}/{g_black.l} out of {g_black.n}, avg true = {g_black.avg_true():.3f}')
    print(f'NN as White vs greedy: W/D/L = {g_white.w}/{g_white.d}/{g_white.l} out of {g_white.n}, avg true = {g_white.avg_true():.3f}')
    print(f'Overall vs greedy:     W/D/L = {g_all.w}/{g_all.d}/{g_all.l} out of {g_all.n}, avg true = {g_all.avg_true():.3f}')
    print(f"[dual-endgame consistency] samples={dual_counter['samples']}, mismatches={dual_counter.get('mismatch',0)}")

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