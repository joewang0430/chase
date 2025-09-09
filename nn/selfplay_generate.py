# PYTHONPATH=. python nn/selfplay_generate.py --out data/selfplay_value.jsonl --games 50 --seed 1

import sys, os
import json, random, ctypes, time, argparse
from ctypes import c_uint64, c_int, POINTER
from botzone.play import OthelloAI, popcount

# pcs: 当前局面已有棋子数 (落子前)；保证 pcs + empties = 64
# 跳过黑方首手记录，首条样本应为白第一手：初始 4 子 + 黑首手 1 子 => pcs=5, empties=59

PASS_INDEX = 64
BOARD_MASK = (1 << 64) - 1
INIT_BLACK = 0x0000000810000000
INIT_WHITE = 0x0000001008000000

# 加载 C 终局库 (ending.so) 供 play_to_end 使用
class EndgameLib:
    def __init__(self):
        self.lib = None
        self.solve = None
        self.play_to_end = None
        self.load()
    def load(self):
        try:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'botzone', 'data')
            # 使用 macOS 平台编译的库文件 ending_mac.so
            so_path = os.path.abspath(os.path.join(base_dir, 'ending_mac.so'))
            if not os.path.exists(so_path):
                return
            lib = ctypes.CDLL(so_path)
            # int solve_endgame(uint64_t my, uint64_t opp, int* best_move)
            lib.solve_endgame.argtypes = [c_uint64, c_uint64, POINTER(c_int)]
            lib.solve_endgame.restype = c_int
            # int play_to_end(uint64_t black, uint64_t white, uint64_t *final_black, uint64_t *final_white)
            lib.play_to_end.argtypes = [c_uint64, c_uint64, POINTER(c_uint64), POINTER(c_uint64)]
            lib.play_to_end.restype = c_int
            self.lib = lib
            self.solve = lib.solve_endgame
            self.play_to_end = lib.play_to_end
        except Exception:
            self.lib = None
            self.solve = None
            self.play_to_end = None

END_LIB = EndgameLib()

ai = OthelloAI()  # 复用它的合法着生成与落子

def enumerate_moves(bb):
    res = []
    m = bb
    while m:
        l = m & -m
        idx = l.bit_length() - 1
        res.append(idx)
        m ^= l
    return res

def choose_random_move(legal_bb):
    if legal_bb == 0:
        return PASS_INDEX
    moves = enumerate_moves(legal_bb)
    return random.choice(moves)

def simulate_prefix(max_random_moves=200):
    """随机对局直到 empties <= 8 或双方连续 PASS。
    仅记录 empties > 8 的样本，并且跳过黑方第一手（raw_ply=0, empties=60）。
    返回 samples: (raw_ply, my_bb, opp_bb, legal_bb, empties, move_idx)"""
    black = INIT_BLACK
    white = INIT_WHITE
    side_black_to_move = True
    pass_streak = 0
    raw_ply = 0
    samples = []
    while True:
        my = black if side_black_to_move else white
        opp = white if side_black_to_move else black
        occ = (my | opp) & BOARD_MASK
        empties = 64 - popcount(occ)
        if empties == 0:
            break
        legal_bb = ai.get_legal_moves(my, opp)
        move_idx = choose_random_move(legal_bb)
        if empties > 8:
            if not (raw_ply == 0 and side_black_to_move and empties == 60):
                samples.append((raw_ply, my, opp, legal_bb, empties, move_idx))
        if move_idx == PASS_INDEX:
            pass_streak += 1
        else:
            pass_streak = 0
            new_my, new_opp = ai.make_move(my, opp, move_idx)
            if side_black_to_move:
                black, white = new_my, new_opp
            else:
                white, black = new_my, new_opp
        if pass_streak == 2:
            break
        occ = (black | white) & BOARD_MASK
        empties_after = 64 - popcount(occ)
        if empties_after <= 8:
            break
        side_black_to_move = not side_black_to_move
        raw_ply += 1
        if raw_ply >= max_random_moves:
            break
    return black, white, side_black_to_move, samples

def finish_with_endgame(black, white):
    """当剩余 <=8 空格时调用 C 枚举补完对局，返回最终 (final_black, final_white, result_res, diff_norm)。
    result_res: 黑胜 +1 / 白胜 -1 / 平 0
    diff_norm: (黑-白)/64
    若无法使用库且仍有空格，退化：直接数子差为当前终局（不精确）。"""
    occ = (black | white) & BOARD_MASK
    empties = 64 - popcount(occ)
    if empties == 0:
        b_cnt = popcount(black)
        w_cnt = popcount(white)
        res = 1 if b_cnt > w_cnt else (-1 if b_cnt < w_cnt else 0)
        return black, white, res, (b_cnt - w_cnt) / 64.0
    if empties > 8 or END_LIB.play_to_end is None:
        # 不能精确收尾：返回当前局面估值（可能会降低标签质量）
        b_cnt = popcount(black)
        w_cnt = popcount(white)
        res = 1 if b_cnt > w_cnt else (-1 if b_cnt < w_cnt else 0)
        return black, white, res, (b_cnt - w_cnt) / 64.0
    final_black = c_uint64(0)
    final_white = c_uint64(0)
    res = END_LIB.play_to_end(black, white, ctypes.byref(final_black), ctypes.byref(final_white))
    fb = final_black.value
    fw = final_white.value
    b_cnt = popcount(fb)
    w_cnt = popcount(fw)
    diff_norm = (b_cnt - w_cnt) / 64.0
    # res 已经: 黑胜=1 / 白胜=-1 / 平=0
    return fb, fw, res, diff_norm

def write_samples(path, samples, final_res, final_diff):
    """写入样本: pcs = 64 - empties (当前局面已有棋子数, 落子前)。
    保留 raw_ply -> turn 字段以备调试；训练无需依赖 turn。
    value_result / value_diff 按 raw_ply 奇偶翻转（黑=偶）。"""
    with open(path, 'a', encoding='utf-8') as f:
        for raw_ply, my, opp, legal_bb, empties, move_idx in samples:
            is_black_turn = (raw_ply % 2 == 0)
            pcs = 64 - empties  # 当前已有棋子数（未执行该着）
            vr = final_res if is_black_turn else -final_res
            vd = final_diff if is_black_turn else -final_diff
            obj = {
                'pcs': pcs,
                'empties': empties,
                'turn': raw_ply,
                'my_bb': str(my),
                'opp_bb': str(opp),
                'legal_bb': str(legal_bb),
                'policy_target_index': move_idx,
                'value_result': vr,
                'value_diff': round(vd, 6)
            }
            f.write(json.dumps(obj, separators=(',', ':')) + '\n')

def generate(data_path='data/selfplay_value.jsonl', games=20, seed=42, truncate=False):
    if truncate and os.path.exists(data_path):
        os.remove(data_path)
    random.seed(seed)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    t0 = time.time()
    total_samples = 0
    for g in range(games):
        black, white, side_black_to_move, samples = simulate_prefix()
        fb, fw, final_res, final_diff = finish_with_endgame(black, white)
        write_samples(data_path, samples, final_res, final_diff)
        total_samples += len(samples)
        first_pcs = (64 - samples[0][4]) if samples else None
        print(f'game {g:03d} samples={len(samples)} first_pcs={first_pcs} final_res={final_res:+d} diff={final_diff:.3f}')
    dt = time.time() - t0
    print(f'Done. games={games} samples={total_samples} time={dt:.2f}s file={data_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='data/selfplay_value.jsonl')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--truncate', action='store_true', help='先删除已存在输出文件')
    args = ap.parse_args()
    generate(args.out, args.games, args.seed, args.truncate)
