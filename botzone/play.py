import os
import sys
import json
import time
import ctypes
from ctypes import c_uint64, c_int, POINTER, byref
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 添加 Python 3.6 兼容 popcount (替换 int.bit_count)
POPCNT8 = [bin(i).count('1') for i in range(256)]
ENDING_SO_NAME = "ending.so"
# 需要加载的 3 个（新结构）模型文件名（位于本目录 data/ 下）
NN_NAME_EARLY_128 = "chase_early_128.pt"
NN_NAME_MID_128   = "chase_mid_128.pt"
NN_NAME_LATE_64   = "chase_late_64.pt"
NN_ALL = [NN_NAME_EARLY_128, NN_NAME_MID_128, NN_NAME_LATE_64]

START_TIME = 0

# --------------- 全局位棋盘常量与基础函数（从 OthelloAI 中抽出便于复用） ---------------
FILE_A       = 0x0101010101010101
FILE_H       = 0x8080808080808080
NOT_FILE_A   = 0xFFFFFFFFFFFFFFFF ^ FILE_A
NOT_FILE_H   = 0xFFFFFFFFFFFFFFFF ^ FILE_H
FULL_MASK    = 0xFFFFFFFFFFFFFFFF

def popcount(x: int) -> int:
    c = 0
    while x:
        c += POPCNT8[x & 0xFF]
        x >>= 8
    return c

# 坐标转换顶层函数（统一风格）
def xy_to_bit(x: int, y: int) -> int:
    return x * 8 + y

def bit_to_xy(bit_pos: int):
    return bit_pos // 8, bit_pos % 8

def set_bit(bitboard: int, x: int, y: int) -> int:
    return bitboard | (1 << (x * 8 + y))

# 位移操作（不绑定实例，纯函数）
def shift_e (bb: int) -> int: return (bb & NOT_FILE_H) << 1 & FULL_MASK
def shift_w (bb: int) -> int: return (bb & NOT_FILE_A) >> 1
def shift_n (bb: int) -> int: return bb >> 8
def shift_s (bb: int) -> int: return (bb << 8) & FULL_MASK
def shift_ne(bb: int) -> int: return (bb & NOT_FILE_H) >> 7
def shift_nw(bb: int) -> int: return (bb & NOT_FILE_A) >> 9
def shift_se(bb: int) -> int: return ((bb & NOT_FILE_H) << 9) & FULL_MASK
def shift_sw(bb: int) -> int: return ((bb & NOT_FILE_A) << 7) & FULL_MASK

# 旋转/镜像工具（内联，避免平台导入依赖）
class Rotators:
    @staticmethod
    def up_diag_mirror(me: int, opp: int) -> Tuple[int, int]:
        """
        反对角线（↗）镜像：(r,c)->(7-c,7-r)
        """
        def _mirror(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = (7 - c) * 8 + (7 - r)
                res |= (1 << j)
                x ^= lsb
            return res
        return _mirror(me), _mirror(opp)

    @staticmethod
    def down_diag_mirror(me: int, opp: int) -> Tuple[int, int]:
        """
        主对角线（↘）镜像：(r,c)->(c,r)
        """
        def _mirror(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = c * 8 + r
                res |= (1 << j)
                x ^= lsb
            return res
        return _mirror(me), _mirror(opp)

    @staticmethod
    def rotation_180(me: int, opp: int) -> Tuple[int, int]:
        """
        180° 旋转：(r,c)->(7-r,7-c)
        """
        def _rot(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = (7 - r) * 8 + (7 - c)
                res |= (1 << j)
                x ^= lsb
            return res
        return _rot(me), _rot(opp)

def legal_moves(me: int, opp: int) -> int:
    """生成 me 的合法着点位棋盘（位=1 表示可落子）。"""
    empty = ~(me | opp) & FULL_MASK
    if empty == 0:
        return 0
    moves = 0
    # East
    x = shift_e(me) & opp; acc = 0
    while x:
        nxt = shift_e(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # West
    x = shift_w(me) & opp; acc = 0
    while x:
        nxt = shift_w(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # North
    x = shift_n(me) & opp; acc = 0
    while x:
        nxt = shift_n(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # South
    x = shift_s(me) & opp; acc = 0
    while x:
        nxt = shift_s(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # NE
    x = shift_ne(me) & opp; acc = 0
    while x:
        nxt = shift_ne(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # NW
    x = shift_nw(me) & opp; acc = 0
    while x:
        nxt = shift_nw(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # SE
    x = shift_se(me) & opp; acc = 0
    while x:
        nxt = shift_se(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    # SW
    x = shift_sw(me) & opp; acc = 0
    while x:
        nxt = shift_sw(x); acc |= nxt; x = nxt & opp
    moves |= acc & empty
    return moves

def apply_move(me: int, opp: int, pos: int):
    """对 me 在坐标 pos (0..63) 落子；返回 (new_me, new_opp)。pos<0 表示 PASS（交换走子方）。"""
    if pos < 0:
        return opp, me  # pass：交换
    move_bit = 1 << pos
    if (me | opp) & move_bit:
        return me, opp  # 非法/已占；原样返回（调用方可选择忽略）
    flips = 0
    # East
    cur = shift_e(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_e(cur)
    if cur & me: flips |= line
    # West
    cur = shift_w(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_w(cur)
    if cur & me: flips |= line
    # North
    cur = shift_n(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_n(cur)
    if cur & me: flips |= line
    # South
    cur = shift_s(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_s(cur)
    if cur & me: flips |= line
    # NE
    cur = shift_ne(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_ne(cur)
    if cur & me: flips |= line
    # NW
    cur = shift_nw(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_nw(cur)
    if cur & me: flips |= line
    # SE
    cur = shift_se(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_se(cur)
    if cur & me: flips |= line
    # SW
    cur = shift_sw(move_bit); line = 0
    while cur & opp:
        line |= cur; cur = shift_sw(cur)
    if cur & me: flips |= line
    if flips == 0:
        return me, opp  # 无翻子：非法位置，忽略
    me |= move_bit | flips
    opp &= ~flips
    return me, opp

# @@ NN part begins
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = self.c1(x)
        y = self.b1(y)
        y = F.relu(y, inplace=True)
        y = self.c2(y)
        y = self.b2(y)
        return F.relu(x + y, inplace=True)


class NetBase(nn.Module):
    """Base net: 8 residual blocks, parameterized channels; heads produce 64 logits and 1 value.
    Matches the state_dict layout saved by nn/model_def_128.py and nn/model_def_96.py.
    """
    def __init__(self, channels):
        super(NetBase, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(8)])
        # Policy head
        self.p_c = nn.Conv2d(channels, 8, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(8)
        self.p_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.p_fc2 = nn.Linear(256, 64)
        # Value head
        self.v_c = nn.Conv2d(channels, 4, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(4)
        self.v_fc1 = nn.Linear(4 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 128)
        self.v_fc3 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()
        # Kaiming init similar to training code
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        # Policy
        p = F.relu(self.p_bn(self.p_c(x)), inplace=True)
        p = p.view(p.size(0), -1)
        p = F.relu(self.p_fc1(p), inplace=True)
        policy_logits = self.p_fc2(p)
        # Value
        v = F.relu(self.v_bn(self.v_c(x)), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v), inplace=True)
        v = F.relu(self.v_fc2(v), inplace=True)
        value = self.tanh(self.v_fc3(v))
        return policy_logits, value


class Net128(NetBase):
    def __init__(self):
        super(Net128, self).__init__(channels=128)


class Net64(NetBase):
    def __init__(self):
        super(Net64, self).__init__(channels=64)
# @@ NN part ends


# Removed numpy; use pure Python integers for bitboards.
class OthelloAI:
    def __init__(self):
        self.my_pieces = 0
        self.opp_pieces = 0
        self.my_color = 1
        # --- Endgame solver handles ---
        self.endgame_lib = None
        self.endgame_fn = None
        self.last_endgame_value = None  # 终局搜索返回的差值 (my - opp)
        self.last_endgame_time_ms = None  # 本次 solve_endgame 调用耗时 ms
        # 可配置阈值
        self.ENDGAME_MOVE_THRESHOLD = 16   # <=16 直接用枚举求精确着法
        self.ENDGAME_VALUE_THRESHOLD = 10  # MCTS 评估截点
        # 调试信息
        self.endgame_error = None  # 记录 ctypes 加载 / 调用失败原因
        # 开局棋谱（full_book）
        self.book_loaded = False
        self.book = {}
        self.book_path = None
        self.book_error = None
        self.book_used_move = False

    # @@ NN part begins
        # 三个模型（前两段 128 通道，末段 96 通道）
        self.nn_models = [None, None, None]  # type: ignore
        self.nn_loaded = False
        self.nn_total_load_ms = None  # 三个模型整体加载耗时
        self.nn_errors = [None, None, None]

    def load_nn(self):
        if self.nn_loaded:
            return
        self.nn_loaded = True
        t0 = time.perf_counter()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for idx, fname in enumerate(NN_ALL):
            try:
                paths = [
                    os.path.join(base_dir, 'data', fname),
                    os.path.join(base_dir, fname),
                ]
                path = next((p for p in paths if os.path.exists(p)), None)
                if path is None:
                    self.nn_errors[idx] = 'not_found'
                    continue
                state = torch.load(path, map_location='cpu')
                # 选择对应结构
                if idx == 0 or idx == 1:
                    net = Net128()
                else:
                    net = Net64()
                net.load_state_dict(state, strict=True)
                net.eval()
                self.nn_models[idx] = net
                self.nn_errors[idx] = None
            except Exception as e:
                msg = str(e)
                if len(msg) > 100:
                    msg = msg[:97] + '...'
                self.nn_errors[idx] = f"{type(e).__name__}:{msg}"
        self.nn_total_load_ms = (time.perf_counter() - t0) * 1000.0
    # @@ NN part ends

    # --- 开局棋谱加载 ---
    def load_book(self):
        if self.book_loaded:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, 'data', 'full_book.jsonl')
        self.book_path = path
        if not os.path.exists(path):
            self.book_error = 'not_found'
            self.book_loaded = True
            return
        loaded = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = obj.get('key')
                    moves = obj.get('best_moves')
                    if isinstance(key, str) and isinstance(moves, list) and moves:
                        mv_list = [m for m in moves if isinstance(m, str) and m]
                        if mv_list:
                            self.book[key] = mv_list
                            loaded += 1
            self.book_error = None
        except Exception as e:
            self.book_error = f"load:{type(e).__name__}"
        self.book_loaded = True
        self.book_loaded_count = loaded

    def _transform_pos(self, mode: str, pos: int) -> int:
        """对单个格点 index (0..63) 应用与棋盘相同的对称变换。"""
        r, c = divmod(pos, 8)
        if mode == 'identity':
            nr, nc = r, c
        elif mode == 'up':  # (r,c)->(7-c,7-r)
            nr, nc = 7 - c, 7 - r
        elif mode == 'down':  # (r,c)->(c,r)
            nr, nc = c, r
        elif mode == 'rot180':  # (r,c)->(7-r,7-c)
            nr, nc = 7 - r, 7 - c
        else:
            nr, nc = r, c
        return nr * 8 + nc

    # --- 开局棋谱查询（仅前 8 步 / 前 12 颗子） ---
    def try_book_move(self):
        # 先判断是否在前 8 步（≤12 子）；否则直接跳过，不加载棋谱
        discs = popcount(self.my_pieces | self.opp_pieces)
        if discs > 12:
            return None
        # 仅在需要使用棋谱时才尝试加载
        self.load_book()
        if self.book_error is not None or not self.book:
            return None
        # 以真实颜色构造 (black, white) 位棋盘
        if self.my_color == 1:
            black_bb = self.my_pieces
            white_bb = self.opp_pieces
        else:
            black_bb = self.opp_pieces
            white_bb = self.my_pieces
        turn_letter = 'B' if self.my_color == 1 else 'W'

        # 使用本文件内联的 Rotators（无外部导入）
        variants = []
        variants.append((black_bb, white_bb, 'identity'))
        b1, w1 = Rotators.up_diag_mirror(black_bb, white_bb)
        variants.append((b1, w1, 'up'))
        b2, w2 = Rotators.down_diag_mirror(black_bb, white_bb)
        variants.append((b2, w2, 'down'))
        b3, w3 = Rotators.rotation_180(black_bb, white_bb)
        variants.append((b3, w3, 'rot180'))

        for b_t, w_t, mode in variants:
            key = f"{b_t:016x}_{w_t:016x}_{turn_letter}"
            # 查找生成的 key
            moves_list = self.book.get(key)
            if not moves_list:
                continue
            # 随机选择一个（未来 best_moves 可能有多个）
            mv = random.choice(moves_list)
            if not isinstance(mv, str) or len(mv) < 2:
                continue
            ms = mv.lower()
            if ms == 'pass':
                self.book_used_move = True
                return (-1, -1)
            col_ch = ms[0]
            row_part = ms[1:]
            if col_ch < 'a' or col_ch > 'h':
                continue
            try:
                row_num = int(row_part)
            except ValueError:
                continue
            if row_num < 1 or row_num > 8:
                continue
            col_idx = ord(col_ch) - ord('a')
            row_idx = row_num - 1
            pos_canonical = row_idx * 8 + col_idx
            pos_actual = self._transform_pos(mode, pos_canonical)
            ar, ac = divmod(pos_actual, 8)
            self.book_used_move = True
            return ar, ac
        return None

    # --- 坐标与位操作 ---
    # bit_to_xy / set_bit 已抽到模块级函数

    # --- 初始棋盘 ---
    def init_standard_board(self, my_color):
        self.my_color = my_color
        if my_color == 1:  # 黑方
            self.my_pieces = set_bit(0, 3, 4) | set_bit(0, 4, 3)
            self.opp_pieces = set_bit(0, 3, 3) | set_bit(0, 4, 4)
        else:  # 白方
            self.my_pieces = set_bit(0, 3, 3) | set_bit(0, 4, 4)
            self.opp_pieces = set_bit(0, 3, 4) | set_bit(0, 4, 3)

    # （位移 / 合法着点 / 落子 已抽取为模块级函数）

    # --- 历史重建 ---
    def get_current_board(self, requests, responses):
        if requests and requests[0]["x"] < 0:
            self.my_color = 1
        else:
            self.my_color = -1
        self.init_standard_board(self.my_color)
        turn_count = len(responses)
        if self.my_color == 1:  # 黑方先手
            for i in range(turn_count):
                mv = responses[i]
                if mv["x"] >= 0:
                    self.my_pieces, self.opp_pieces = apply_move(self.my_pieces, self.opp_pieces, xy_to_bit(mv["x"], mv["y"]))
                if i + 1 < len(requests):
                    opp_mv = requests[i + 1]
                    if opp_mv["x"] >= 0:
                        self.opp_pieces, self.my_pieces = apply_move(self.opp_pieces, self.my_pieces, xy_to_bit(opp_mv["x"], opp_mv["y"]))
        else:  # 白方后手
            for i in range(turn_count):
                opp_mv = requests[i]
                if opp_mv["x"] >= 0:
                    self.opp_pieces, self.my_pieces = apply_move(self.opp_pieces, self.my_pieces, xy_to_bit(opp_mv["x"], opp_mv["y"]))
                my_mv = responses[i]
                if my_mv["x"] >= 0:
                    self.my_pieces, self.opp_pieces = apply_move(self.my_pieces, self.opp_pieces, xy_to_bit(my_mv["x"], my_mv["y"]))
            if turn_count < len(requests):
                cur = requests[turn_count]
                if cur["x"] >= 0:
                    self.opp_pieces, self.my_pieces = apply_move(self.opp_pieces, self.my_pieces, xy_to_bit(cur["x"], cur["y"]))

    # --- Botzone 交互（坐标入站交换 / 出站还原） ---
    def parse_and_convert(self, raw_json_str):
        data = json.loads(raw_json_str) if raw_json_str else {"requests":[],"responses":[]}
        def swap_list(lst):
            out = []
            for mv in lst:
                if mv.get("x", -1) < 0:
                    out.append({"x": -1, "y": -1})
                else:
                    out.append({"x": mv["y"], "y": mv["x"]})  # 交换 行列
            return out
        return swap_list(data.get("requests", [])), swap_list(data.get("responses", []))
    
    # --- C语言部分 ---
    def load_endgame(self):
        """懒加载 ending.so (部署版本). 若不存在则静默回退。"""
        if self.endgame_fn is not None or self.endgame_lib is not None:
            return
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            so_path = os.path.join(base_dir, "data", ENDING_SO_NAME)
            if not os.path.exists(so_path):
                self.endgame_error = "not_found"
                return
            lib = ctypes.CDLL(so_path)
            fn = lib.solve_endgame
            fn.argtypes = [c_uint64, c_uint64, ctypes.POINTER(c_int)]
            fn.restype = c_int
            self.endgame_lib = lib
            self.endgame_fn = fn
            self.endgame_error = None
        except Exception as e:
            self.endgame_lib = None
            self.endgame_fn = None
            self.endgame_error = f"load:{type(e).__name__}"

    def endgame_search(self):
        """调用 C 终局枚举：返回 (x,y, value)
        x,y 为内部 (行,列)；value 为 (my - opp) 完全搜索分差。
        best_move = -1 → PASS
        """
        self.load_endgame()
        if self.endgame_fn is None:
            return None  # 没有可用终局库
        try:
            best_pos_c = c_int(-1)
            t0 = time.perf_counter()
            score = self.endgame_fn(self.my_pieces, self.opp_pieces, byref(best_pos_c))
            t1 = time.perf_counter()
            self.last_endgame_time_ms = (t1 - t0) * 1000.0
            best_pos = best_pos_c.value
            self.last_endgame_value = score
            self.endgame_error = None         # 成功后清理历史错误
            if best_pos < 0:
                return (-1, -1, score)
            x, y = bit_to_xy(best_pos)
            return (x, y, score)
        except Exception as e:
            self.endgame_error = f"call:{type(e).__name__}"
            self.last_endgame_time_ms = None
            self.last_endgame_value = None    # 防止复用旧值
            return None

    def choose_move(self):
        """决策入口：优先终局求解，其次普通位运算首合法点"""
        # 1. 开局棋谱（前 8 步 / 前 12 颗子）
        self.book_used_move = False
        book_mv = self.try_book_move()
        if book_mv is not None:
            return book_mv
        # 0. 终局精确搜索
        empties = 64 - popcount(self.my_pieces | self.opp_pieces)
        if empties <= self.ENDGAME_MOVE_THRESHOLD:
            eg_result = self.endgame_search()
            if eg_result is not None:
                x, y, _ = eg_result
                return x, y  # 可能为 (-1,-1)
        # 2. TODO: MCTS 插入点
        # 3. 简单策略：取第一个合法着
        moves_bb = legal_moves(self.my_pieces, self.opp_pieces)
        if moves_bb == 0:
            return -1, -1
        lsb = moves_bb & -moves_bb
        pos = lsb.bit_length() - 1
        x, y = bit_to_xy(pos)
        return x, y

    def count_empties(self):
        return 64 - popcount(self.my_pieces | self.opp_pieces)
    
    def build_debug(self, empties: int) -> str:
        """
        统一构造 debug 字段，返回形如：
        "empties=xx;flag=...;val=...;t=...;msg=..."
        """
        mcts_fn_flag = 200     # 200 mcts 搜索被执行
        endgame_fn_flag = 300  # 300 进入残局，且枚举 endgame 成功加载
        book_fn_flag = 100        # 100 棋谱被调用
        botzone_time = "DNE"
        botzone_value = "DNE"
        botzone_flag = "DNE"
        botzone_message = "DNE"
        ### 残局 ###
        if empties <= self.ENDGAME_MOVE_THRESHOLD:
            if self.endgame_fn:
                if self.last_endgame_value is not None:     # 在残局，已成功加载 endgame 库，若本回合调用过并有返回值
                    if self.last_endgame_time_ms is not None:   
                        botzone_flag = str(endgame_fn_flag)
                        botzone_value = str(self.last_endgame_value)
                        botzone_time = f"{self.last_endgame_time_ms:.2f}ms"
                        botzone_message = "OK"
                    else:                                   # 若没记录耗时
                        botzone_flag = str(endgame_fn_flag)
                        botzone_value = str(self.last_endgame_value)
                        botzone_message = "OK"
                else:                                       # 优先展示调用错误，其次才是 no call
                    botzone_flag = str(endgame_fn_flag)
                    if self.endgame_error:
                        botzone_message = f"endgame_error {self.endgame_error}"
                    else:
                        botzone_message = "endgame_fn no call"  # 在残局，但库未加载
            else:                                           
                if self.endgame_error:                      # 若有错误原因：solver=FAIL <原因>（比如 not_found / load:异常类型）
                    botzone_message = f"endgame_error {self.endgame_error}"
                else:
                    botzone_message = f"MISS in endgame"    # 无原因则标记 solver=MISS
        ### 非残局 ###
        else:          
            botzone_flag = str(mcts_fn_flag)                         
            botzone_message = "OK"
        # 附加整局 run() 耗时：保持原逻辑值，再补 "/xx.xxms"
        if hasattr(self, 'last_run_time_ms') and self.last_run_time_ms is not None:
            botzone_time = f"{botzone_time}/{self.last_run_time_ms:.2f}ms"
        ### 整合输出 ###
        dbg_parts = [f"empties={empties}", f"flag={botzone_flag}", f"val={botzone_value}", f"t={botzone_time}", f"msg={botzone_message}"]

    # @@ NN part begins
        # NN 加载状态
        if self.nn_loaded:
            ok = sum(1 for m in self.nn_models if m is not None)
            if self.nn_total_load_ms is not None:
                dbg_parts.append(f"nn={ok}/3@{self.nn_total_load_ms:.1f}ms")
            else:
                dbg_parts.append(f"nn={ok}/3")
            # 若有失败，附加简短错误索引
            if ok < 3:
                fail_indices = [str(i+1) for i,e in enumerate(self.nn_errors) if e]
                if fail_indices:
                    dbg_parts.append("fail=" + ','.join(fail_indices))
        else:
            dbg_parts.append("nn=?")
        # NN 基准（3 个模型分别 100 次）
        if hasattr(self, "nn_bench_total_ms"):
            if self.nn_bench_total_ms is not None:
                # 输出每个模型单独耗时与平均（每次）耗时
                if hasattr(self, "nn_bench_each_ms") and isinstance(self.nn_bench_each_ms, list):
                    labels = getattr(self, "nn_bench_labels", ["m1","m2","m3"])
                    runs = getattr(self, "nn_bench_runs", 100)
                    parts = []
                    for i, ms in enumerate(self.nn_bench_each_ms):
                        avg = (ms / runs) if runs else 0.0
                        parts.append(f"{labels[i]}:{runs}@{ms:.1f}ms/{avg:.3f}ms")
                    dbg_parts.append("bench3=" + ','.join(parts))
            elif hasattr(self, "nn_bench_err"):
                dbg_parts.append(f"bench_err={self.nn_bench_err}")
        # 若使用棋谱，覆盖 flag & 在 msg 后附加 BOOK 标记
        if getattr(self, 'book_used_move', False):
            # 替换 flag=...
            for i, part in enumerate(dbg_parts):
                if part.startswith('flag='):
                    dbg_parts[i] = f'flag={book_fn_flag}'
                if part.startswith('msg='):
                    dbg_parts[i] = part + '|BOOK'
            # 附加棋谱条目计数（可选）
            if hasattr(self, 'book_loaded_count'):
                dbg_parts.append(f'book_entries={self.book_loaded_count}')
        return ';'.join(dbg_parts)
    # @@ NN part ends


    def run(self):
        run_start = time.perf_counter()
        # ----------------------------- 处理 Botzone 输入，获取棋盘
        raw = sys.stdin.readline().strip()
        if not raw:
            self.last_run_time_ms = (time.perf_counter() - run_start) * 1000.0
            print('{"response":{"x":-1,"y":-1}}')
            return
        requests, responses = self.parse_and_convert(raw)
        self.get_current_board(requests, responses)

        # ----------------------------- @@ NN part begins
        self.load_nn()

        # 基准：若模型成功加载，对当前局面构造一次输入，做 3 个模型各 100 次前向测时（共 300 次）
        if any(m is not None for m in self.nn_models) and not getattr(self, 'nn_bench_done', False):
            # 构造输入平面 (1,4,8,8): 0=my,1=opp,2=empty,3=legal
            planes = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
            my = self.my_pieces
            opp = self.opp_pieces
            occ = my | opp
            # my pieces
            tmp = my
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                r, c = divmod(idx, 8)
                planes[0, 0, r, c] = 1.0
                tmp ^= lsb
            # opp pieces
            tmp = opp
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                r, c = divmod(idx, 8)
                planes[0, 1, r, c] = 1.0
                tmp ^= lsb
            # empty
            empty = (~occ) & FULL_MASK
            tmp = empty
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                r, c = divmod(idx, 8)
                planes[0, 2, r, c] = 1.0
                tmp ^= lsb
            # legal
            legal_bb = legal_moves(my, opp)
            tmp = legal_bb
            while tmp:
                lsb = tmp & -tmp
                idx = lsb.bit_length() - 1
                r, c = divmod(idx, 8)
                planes[0, 3, r, c] = 1.0
                tmp ^= lsb
            per_model_runs = 100
            try:
                with torch.no_grad():
                    total_calls = 0
                    total_time = 0.0
                    each_ms = []
                    labels = ["early128", "mid128", "late64"]
                    for net in self.nn_models:
                        if net is None:
                            each_ms.append(0.0)
                            continue
                        _ = net(planes)  # warmup
                        t0 = time.perf_counter()
                        i = 0
                        while i < per_model_runs:
                            _ = net(planes)
                            i += 1
                        t1 = time.perf_counter()
                        elapsed_ms = (t1 - t0) * 1000.0
                        each_ms.append(elapsed_ms)
                        total_time += (t1 - t0)
                        total_calls += per_model_runs
                if total_calls > 0:
                    self.nn_bench_n = total_calls
                    self.nn_bench_total_ms = total_time * 1000.0
                    self.nn_bench_avg_ms = self.nn_bench_total_ms / total_calls
                    self.nn_bench_each_ms = each_ms
                    self.nn_bench_labels = labels
                    self.nn_bench_runs = per_model_runs
                else:
                    self.nn_bench_n = 0
                    self.nn_bench_total_ms = None
                    self.nn_bench_avg_ms = None
                    self.nn_bench_each_ms = None
            except Exception as e:
                self.nn_bench_total_ms = None
                self.nn_bench_avg_ms = None
                self.nn_bench_err = type(e).__name__
            self.nn_bench_done = True  # 标记只跑一次
        # ----------------------------- @@ NN part ends

        # ----------------------------- 落子决策
        empties = self.count_empties()
        x, y = self.choose_move()
        out = {"response": {"x": -1, "y": -1}} if x < 0 else {"response": {"x": y, "y": x}}

        # ----------------------------- 统一 debug 字段，Botzone 输出
        self.last_run_time_ms = (time.perf_counter() - run_start) * 1000.0
        out["debug"] = self.build_debug(empties)
        print(json.dumps(out, separators=(',', ':')))

if __name__ == "__main__":
    OthelloAI().run()