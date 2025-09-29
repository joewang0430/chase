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

# 添加 Python 3.6 兼容 popcount (替换 int.bit_count)
POPCNT8 = [bin(i).count('1') for i in range(256)]
ENDING_SO_NAME = "ending.so"
NN_NAME_1 = "chase8.pt"

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
        # 1. 终局精确搜索
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
        mcts_fn_flag = random.randint(100, 109)     # 100-109 mcts 搜索被执行
        endgame_fn_flag = random.randint(110, 119)  # 110-119 进入残局，且枚举 endgame 成功加载
        # book_flag = random.randint(120, 129)        # 120-129 棋谱被调用
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
        ### 整合输出 ###
        dbg_parts = [f"empties={empties}", f"flag={botzone_flag}", f"val={botzone_value}", f"t={botzone_time}", f"msg={botzone_message}"]
        return ';'.join(dbg_parts)


    def run(self):
        # ----------------------------- 处理 Botzone 输入，获取棋盘
        raw = sys.stdin.readline().strip()
        if not raw:
            print('{"response":{"x":-1,"y":-1}}')
            return
        requests, responses = self.parse_and_convert(raw)
        self.get_current_board(requests, responses)

        # ----------------------------- 落子决策
        empties = self.count_empties()
        x, y = self.choose_move()
        out = {"response": {"x": -1, "y": -1}} if x < 0 else {"response": {"x": y, "y": x}}

        # ----------------------------- 统一 debug 字段，Botzone 输出
        out["debug"] = self.build_debug(empties)
        print(json.dumps(out, separators=(',', ':')))

if __name__ == "__main__":
    OthelloAI().run()