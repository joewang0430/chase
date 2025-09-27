import os
import sys
import json
import time
import ctypes
from ctypes import c_uint64, c_int, POINTER, byref
import random

# 添加 Python 3.6 兼容 popcount (替换 int.bit_count)
POPCNT8 = [bin(i).count('1') for i in range(256)]
ENDING_SO_NAME = "ending.so"

def popcount(x: int) -> int:
    c = 0
    while x:
        c += POPCNT8[x & 0xFF]
        x >>= 8
    return c

# Removed numpy; use pure Python integers for bitboards.
class OthelloAI:
    def __init__(self):
        self.my_pieces = 0
        self.opp_pieces = 0
        self.my_color = 1
        # Masks
        self.FILE_A = 0x0101010101010101
        self.FILE_H = 0x8080808080808080
        self.NOT_FILE_A = 0xFFFFFFFFFFFFFFFF ^ self.FILE_A
        self.NOT_FILE_H = 0xFFFFFFFFFFFFFFFF ^ self.FILE_H
        self.FULL_MASK = 0xFFFFFFFFFFFFFFFF
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
    def xy_to_bit(self, x, y):
        return x * 8 + y
    def bit_to_xy(self, bit_pos):
        return bit_pos // 8, bit_pos % 8
    def set_bit(self, bitboard, x, y):
        return bitboard | (1 << (x * 8 + y))

    # --- 初始棋盘 ---
    def init_standard_board(self, my_color):
        self.my_color = my_color
        if my_color == 1:  # 黑方
            self.my_pieces = self.set_bit(0, 3, 4) | self.set_bit(0, 4, 3)
            self.opp_pieces = self.set_bit(0, 3, 3) | self.set_bit(0, 4, 4)
        else:  # 白方
            self.my_pieces = self.set_bit(0, 3, 3) | self.set_bit(0, 4, 4)
            self.opp_pieces = self.set_bit(0, 3, 4) | self.set_bit(0, 4, 3)

    # --- 位移函数 ---
    def _shift_e (self, bb): return (bb & self.NOT_FILE_H) << 1 & self.FULL_MASK
    def _shift_w (self, bb): return (bb & self.NOT_FILE_A) >> 1
    def _shift_n (self, bb): return bb >> 8
    def _shift_s (self, bb): return (bb << 8) & self.FULL_MASK
    def _shift_ne(self, bb): return (bb & self.NOT_FILE_H) >> 7
    def _shift_nw(self, bb): return (bb & self.NOT_FILE_A) >> 9
    def _shift_se(self, bb): return ((bb & self.NOT_FILE_H) << 9) & self.FULL_MASK
    def _shift_sw(self, bb): return ((bb & self.NOT_FILE_A) << 7) & self.FULL_MASK

    # --- 合法着点生成 ---
    def get_legal_moves(self, me, opp):
        """Optimized legal move generation without per-call inner function closures."""
        empty = ~(me | opp) & self.FULL_MASK
        if empty == 0:
            return 0
        moves = 0
        me_local = me; opp_local = opp; empty_local = empty
        shift_e  = self._shift_e;  shift_w  = self._shift_w
        shift_n  = self._shift_n;  shift_s  = self._shift_s
        shift_ne = self._shift_ne; shift_nw = self._shift_nw
        shift_se = self._shift_se; shift_sw = self._shift_sw
        # East
        x = shift_e(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_e(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # West
        x = shift_w(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_w(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # North
        x = shift_n(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_n(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # South
        x = shift_s(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_s(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # NE
        x = shift_ne(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_ne(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # NW
        x = shift_nw(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_nw(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # SE
        x = shift_se(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_se(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        # SW
        x = shift_sw(me_local) & opp_local; acc = 0
        while x:
            nxt = shift_sw(x); acc |= nxt; x = nxt & opp_local
        moves |= acc & empty_local
        return moves

    # --- 落子 ---
    def make_move(self, me, opp, pos):
        """Optimized move application (inline directional scans)."""
        if pos < 0:  # PASS
            return opp, me
        move_bit = 1 << pos
        if (me | opp) & move_bit:  # occupied
            return me, opp
        flips = 0
        me_local = me; opp_local = opp
        shift_e  = self._shift_e;  shift_w  = self._shift_w
        shift_n  = self._shift_n;  shift_s  = self._shift_s
        shift_ne = self._shift_ne; shift_nw = self._shift_nw
        shift_se = self._shift_se; shift_sw = self._shift_sw
        # For each direction, walk once; collect line only if terminated by own piece.
        # East
        cur = shift_e(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_e(cur)
        if cur & me_local: flips |= line
        # West
        cur = shift_w(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_w(cur)
        if cur & me_local: flips |= line
        # North
        cur = shift_n(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_n(cur)
        if cur & me_local: flips |= line
        # South
        cur = shift_s(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_s(cur)
        if cur & me_local: flips |= line
        # NE
        cur = shift_ne(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_ne(cur)
        if cur & me_local: flips |= line
        # NW
        cur = shift_nw(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_nw(cur)
        if cur & me_local: flips |= line
        # SE
        cur = shift_se(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_se(cur)
        if cur & me_local: flips |= line
        # SW
        cur = shift_sw(move_bit); line = 0
        while cur & opp_local:
            line |= cur; cur = shift_sw(cur)
        if cur & me_local: flips |= line
        if flips == 0:
            return me, opp
        me |= move_bit | flips
        opp &= ~flips
        return me, opp

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
                    self.my_pieces, self.opp_pieces = self.make_move(self.my_pieces, self.opp_pieces, self.xy_to_bit(mv["x"], mv["y"]))
                if i + 1 < len(requests):
                    opp_mv = requests[i + 1]
                    if opp_mv["x"] >= 0:
                        self.opp_pieces, self.my_pieces = self.make_move(self.opp_pieces, self.my_pieces, self.xy_to_bit(opp_mv["x"], opp_mv["y"]))
        else:  # 白方后手
            for i in range(turn_count):
                opp_mv = requests[i]
                if opp_mv["x"] >= 0:
                    self.opp_pieces, self.my_pieces = self.make_move(self.opp_pieces, self.my_pieces, self.xy_to_bit(opp_mv["x"], opp_mv["y"]))
                my_mv = responses[i]
                if my_mv["x"] >= 0:
                    self.my_pieces, self.opp_pieces = self.make_move(self.my_pieces, self.opp_pieces, self.xy_to_bit(my_mv["x"], my_mv["y"]))
            if turn_count < len(requests):
                cur = requests[turn_count]
                if cur["x"] >= 0:
                    self.opp_pieces, self.my_pieces = self.make_move(self.opp_pieces, self.my_pieces, self.xy_to_bit(cur["x"], cur["y"]))

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
            x, y = self.bit_to_xy(best_pos)
            return (x, y, score)
        except Exception as e:
            self.endgame_error = f"call:{type(e).__name__}"
            self.last_endgame_time_ms = None
            self.last_endgame_value = None    # 防止复用旧值
            return None

    def choose_move(self):
        """决策入口：优先终局求解，其次普通位运算首合法点"""
        ### 残局枚举 ###
        empties = 64 - popcount(self.my_pieces | self.opp_pieces)
        if empties <= self.ENDGAME_MOVE_THRESHOLD:
            eg_result = self.endgame_search()
            if eg_result is not None:
                x, y, _ = eg_result
                return x, y  # 可能为 (-1,-1)
        ### TODO: later mcts here ###
        moves_bb = self.get_legal_moves(self.my_pieces, self.opp_pieces)
        if moves_bb == 0:
            return -1, -1
        lsb = moves_bb & -moves_bb
        pos = lsb.bit_length() - 1
        x, y = self.bit_to_xy(pos)
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