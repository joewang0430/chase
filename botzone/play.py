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
# 需要同时加载的 4 个（旧 Torch 兼容格式）模型文件名
NN_NAME_1 = "chase1_compat.pt"
NN_NAME_2 = "chase2_compat.pt"
NN_NAME_3 = "chase3_compat.pt"
NN_NAME_4 = "chase4_compat.pt"
NN_ALL = [NN_NAME_1, NN_NAME_2, NN_NAME_3, NN_NAME_4]

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

# @@ NN part begins
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        h = self.c1(x)
        h = self.b1(h)
        h = self.act(h)
        h = self.c2(h)
        h = self.b2(h)
        return self.act(x + h)
    
class Net(nn.Module):
    """7 residual blocks, 64 channels (与 nn/model_def.Net 当前版本保持一致)."""
    def __init__(self, channels: int = 64, n_blocks: int = 7, input_planes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        # policy head
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.p_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.p_fc2 = nn.Linear(128, 65)  # 64 + pass
        # value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(4 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 64)
        self.v_fc3 = nn.Linear(64, 1)
        self.act = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan))
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        # policy
        p = self.p_head(x)
        p = p.view(p.size(0), -1)
        p = self.act(self.p_fc1(p))
        p = self.p_fc2(p)  # raw logits
        # value
        v = self.v_head(x)
        v = v.view(v.size(0), -1)
        v = self.act(self.v_fc1(v))
        v = self.act(self.v_fc2(v))
        v = self.tanh(self.v_fc3(v))
        return p, v
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

    # @@ NN part begins
        # 四个模型（7res x 64ch）
        self.nn_models = [None, None, None, None]  # type: ignore
        self.nn_loaded = False
        self.nn_total_load_ms = None  # 四个模型整体加载耗时
        self.nn_errors = [None, None, None, None]

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
                ckpt = torch.load(path, map_location='cpu')
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    arch = ckpt.get('arch', {})
                    ch = arch.get('channels', 64)
                    blocks = arch.get('n_blocks', 7)
                    inp = arch.get('input_planes', 4)
                    net = Net(channels=ch, n_blocks=blocks, input_planes=inp)
                    state = ckpt['model_state']
                elif isinstance(ckpt, dict):
                    net = Net()  # 64x7 默认
                    state = ckpt
                else:
                    self.nn_errors[idx] = 'bad_format'
                    continue
                # 旧命名兼容（若训练脚本还在用 blocks./p_c 等）
                if any(k.startswith(prefix) for prefix in ("blocks.", "p_c", "p_bn", "v_c", "v_bn") for k in state.keys()):
                    remapped = {}
                    for k, v in state.items():
                        nk = k
                        if nk.startswith('blocks.'):
                            nk = 'body.' + nk[len('blocks.') :]
                        elif nk.startswith('p_c.'):
                            nk = 'p_head.0.' + nk[len('p_c.') :]
                        elif nk.startswith('p_bn.'):
                            nk = 'p_head.1.' + nk[len('p_bn.') :]
                        elif nk.startswith('v_c.'):
                            nk = 'v_head.0.' + nk[len('v_c.') :]
                        elif nk.startswith('v_bn.'):
                            nk = 'v_head.1.' + nk[len('v_bn.') :]
                        remapped[nk] = v
                    state = remapped
                net.load_state_dict(state, strict=False)
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
                dbg_parts.append(f"nn={ok}/4@{self.nn_total_load_ms:.1f}ms")
            else:
                dbg_parts.append(f"nn={ok}/4")
            # 若有失败，附加简短错误索引
            if ok < 4:
                fail_indices = [str(i+1) for i,e in enumerate(self.nn_errors) if e]
                if fail_indices:
                    dbg_parts.append("fail=" + ','.join(fail_indices))
        else:
            dbg_parts.append("nn=?")
        # NN 基准
        if hasattr(self, "nn_bench_total_ms"):
            if self.nn_bench_total_ms is not None:
                dbg_parts.append(
                    f"bench={self.nn_bench_n}@{self.nn_bench_total_ms:.1f}ms/{self.nn_bench_avg_ms:.3f}ms"
                )
            elif hasattr(self, "nn_bench_err"):
                dbg_parts.append(f"bench_err={self.nn_bench_err}")
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

        # 基准：若模型成功加载，对当前局面构造一次输入，做 1000 次前向测时
        # 四模型基准：每个 250 次，总共 1000 次（成功加载的数量 * 250）
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
            per_model_runs = 250
            try:
                with torch.no_grad():
                    total_calls = 0
                    total_time = 0.0
                    for net in self.nn_models:
                        if net is None:
                            continue
                        _ = net(planes)  # warmup
                        t0 = time.perf_counter()
                        for _ in range(per_model_runs):
                            _ = net(planes)
                        t1 = time.perf_counter()
                        total_time += (t1 - t0)
                        total_calls += per_model_runs
                if total_calls > 0:
                    self.nn_bench_n = total_calls
                    self.nn_bench_total_ms = total_time * 1000.0
                    self.nn_bench_avg_ms = self.nn_bench_total_ms / total_calls
                else:
                    self.nn_bench_n = 0
                    self.nn_bench_total_ms = None
                    self.nn_bench_avg_ms = None
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