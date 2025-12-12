from __future__ import annotations

import os
import re
import sys
import json
import time
import gzip
import uuid
import argparse
from dataclasses import dataclass
from typing import Optional, IO, Dict, Any, Tuple, List
import random
import ctypes
import subprocess
import math

#!/usr/bin/env python3
"""
Dataset generator (part 1): run bootstrap + chunked writer.

中文人话：这一版只做“支架”：
- 创建 run 目录、写 meta.json；
- 提供 RunWriter，负责把每条样本一行行写到 JSONL，写满自动滚动切片（可选 gzip）。
后面再把对局/驱动接进来。

Human notes (why this file exists):
- 我们先把“落盘骨架”搭好：一次运行一个 run 目录；jsonl 切片；崩溃后能恢复；
    后续只要调用 append_position(record) 就能把单步样本写进去。
- 记录格式遵守 dataset/README.md 的“精简版每行结构”。本阶段不校验所有字段，
    但会做最基本的 key/合法计数/stage 的占位（后续会在构造记录时计算）。
"""

# 测试单个棋谱：
# python3 auto/generate_dataset.py --collect-one --driver mac --no-compress

# 在 600 秒内尽量多采（但每盘会完整结束） 上限是600秒，因为 --games 很大：
# python3 auto/generate_dataset.py --collect-one --driver mac --time-budget 300 --games 9999 --no-compress

# 采 10 局，不压缩：
# python3 auto/generate_dataset.py --collect-one --driver mac --games 10 --no-compress

# 7 小时
# python3 auto/generate_dataset.py --collect-one --driver mac --time-budget 25200 --games 9999 --no-compress

# 上面的的命令行例子都不 compress

'''
TODO: after 11.1:
    1. 更改 p 的拉回函数，当前的太线性，0-10 和 10-20 都太一致，要让 0-10 的拉回度增长远高于 10-20. 可以先试试修改 k 值，但是大概率得改整个函数公式。
    2. 修改每一个 pcs 阶段对应的 engine_time。实现合理效率，但 注重准确定优先。
    3. 测试无误，开跑。
    (4). 除此之外可能整个棋谱对弈逻辑都得改。
'''

'''
TODO: after 11.1:
    1. best_moves 这部分目前好像还是只记录一个，我希望有多少记录多少。
    2. othello sensei 引擎崩溃问题得解决。
    3. 测试无误，开跑。
    (4). 除此之外可能整个棋谱对弈逻辑都得改。
'''


ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
DATASET_DIR = os.path.join(ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")

# 确保以 `python3 auto/generate_dataset.py` 方式运行时也能导入 `auto.*` 包
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ------------------------------
# pcs -> p (oracle 概率基线) 映射表
# 约定：
# - pcs ∈ [12,25]   => p = 0.65
# - pcs ∈ [26,39]   => p = 0.75
# - pcs ∈ [40,53]   => p = 0.90
# 只定义字典与取值函数，不改变其他流程。

def _build_default_p_schedule() -> Dict[int, float]:
    d: Dict[int, float] = {}
    for pcs in range(12, 26):
        d[pcs] = 0.65
    for pcs in range(26, 40):
        d[pcs] = 0.75
    for pcs in range(40, 54):
        d[pcs] = 0.90
    return d


# 模块级默认表：12..53 全覆盖
P_SCHEDULE: Dict[int, float] = _build_default_p_schedule()


def p_for_pcs(pcs: int, schedule: Optional[Dict[int, float]] = None, default: float = 0.75) -> float:
    """根据子数 pcs 返回“原始基线”的 p（不含 net_win 拉回）。

    仅做字典查表；默认表覆盖 12..53。若超界或缺失，返回 default。
    """
    m = P_SCHEDULE if schedule is None else schedule
    try:
        return float(m.get(int(pcs), float(default)))
    except Exception:
        return float(default)


# ------------------------------
# pcs -> engine_time（秒）映射表（12..53）。
# 明确按键硬写，便于随手改动（不依赖构造函数）。
# 约定：默认所有 pcs（12..53）均为 2.0 秒；按需可直接改任意键的值。

ENGINE_TIME_SCHEDULE: Dict[int, float] = {
    12: 6.0,
    13: 6.0,
    14: 6.0,
    15: 6.0,
    16: 6.0,
    17: 6.0,
    18: 6.0,
    19: 6.0,
    20: 6.0,
    21: 6.0,
    22: 6.0,
    23: 6.0,
    24: 6.0,
    25: 6.0,
    26: 6.0,
    27: 6.0,
    28: 6.0,
    29: 6.0,
    30: 6.0,
    31: 6.0,
    32: 6.0,
    33: 6.0,
    34: 6.0,
    35: 6.0,
    36: 6.0,
    37: 6.0,
    
    38: 4.0,
    39: 4.0,
    40: 3.0,
    41: 3.0,
    42: 2.0,
    43: 2.0,

    44: 1.0,
    45: 1.0,
    46: 1.0,
    47: 1.0,

    48: 1.0,
    49: 1.0,
    50: 1.0,
    51: 1.0,
    52: 1.0,
    53: 1.0,
}


def engine_time_for_pcs(pcs: int, schedule: Optional[Dict[int, float]] = None, default: float = 6.0) -> float:
    """根据 pcs 返回 engine_time（秒）。默认表覆盖 12..53；超界返回 default。

    用法示例：engine_time_for_pcs(current_pcs) → 6.0
    后续如需差异化，只需修改 ENGINE_TIME_SCHEDULE 对应键的数值。
    """
    m = ENGINE_TIME_SCHEDULE if schedule is None else schedule
    try:
        return float(m.get(int(pcs), float(default)))
    except Exception:
        return float(default)


# ------------------------------
# C 端随机评估器（eva_noise_random.c）绑定（仅用于开局 3-8 步）

class _CSampler:
    """轻量封装 c/eva_noise_random.c，提供 choose_move 和 generate_moves。

    仅在需要时加载；找不到库则抛错（严格失败）。
    """
    def __init__(self) -> None:
        self.lib: Optional[ctypes.CDLL] = None
        self._choose_move = None
        self._gen_moves = None

    def _lib_paths(self) -> Tuple[str, str]:
        src = os.path.join(ROOT, "c", "eva_noise_random.c")
        if sys.platform == "darwin":
            lib = os.path.join(ROOT, "c", "eva_noise_random.dylib")
        else:
            lib = os.path.join(ROOT, "c", "eva_noise_random.so")
        return src, lib

    def _maybe_build(self, src: str, out: str) -> None:
        # 尝试在库缺失或过期时构建；若本机没有编译器则抛错
        need = (not os.path.exists(out)) or (os.path.getmtime(out) < os.path.getmtime(src))
        if not need:
            return
        cc_candidates = ["clang", "gcc"]
        is_darwin = sys.platform == "darwin"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        last_err = None
        for cc in cc_candidates:
            try:
                if is_darwin:
                    cmd = [cc, "-O3", "-std=c11", "-fPIC", "-dynamiclib", "-o", out, src]
                else:
                    cmd = [cc, "-O3", "-std=c11", "-fPIC", "-shared", "-o", out, src]
                subprocess.run(cmd, check=True)
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"failed to build eva_noise_random: {last_err}")

    def ensure_loaded(self) -> None:
        if self.lib is not None:
            return
        src, libp = self._lib_paths()
        self._maybe_build(src, libp)
        lib = ctypes.CDLL(libp)
        # int choose_move(uint64_t my_pieces, uint64_t opp_pieces)
        lib.choose_move.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
        lib.choose_move.restype = ctypes.c_int
        # u64 generate_moves(Board board)
        class CBoard(ctypes.Structure):
            _fields_ = [("board", ctypes.c_uint64 * 2)]
        self.CBoard = CBoard  # type: ignore[attr-defined]
        lib.generate_moves.argtypes = (CBoard,)  # type: ignore[attr-defined]
        lib.generate_moves.restype = ctypes.c_uint64
        self.lib = lib
        self._choose_move = lib.choose_move
        self._gen_moves = lib.generate_moves

    def choose_move(self, my_bb: int, opp_bb: int) -> int:
        self.ensure_loaded()
        pos = int(self._choose_move(ctypes.c_uint64(my_bb), ctypes.c_uint64(opp_bb)))  # type: ignore[misc]
        return pos

    def generate_moves(self, my_bb: int, opp_bb: int) -> int:
        self.ensure_loaded()
        b = self.CBoard()  # type: ignore[attr-defined]
        b.board[0] = ctypes.c_uint64(int(my_bb))
        b.board[1] = ctypes.c_uint64(int(opp_bb))
        return int(self._gen_moves(b))  # type: ignore[misc]


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 字符串（到毫秒），用于 meta.json 时间戳。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"


def _shortid(n: int = 4) -> str:
    """返回一个简短随机 hex 串，用来拼到 run_id/game_id 上避免重名。"""
    return uuid.uuid4().hex[:max(1, int(n))]


def _ensure_dir(path: str) -> None:
    """确保目录存在（不存在就递归创建）。"""
    os.makedirs(path, exist_ok=True)


@dataclass
class RunPaths:
    """run 目录的路径集合：便于主流程传参和打印。"""
    run_id: str
    day_dir: str
    run_dir: str
    meta_path: str


def start_new_run() -> RunPaths:
    """创建 run 目录并写 meta.json，返回该 run 的路径信息。"""

    # 一个采集“会话”对应一个 run 目录，文件都放在里面；
    # run_id 里包含时间戳+短随机串；meta.json 先写骨架（驱动/配置稍后补）。
    day = time.strftime("%Y-%m-%d")
    ts_for_id = time.strftime("%Y%m%dT%H%M%S")
    rid = f"{ts_for_id}_{_shortid(4)}"

    day_dir = os.path.join(RAW_DIR, day)
    run_dir = os.path.join(day_dir, f"run_{rid}")
    _ensure_dir(run_dir)

    meta_path = os.path.join(run_dir, "meta.json")
    meta = {
        "run_id": rid,
        "started_at": _now_iso(),
        # 下面这些信息后续可以补齐/更新
        "driver_version": None,
        "sensei_version": None,
        "calib_hash": None,
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "params": {},  # CLI 与策略摘要后续填
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return RunPaths(run_id=rid, day_dir=day_dir, run_dir=run_dir, meta_path=meta_path)


class RunWriter:
    """Chunked JSONL writer with optional gzip compression.

    关键点：
    - 我们不一次性写巨文件；达到阈值就滚动到 positions_0002.jsonl(.gz) …
    - 崩溃后不丢太多：每写 N 行 flush 一次（可调）。
    - 只负责“写行”；字段合法性在“构造 record”阶段保证。
    """

    def __init__(
        self,
        base_dir: str,
        chunk_lines: int = 200_000,
        compress: bool = True,
        flush_every: int = 1000,
    ) -> None:
        """初始化 writer：指定目录、每片行数、是否压缩、多久 flush 一次。"""
        self.base_dir = base_dir
        self.chunk_lines = max(1, int(chunk_lines))
        self.compress = bool(compress)
        self.flush_every = max(1, int(flush_every))
        self._idx = 0
        self._lines_in_chunk = 0
        self._fh: Optional[IO[bytes]] = None if self.compress else None  # type: ignore
        self._open_next_chunk()

    def _chunk_path(self) -> str:
        """返回当前切片文件的完整路径（含 .gz 后缀与否）。"""
        stem = f"positions_{self._idx:04d}.jsonl"
        return os.path.join(self.base_dir, stem + (".gz" if self.compress else ""))

    def _open_next_chunk(self) -> None:
        """关闭旧文件并打开下一个切片文件，重置行计数。"""
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
        self._idx += 1
        path = self._chunk_path()
        # 以二进制写入（gzip 需要 bytes）
        self._fh = gzip.open(path, "ab", compresslevel=5) if self.compress else open(path, "ab")
        self._lines_in_chunk = 0

    def append_position(self, record: Dict[str, Any]) -> None:
        """Append one JSONL record (dict -> compact JSON line).

        调用方传“已经准备好”的 dict。本方法不去聪明地改字段，只保证按行追加。
        """
        if not isinstance(record, dict):
            raise TypeError("record must be a dict")
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        # 加一个换行，jsonl 一行一条
        data = (line + "\n").encode("utf-8")
        assert self._fh is not None, "writer not opened"
        self._fh.write(data)
        self._lines_in_chunk += 1
        # 定期 flush，崩溃损失更小
        if (self._lines_in_chunk % self.flush_every) == 0:
            try:
                self._fh.flush()
            except Exception:
                pass
        # 轮换新文件
        if self._lines_in_chunk >= self.chunk_lines:
            self._open_next_chunk()

    def close(self) -> None:
        """关闭底层文件句柄；幂等调用安全。"""
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None


# ------------------------------
# 下一步：参数与记录构造的“辅助件”（先搭骨架，不跑对局）

@dataclass
class DatasetParams:
    """本次采集的重要参数（会写进 meta.json，便于复现与审计）。

    人话：这些不是“写文件必须用”的参数，而是让将来知道这批数据是怎么跑出来的。
    """
    engine_time: float = 4.0          # Sensei 求解时限（秒）
    early_random_moves: int = 6       # 开局前 N 步走“更随机”的策略
    pcs_min: int = 12                 # 只收子数下界（含）
    pcs_max: int = 53                 # 只收子数上界（含）；到 53 直接结束一盘
    skip_pass: bool = True            # PASS 步不收

    def to_dict(self) -> Dict[str, Any]:
        """转为能写进 meta.json 的 dict。"""
        return {
            "engine_time": float(self.engine_time),
            "early_random_moves": int(self.early_random_moves),
            "pcs_min": int(self.pcs_min),
            "pcs_max": int(self.pcs_max),
            "skip_pass": bool(self.skip_pass),
        }


def update_meta_params(meta_path: str, params: DatasetParams) -> None:
    """把参数合并写入 meta.json（保留已有字段）。

    人话：不覆盖历史信息，只是把 params 转成 dict 后合并进去。
    """
    try:
        cur = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                cur = json.load(f)
        cur.setdefault("params", {})
        cur["params"].update(params.to_dict())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 这里不抛，避免写参数影响主流程；打印提示即可
        print(f"[WARN] update_meta_params failed: {e}")


# ------------------------------
# p 值根据 net_win 的“拉回函数”（只提供函数，不改动调用点）

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """把 x 夹到 [lo, hi]。"""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def adjust_p_with_net_win(
    p_base: float,
    net_win: float,
    *,
    p_lo: float = 0.05,
    p_hi: float = 0.95,
    K: float = 0.5,
    gamma: float = 1.2,
) -> float:
    """根据当前净胜子对基线 p 做方向性拉回。

    公式（方案A-凸组合）：
        w = clamp(net_win / 64, -1, 1)
        s = clamp(K * |w| ** gamma, 0, 1)
        p_target = p_lo (w>0 领先)；p_hi (w<0 落后)；w=0 则 p_target=p_base
        p' = clamp((1 - s) * p_base + s * p_target, 0, 1)
    """
    # 规范化净胜子到 [-1,1]
    try:
        w = float(net_win) / 64.0
    except Exception:
        w = 0.0
    w = _clamp(w, -1.0, 1.0)

    # 拉动强度 s ∈ [0,1]
    s = _clamp(K * (abs(w) ** float(gamma)), 0.0, 1.0)

    # 目标值：领先(>0)往更随机的低 p 拉；落后(<0)往更稳的高 p 拉
    if w > 0.0:
        p_target = float(p_lo)
    elif w < 0.0:
        p_target = float(p_hi)
    else:
        p_target = float(p_base)

    # 凸组合 + 夹到 [0,1]
    try:
        p0 = float(p_base)
    except Exception:
        p0 = 0.5
    p_new = (1.0 - s) * p0 + s * p_target
    return _clamp(p_new, 0.0, 1.0)


def adjust_p_with_net_win_new(net_win: float, *, c: float = 0.08167) -> float:
    """统一新策略：仅根据 net_win 计算使用 oracle 的概率 p。

    规则：
    - 基线为 0.5；不分 pcs 阶段，不用旧的分桶表。
    - 取 a = |net_win|，计算 k0 = log2(a)。若 k0 < 0，强制置 0。
    - 根据 net_win 的符号恢复符号：net_win>0 → k = +k0；net_win<0 → k = -k0；net_win==0 → k = 0。
    - 常数 c=0.08167，最终概率 p = 0.5 - k * c。
    - 返回值夹到 [0,1]。
    """
    try:
        a = abs(float(net_win))
    except Exception:
        a = 0.0

    # 处理 a=0 的对数与负对数归零
    if a <= 0.0:
        k0 = 0.0
    else:
        k0 = math.log(a, 2.0)
        if k0 < 0.0:
            k0 = 0.0

    # 恢复符号到 [-6,6]（log2(64)=6 的自然上界，不强制截断）
    if net_win > 0.0:
        k = +k0
    elif net_win < 0.0:
        k = -k0
    else:
        k = 0.0

    p = 0.5 - (k * float(c))
    return _clamp(p, 0.0, 1.0)


# ------------------------------
# 开局前 8 手：1-6（纯本地），7-8（带 OCR/oracle），不写盘，仅推进状态

# 复用 botzone/play.py 的简单位棋盘函数（legal_moves/apply_move）
try:
    from botzone.play import legal_moves as _legal_moves, apply_move as _apply_move
except Exception:
    _legal_moves = None
    _apply_move = None

def _ensure_play_funcs() -> None:
    if _legal_moves is None or _apply_move is None:
        raise RuntimeError("botzone.play legal_moves/apply_move not available")

def _mv_to_idx(mv: str) -> int:
    from utils.converters import Converters
    return Converters.mv_to_dmv(mv)

def _idx_to_mv(idx: int) -> str:
    from utils.converters import Converters
    return Converters.dmv_to_mv(idx)

def _black_white_to_my_opp(black: int, white: int, side: str) -> Tuple[int, int]:
    """side: 'B' or 'W' → 返回 (my, opp)。"""
    if side.upper() == 'B':
        return black, white
    else:
        return white, black

def _apply_on_bw(black: int, white: int, side: str, pos: int) -> Tuple[int, int, str]:
    """在黑白位板上以 side 落子 pos（0..63），返回更新后的 (black, white, next_side)。
    使用 botzone.play.apply_move（me,opp）语义。
    """
    _ensure_play_funcs()
    me, opp = _black_white_to_my_opp(black, white, side)
    # 严格校验：pos 必须在合法棋位中
    legal = _legal_moves(me, opp)
    if pos < 0 or pos > 63 or not (legal & (1 << pos)):
        raise RuntimeError(f"illegal move @{pos} for side={side}")
    out = _apply_move(me, opp, pos)
    if not out:
        raise RuntimeError(f"illegal move @{pos} for side={side}")
    me2, opp2 = out
    if side.upper() == 'B':
        black2, white2 = me2, opp2
        next_side = 'W'
    else:
        black2, white2 = opp2, me2
        next_side = 'B'
    return black2, white2, next_side

def _has_legal(black: int, white: int, side: str) -> bool:
    _ensure_play_funcs()
    me, opp = _black_white_to_my_opp(black, white, side)
    return _legal_moves(me, opp) != 0

def opening_moves_1_to_6(*, rng: Optional[random.Random] = None) -> Tuple[List[str], int, int, str]:
    """生成开局前 1-6 手，不用 OCR：
    - 第 1 手黑：等概率随机 d3/c4/e6/f5；
    - 第 2 手白：按首手显式表的 0.55/0.35/0.10 分布；
    - 第 3-6 手：用 C 评估器 choose_move 抽样；
    - 遇 PASS 不计数，直到实际落子才计数；
    返回：(moves, black_bb, white_bb, side_to_move)。
    """
    _ensure_play_funcs()
    rng = rng or random.Random()
    # 初始标准盘（黑先）
    black = (1 << (3*8+4)) | (1 << (4*8+3))
    white = (1 << (3*8+3)) | (1 << (4*8+4))
    side = 'B'
    moves: List[str] = []
    sampler = _CSampler()

    # 第 1 手：黑等概率
    first_candidates = ["d3", "c4", "e6", "f5"]
    mv1 = rng.choice(first_candidates)
    idx1 = _mv_to_idx(mv1)
    black, white, side = _apply_on_bw(black, white, side, idx1)
    moves.append(mv1)

    # 第 2 手：白显式表
    second_map: Dict[str, Tuple[List[str], List[float]]] = {
        "d3": (["c5", "c3", "e3"], [0.55, 0.35, 0.10]),
        "c4": (["e3", "c3", "c5"], [0.55, 0.35, 0.10]),
        "e6": (["f4", "f6", "d6"], [0.55, 0.35, 0.10]),
        "f5": (["d6", "f6", "f4"], [0.55, 0.35, 0.10]),
    }
    cand2, prob2 = second_map[mv1]
    mv2 = rng.choices(cand2, weights=prob2, k=1)[0]
    idx2 = _mv_to_idx(mv2)
    black, white, side = _apply_on_bw(black, white, side, idx2)
    moves.append(mv2)

    # 第 3-6 手：C 抽样（处理 PASS：不计数，切换走子）
    while len(moves) < 6:
        if not _has_legal(black, white, side):
            # PASS：切换一手；若连续 PASS（理论上不会发生在前 8 手），直接 break 防御
            side = 'W' if side == 'B' else 'B'
            if not _has_legal(black, white, side):
                break
            # 不计数，继续
            continue
        me, opp = _black_white_to_my_opp(black, white, side)
        pos = sampler.choose_move(me, opp)
        if pos < 0 or pos > 63:
            raise RuntimeError(f"C sampler returned invalid move: {pos}")
        mv = _idx_to_mv(pos)
        black, white, side = _apply_on_bw(black, white, side, pos)
        moves.append(mv)

    return moves, black, white, side


def opening_moves_7_to_8(
    *,
    moves_so_far: List[str],
    black_bb: int,
    white_bb: int,
    side_to_move: str,
    driver_name: Optional[str] = None,
    mock: bool = False,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], int, int, str, List[Dict[str, Any]]]:
    """推进第 7-8 手：每手 50% C 噪声，50% oracle（OCR 最佳点取 top-1）。

    随机性说明：为避免调用方传入的 rng（例如固定种子或被 1–6 手消耗）影响 7–8 的分支掷币，
    此处改用 SystemRandom() 进行分支选择，使每次运行更接近“真实随机”。

    入参：截至 6 手的 moves_so_far、黑白位板、轮到方；
    返回：新增的 moves_7to8、更新位板和 side_to_move，以及每步的日志条目（来源/所选 move/net_win 等）。
    严格模式：任何不可预期情况直接抛错（由调用者负责保存前序数据）。
    """
    from auto.drivers import create_driver
    _ensure_play_funcs()
    rng = rng or random.Random()
    sys_rng = random.SystemRandom()  # 仅用于分支掷币
    sampler = _CSampler()

    logs: List[Dict[str, Any]] = []
    new_moves: List[str] = []
    black = int(black_bb)
    white = int(white_bb)
    side = side_to_move.upper()

    # 准备 driver（oracle 2s），不调用 solve()/reset/replay；只 ensure + wait_and_read + click
    drv = create_driver(engine_time=2.0, driver=driver_name, mock=mock)
    # 只需确保一次应用在前台（不会重置盘面）
    try:
        drv.ensure_running()
    except Exception as e:
        raise RuntimeError(f"driver ensure_running failed: {e}")

    while len(new_moves) < 2:  # 两手，PASS 不计数
        # PASS 处理：不计数；若连续 PASS（虽然前 8 手不会出现），仍做防御
        if not _has_legal(black, white, side):
            side = 'W' if side == 'B' else 'B'
            if not _has_legal(black, white, side):
                raise RuntimeError("consecutive PASS within first 8 moves")
            continue

        use_oracle = sys_rng.random() < 0.75
        if use_oracle:
            # 直接读取（等待 engine_time 秒），取 top-1 最佳黄格。
            best_moves, net_win = drv.wait_and_read()
            if not isinstance(best_moves, list) or len(best_moves) == 0:
                raise RuntimeError(f"oracle returned empty best_moves: {best_moves}")
            mv = str(best_moves[0])  # 只取第一名（不在候选间随机）
            pos = _mv_to_idx(mv)
            # 严格校验合法性
            me_chk, opp_chk = _black_white_to_my_opp(black, white, side)
            legal = _legal_moves(me_chk, opp_chk)
            if pos < 0 or pos > 63 or not (legal & (1 << pos)):
                raise RuntimeError(f"oracle proposed illegal move {mv} (pos={pos}) at side={side}")
            # 点击以保持 UI 同步
            try:
                drv.click_move(mv)
            except Exception as e:
                raise RuntimeError(f"driver click_move failed for {mv}: {e}")
            src = "oracle"
            logs.append({"src": src, "mv": mv, "net_win": float(net_win) if net_win is not None else None})
        else:
            # C 噪声
            me, opp = _black_white_to_my_opp(black, white, side)
            pos = sampler.choose_move(me, opp)
            if pos < 0 or pos > 63:
                raise RuntimeError(f"C sampler returned invalid move: {pos}")
            mv = _idx_to_mv(pos)
            # 点击以保持 UI 同步（不等待分析）
            try:
                drv.click_move(mv)
            except Exception as e:
                raise RuntimeError(f"driver click_move failed for {mv}: {e}")
            src = "noise"
            logs.append({"src": src, "mv": mv})

        # 应用到黑白位板 & 累积 moves_so_far
        black, white, side = _apply_on_bw(black, white, side, pos)
        new_moves.append(mv)
        moves_so_far.append(mv)

    return new_moves, black, white, side, logs


def stage_from_pcs(pcs: int) -> int:
    """按约定把子数映射到阶段：1:[12,25], 2:[26,39], 3:[40,53]。

    人话：这只是一个“分桶函数”，方便训练分段用。
    """
    if pcs <= 25:
        return 1
    if pcs <= 39:
        return 2
    return 3


def should_capture(pcs: int, is_pass: bool, cfg: DatasetParams) -> bool:
    """决定这一手要不要入库：
    - 跳过 PASS（可配置）；
    - 只收 pcs ∈ [cfg.pcs_min, cfg.pcs_max]。
    """
    if cfg.skip_pass and is_pass:
        return False
    return (cfg.pcs_min <= pcs <= cfg.pcs_max)


def bit_count(x: int) -> int:
    """Python 的 int 有大位宽，这里用内置 bit_count；老版本可换成 bin(x).count('1')。"""
    try:
        return int(x).bit_count()  # type: ignore[attr-defined]
    except AttributeError:
        return bin(int(x)).count("1")


def compute_pcs(my_bb: int, opp_bb: int) -> int:
    """计算当前子数（黑白总和），用于过滤与分段。"""
    return bit_count(my_bb) + bit_count(opp_bb)


def make_key(my_bb: int, opp_bb: int, player: int) -> str:
    """构造简单的去重 key：<my>:<opp>:<player>（十六进制）。"""
    return f"{my_bb:016x}:{opp_bb:016x}:{player}"


def build_record_skeleton(
    *,
    game_id: str,
    player: int,
    my_bb: int,
    opp_bb: int,
    legal_bb: Optional[int] = None,
    legal_count: Optional[int] = None,
    best_moves: Optional[list[str]] = None,
    net_win: Optional[float] = None,
    engine_time: Optional[float] = None,
    probe_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """按约定字段产出一条“待写入”的记录骨架。

    人话：
    - 这里不强制所有字段都有值（有些留给后续阶段补全）。
    - pcs/stage/key 在这里直接算好；其它从调用者传入。
    - legal_bb/最佳着法/数值等，调用前应在上层算好并传进来。
    """
    pcs = compute_pcs(my_bb, opp_bb)
    rec: Dict[str, Any] = {
        "key": make_key(my_bb, opp_bb, player),
        "game_id": str(game_id),
        "player": int(player),
        "my_bb": int(my_bb),
        "opp_bb": int(opp_bb),
        "pcs": int(pcs),
        "stage": stage_from_pcs(pcs),
        "timestamp": int(time.time()),
    }
    if legal_bb is not None:
        rec["legal_bb"] = int(legal_bb)
    if legal_count is not None:
        rec["legal_count"] = int(legal_count)
    if best_moves is not None:
        rec["best_moves"] = [str(m) for m in best_moves]
    if net_win is not None:
        # 夹到 [-64,64]
        try:
            v = float(net_win)
            rec["net_win"] = max(-64.0, min(64.0, v))
        except Exception:
            pass
    if engine_time is not None:
        rec["engine_time"] = float(engine_time)
    if probe_ms is not None:
        rec["probe_ms"] = float(probe_ms)
    return rec


def collect_game_12_to_53(
    *,
    writer: RunWriter,
    driver_name: Optional[str] = None,
    mock: bool = False,
    rng: Optional[random.Random] = None,
    params: Optional[DatasetParams] = None,
    game_id: Optional[str] = None,
) -> Dict[str, Any]:
    """从开局推进到 53pcs，按每手 OCR 记录并选择 oracle/noise 点击落子。

    流程（每手循环）：
    - 若当前方无合法着法：PASS（不计数，不落盘），切换走子；若双方都无着法则终局。
    - pcs -> engine_time；设置到 driver；OCR 读取 best_moves, net_win；
    - 写本步记录（best_moves/net_win/engine_time/合法着点统计等）；
    - 用“本步 net_win”调整“本步的 p”，SystemRandom 掷币：
        - oracle：取 top-1（严格合法校验），click_move；
        - noise：C 抽样一个合法点，click_move；
    - 落子到位板，切换走子；pcs 达到 params.pcs_max(=53) 即结束。

    返回：一个简短的 summary dict。
    约定：本函数内部会自行创建/确保 driver 前台；开局 1-8 手复用现有实现，7-8 会点击到 UI 保持同步。
    """
    from auto.drivers import create_driver

    _ensure_play_funcs()
    rng = rng or random.Random()
    cfg = params or DatasetParams()
    sys_rng = random.SystemRandom()
    sampler = _CSampler()

    # 1) 开局 1-6（本地）
    moves: List[str] = []
    m1_6, black, white, side = opening_moves_1_to_6(rng=rng)
    moves.extend(m1_6)

    # 2) 第 7-8 手（保证 UI 同步：内部 ensure + wait_and_read/click）
    # 先把 UI 同步到内存中的 1–6 手终局：reset -> replay(1..6)
    try:
        from auto.drivers import create_driver as _create_driver_sync  # 局部导入避免顶层依赖
        drv_sync = _create_driver_sync(engine_time=2.0, driver=driver_name, mock=mock)
        drv_sync.ensure_running()
        if hasattr(drv_sync, "reset_board"):
            drv_sync.reset_board()
        if hasattr(drv_sync, "replay_moves"):
            drv_sync.replay_moves(m1_6)
        time.sleep(0.20)  # 稍作等待，避免动画/评估未稳定
    except Exception as e:
        raise RuntimeError(f"failed to sync UI to opening 1–6: {e}")

    m7_8, black, white, side, logs78 = opening_moves_7_to_8(
        moves_so_far=moves,
        black_bb=black,
        white_bb=white,
        side_to_move=side,
        driver_name=driver_name,
        mock=mock,
        rng=rng,
    )
    moves.extend(m7_8)

    # 3) 准备 midgame driver（不 reset，不 replay；只用于 OCR 与点击）
    drv = create_driver(engine_time=2.0, driver=driver_name, mock=mock)
    try:
        drv.ensure_running()
    except Exception as e:
        raise RuntimeError(f"driver ensure_running failed: {e}")

    gid = game_id or f"g_{_shortid(8)}"
    steps_written = 0
    pending_records: List[Dict[str, Any]] = []
    oracle_moves = 0
    noise_moves = 0
    pass_moves = 0

    def _both_no_legal(b: int, w: int, s: str) -> bool:
        if not _has_legal(b, w, s):
            s2 = 'W' if s == 'B' else 'B'
            return not _has_legal(b, w, s2)
        return False

    # 4) 主循环：直到 53pcs 或无子可下
    while True:
        # 终止条件
        if _both_no_legal(black, white, side):
            break

        # PASS：不计数不写盘
        if not _has_legal(black, white, side):
            pass_moves += 1
            side = 'W' if side == 'B' else 'B'
            continue

        pcs = compute_pcs(black, white)
        # 达到上限直接结束（不再生成新着）
        if pcs >= cfg.pcs_max:
            break

        # 当前步分析时间（按 pcs 表设置）
        et = engine_time_for_pcs(pcs)
        drv.engine_time = float(et)

        # OCR 读取（容错：失败重试一次）
        best_moves: List[str] = []
        net_win: Optional[float] = None
        for _try in range(2):
            try:
                bm, nw = drv.wait_and_read()
                if isinstance(bm, list) and len(bm) > 0:
                    best_moves = [str(m) for m in bm]
                net_win = float(nw) if nw is not None else None
                break
            except Exception:
                best_moves = []
                net_win = None

        # 记录本步（若在采集区间且非 PASS）
        me, opp = _black_white_to_my_opp(black, white, side)
        legal_bb = _legal_moves(me, opp)
        legal_cnt = bit_count(legal_bb)

        # 严格要求：一旦 OCR 的 top-1 不在当前走方的合法集合中，立即报错并退出
        if best_moves:
            try:
                top_mv = str(best_moves[0])
                top_pos = _mv_to_idx(top_mv)
            except Exception:
                top_mv = str(best_moves[0])
                top_pos = -1
            if top_pos < 0 or not (legal_bb & (1 << top_pos)):
                pcs_dbg = compute_pcs(black, white)
                raise RuntimeError(
                    f"OCR top-1 illegal: mv={top_mv} pos={top_pos} side={side} pcs={pcs_dbg} legal_count={legal_cnt}"
                )
        if should_capture(pcs, False, cfg):
            rec = build_record_skeleton(
                game_id=gid,
                player=(0 if side == 'B' else 1),
                my_bb=me,
                opp_bb=opp,
                legal_bb=legal_bb,
                legal_count=legal_cnt,
                best_moves=best_moves if best_moves else None,
                net_win=net_win,
                engine_time=et,
                probe_ms=None,
            )
            pending_records.append(rec)

        # 决策：使用新策略，仅依赖 net_win 计算 p
        p_adj = adjust_p_with_net_win_new(net_win if net_win is not None else 0.0)
        use_oracle = sys_rng.random() < p_adj

        # 产生并点击落子
        pos: Optional[int] = None
        mv: Optional[str] = None
        if use_oracle and best_moves:
            mv = str(best_moves[0])
            pos = _mv_to_idx(mv)
            if pos < 0 or not (legal_bb & (1 << pos)):
                # 严格：OCR top-1 非法 → 立即错误并退出
                raise RuntimeError(
                    f"OCR top-1 illegal: mv={mv} pos={pos} side={side} pcs={pcs} legal_count={legal_cnt}"
                )

        if not use_oracle:
            # 噪声：采样一个合法点
            me2, opp2 = me, opp
            for _ in range(3):  # 最多尝试 3 次
                cand = sampler.choose_move(me2, opp2)
                if 0 <= cand <= 63 and (legal_bb & (1 << cand)):
                    pos = cand
                    mv = _idx_to_mv(pos)
                    break
            if mv is None or pos is None:
                # 兜底：取最低位合法点
                if legal_bb == 0:
                    # 理论上不会发生（上面已检查 has_legal）
                    side = 'W' if side == 'B' else 'B'
                    continue
                lsb = (legal_bb & -legal_bb)
                pos = (lsb.bit_length() - 1)
                mv = _idx_to_mv(pos)

        # 点击（保持 UI 同步）
        try:
            assert mv is not None and pos is not None
            drv.click_move(mv)
        except Exception as e:
            raise RuntimeError(f"driver click_move failed for {mv}: {e}")

        # 应用到位板
        assert pos is not None
        black, white, side = _apply_on_bw(black, white, side, pos)
        moves.append(mv)  # type: ignore[arg-type]
        if use_oracle:
            oracle_moves += 1
        else:
            noise_moves += 1

        # 结束条件：落子后 pcs 达到上限
        if compute_pcs(black, white) >= cfg.pcs_max:
            break

    # 正常结束：一次性把本盘缓存写入
    for rec in pending_records:
        writer.append_position(rec)
    steps_written = len(pending_records)

    return {
        "game_id": gid,
        "moves": moves,
        "steps_written": steps_written,
        "oracle": oracle_moves,
        "noise": noise_moves,
        "pass": pass_moves,
    }


def _mock_bitboards(total_pcs: int, player: int) -> Tuple[int, int]:
    """合成一对“互不重叠”的 bitboard，以便演示：
    - 低位给对手，高位给当前方；
    - 总子数为 total_pcs；大致对半分配（向下取整给当前方）。

    注：这只是演示数据，不代表真实棋型，也不与 legal_bb 对齐。
    """
    total = max(0, min(63, int(total_pcs)))
    m = max(0, total // 2)
    o = total - m
    my_mask = ((1 << m) - 1) << (64 - m) if m > 0 else 0
    opp_mask = ((1 << o) - 1) if o > 0 else 0
    if player not in (0, 1):
        player = 0
    if player == 0:
        return my_mask, opp_mask
    else:
        return opp_mask, my_mask


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数：控制切片大小、是否压缩、flush 频率，以及少量采集参数。"""
    ap = argparse.ArgumentParser(description="Generate raw dataset (run bootstrap + writer)")
    ap.add_argument("--chunk-lines", type=int, default=200_000, help="lines per chunk before rolling")
    ap.add_argument("--no-compress", action="store_true", help="disable gzip compression for chunks")
    ap.add_argument("--flush-every", type=int, default=1000, help="flush after N lines")
    # 先把关键运行参数也接进来，并写到 meta.json 里（不驱动对局，仅记录）
    ap.add_argument("--engine-time", type=float, default=4.0, help="seconds budget for Sensei solving")
    ap.add_argument("--early-random", type=int, default=6, help="number of early random moves")
    ap.add_argument("--demo-lines", type=int, default=2, help="write N mock records for a dry-run demo")
    # 采集开关（可选）：运行一盘 12..53 并写盘
    ap.add_argument("--collect-one", action="store_true", help="run one midgame collection (12..53 pcs) and write records")
    ap.add_argument("--games", type=int, default=1, help="number of games to collect when --collect-one is set")
    ap.add_argument("--seed", type=int, default=None, help="base random seed (per game adds +i)")
    ap.add_argument("--time-budget", type=float, default=None, help="seconds budget for multi-game collection; finish the current game after crossing the deadline, then stop")
    ap.add_argument("--driver", type=str, default=None, help="driver name, e.g., mac/mock/windows/linux")
    ap.add_argument("--mock", action="store_true", help="use mock driver (no UI)")
    # 提前占个位：后续会新增 --driver/--time-budget/--games 等参数
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """程序入口：创建 run 目录，初始化 writer（可选运行一盘 12..53 采集）。"""
    args = parse_args(argv)

    # 1) 创建 run 目录 + 写 meta.json
    rp = start_new_run()
    print(f"[RUN] {rp.run_dir}")

    # 2) 准备 chunked writer
    writer = RunWriter(
        base_dir=rp.run_dir,
        chunk_lines=int(args.chunk_lines),
        compress=(not args.no_compress),
        flush_every=int(args.flush_every),
    )

    # 3) 把本次“关键运行参数”落在 meta.json，便于追溯
    params = DatasetParams(engine_time=float(args.engine_time), early_random_moves=int(args.early_random))
    update_meta_params(rp.meta_path, params)

    # 4) 预留：后续加 --collect-one / --driver 等参数触发真实采集。
    if getattr(args, "collect_one", False):
        total_steps = 0
        total_oracle = 0
        total_noise = 0
        total_pass = 0
        done_games = 0
        games_target = max(1, int(getattr(args, "games", 1)))
        base_seed = getattr(args, "seed", None)
        time_budget = getattr(args, "time_budget", None)
        t_start = time.time()
        deadline = (t_start + float(time_budget)) if (time_budget is not None) else None

        if deadline is not None:
            print(f"[COLLECT] start (time-budget={float(time_budget):.2f}s, up to {games_target} game(s)) …")
        else:
            print(f"[COLLECT] start (up to {games_target} game(s)) …")

        i = 0
        try:
            while i < games_target:
                # 若设了 time budget：只有在未到截止时才开始新的一盘
                if deadline is not None and time.time() >= deadline:
                    break  # 不再开启新盘

                # per-game RNG（若提供种子的话）
                rng = random.Random(int(base_seed) + i) if base_seed is not None else random.Random()

                print(f"[COLLECT] game#{i+1} starting …")
                summary = collect_game_12_to_53(
                    writer=writer,
                    driver_name=getattr(args, "driver", None),
                    mock=bool(getattr(args, "mock", False)),
                    rng=rng,
                    params=DatasetParams(engine_time=float(args.engine_time), early_random_moves=int(args.early_random)),
                )
                done_games += 1
                i += 1

                print(
                    f"[COLLECT] game#{i}: steps={summary.get('steps_written')} "
                    f"oracle={summary.get('oracle')} noise={summary.get('noise')} pass={summary.get('pass')}"
                )
                total_steps += int(summary.get('steps_written') or 0)
                total_oracle += int(summary.get('oracle') or 0)
                total_noise += int(summary.get('noise') or 0)
                total_pass += int(summary.get('pass') or 0)

                # 若设置了时间预算，且完成这一盘后已越过截止时间，则到此为止（满足“超过限时的第一盘结束后停止”）
                if deadline is not None and time.time() >= deadline:
                    print("[COLLECT] time budget reached after finishing current game; stopping …")
                    break
        except KeyboardInterrupt:
            print("\n[COLLECT] interrupted by user; finishing up …")

        elapsed = time.time() - t_start
        print(
            f"[COLLECT] all done: games={done_games} steps={total_steps} oracle={total_oracle} "
            f"noise={total_noise} pass={total_pass} elapsed={elapsed:.2f}s"
        )

    writer.close()
    print("[DONE] run scaffold created. Writer opened and closed successfully.")


if __name__ == "__main__":
    main()
