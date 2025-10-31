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
from typing import Optional, IO, Dict, Any, Tuple


ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
DATASET_DIR = os.path.join(ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")


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
    # 提前占个位：后续会新增 --driver/--time-budget/--games 等参数
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """程序入口：创建 run 目录，初始化 writer（先演示打开/关闭）。"""
    args = parse_args(argv)

    # 1) 创建 run 目录 + 写 meta.json
    rp = start_new_run()
    print(f"[RUN] {rp.run_dir}")

    # 2) 准备 chunked writer（此处仅演示打开与关闭，后续会写入真实样本）
    writer = RunWriter(
        base_dir=rp.run_dir,
        chunk_lines=int(args.chunk_lines),
        compress=(not args.no_compress),
        flush_every=int(args.flush_every),
    )

    # 3) 把本次“关键运行参数”落在 meta.json，便于追溯
    params = DatasetParams(engine_time=float(args.engine_time), early_random_moves=int(args.early_random))
    update_meta_params(rp.meta_path, params)

    # 4) 演示：写入极简“假数据”N 行，便于验证写入结构（不接 driver，不做合法性）
    # demo_n = int(getattr(args, "demo_lines", 0) or 0)
    # if demo_n > 0:
    #     # 选择若干“目标子数”制造分段覆盖（只是演示）
    #     pcs_targets = [14, 28, 40, 52]
    #     game_id = f"demo_{_shortid(6)}"
    #     for i in range(min(demo_n, len(pcs_targets))):
    #         pcs_target = pcs_targets[i]
    #         player = i % 2  # 交替当前方
    #         my_bb, opp_bb = _mock_bitboards(pcs_target, player)
    #         rec = build_record_skeleton(
    #             game_id=game_id,
    #             player=player,
    #             my_bb=my_bb,
    #             opp_bb=opp_bb,
    #             best_moves=["d3", "c5"],  # 演示字段
    #             net_win=0.0,                # 演示字段
    #             engine_time=float(args.engine_time),
    #             probe_ms=None,
    #         )
    #         writer.append_position(rec)

    writer.close()
    print("[DONE] run scaffold created. Writer opened and closed successfully.")


if __name__ == "__main__":
    main()
