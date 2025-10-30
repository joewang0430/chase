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
from typing import Optional, IO, Dict, Any


ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
DATASET_DIR = os.path.join(ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")


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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数：控制切片大小、是否压缩、flush 频率。"""
    ap = argparse.ArgumentParser(description="Generate raw dataset (run bootstrap + writer)")
    ap.add_argument("--chunk-lines", type=int, default=200_000, help="lines per chunk before rolling")
    ap.add_argument("--no-compress", action="store_true", help="disable gzip compression for chunks")
    ap.add_argument("--flush-every", type=int, default=1000, help="flush after N lines")
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

    # 这里先不写数据。下一步把 driver 接进来，
    #      在对局循环里构造 record 并调用 writer.append_position(record)。

    writer.close()
    print("[DONE] run scaffold created. Writer opened and closed successfully.")


if __name__ == "__main__":
    main()
