#!/usr/bin/env python3
# 

"""
Simple filler: read full_book.jsonl -> update incomplete records -> overwrite same file.
- No lock file
- No tmp file / atomic swap
- Make a lightweight backup: full_book.jsonl.bak
- Idempotent: completed records are skipped

Record schema (per line):
{
  "key": "...",
  "depth": <int>,
  "moves_played": ["d3","c5","--"],
  "best_moves": [],
  "net_win": null,          # to be replaced with float in [-64,64]
  "timestamp": <int>,
  "engine_time": 4.0
}
"""
import os
import sys
import time
import json
import argparse
import shutil
from typing import List, Tuple, Optional

ROOT = os.path.dirname(os.path.dirname(__file__))  # /Users/.../chase
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from auto.software_driver import SoftwareDriver
BOOK_DIR = os.path.join(ROOT, "book")
INPUT_PATH = os.path.join(BOOK_DIR, "full_book.jsonl")
BACKUP_PATH = INPUT_PATH + ".bak"

# best_moves 为空 或 net_win 为 null(None) 时需要补全
def needs_work(obj: dict) -> bool:
    return (not obj.get("best_moves")) or (obj.get("net_win", None) is None)

def process(limit: Optional[int], time_budget: Optional[float], mock: bool) -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"[ERR] input not found: {INPUT_PATH}", file=sys.stderr) 
        sys.exit(1)
    
    start = time.time()
    deadline = start + time_budget if time_budget is not None else None
    updated = 0                     # 含义：已读并处理的记录行数（无论是否更新）。何时加一：每读到一行合法 JSON 就加一。遇到空行/坏行也会把它计入并原样写回。
    processed = 0                   # 含义：本次真正“填充成功”的记录数。何时加一：当 needs_work 为真且未触达 limit/未超时，并且 SoftwareDriver.solve 成功返回后，把 best_moves/net_win 写入该行时加一。
    out_lines: List[str] = []       # 含义：待写回文件的字符串列表（保持与输入同顺序）。内容：对每一行，若更新了则存更新后的 JSON 字符串；若未更新或解析失败则存原始行。

    driver = SoftwareDriver(engine_time=4.0, mock=mock)
    
    # 读取整本 -> 在线更新 -> 收集为 out_lines
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, 1):     # 按行读取文件；idx 是行号，从 1 开始计数；line 是当前行的原始文本（含末尾换行）。
            line = line.rstrip("\n")            # 去掉行尾的单个换行符，便于后续拼接写回时自己控制换行。
            if not line:                        # 如果这一行在去掉换行后是空串（空行）：
                out_lines.append("")            # 把一个空字符串收集到输出列表 out_lines（保持空行不变）。
                continue
            try:
                obj = json.loads(line)          # 把 JSON 文本解析成 Python 对象（通常是 dict）。成功则后续可基于字段进行更新。
            except Exception as e:
                # 解析失败：原样把这一行文本放回 out_lines，保证写回时不丢内容。
                out_lines.append(line)
                print(f"[WARN] bad json at line {idx}: {e}", file=sys.stderr)   # 向标准错误输出一条告警，指出第几行解析失败以及异常信息，便于排查。
                continue

            # 到此为止，一定是：该行是“非空且可成功解析的 JSON”。
            processed += 1
            can_update = True
            if limit is not None and updated >= limit:
                can_update = False
            if deadline is not None and time.time() >= deadline:            # 是不是超时/超工作量了
                can_update = False

            if needs_work(obj) and can_update:
            # -- 到此为止，一定是：该行是“可成功解析的 && 还没有被complete的 JSON”。且工作量规定允许（没有做超过规定的量）
                try:
                    driver.engine_time = float(obj.get("engine_time", 4.0))
                    best_moves, net_win = driver.solve(obj.get("moves_played", []))     # -- 这里调用cv一系列操作得到 sensei 解
                    if not isinstance(best_moves, list):    # 程序检查 best_moves 是否是列表；如果不是列表，就把它包成单元素列表。
                        best_moves = [str(best_moves)]
                    obj["best_moves"] = [str(m) for m in best_moves]    # 程序把 best_moves 中的每个元素都转换成字符串，并写回到 obj["best_moves"]（例如 'e3' 或 '--'）。
                    try:
                        net_val = float(net_win)                        # 程序把 net_val 变成小数；变不成就报错
                    except (TypeError, ValueError):
                        raise ValueError(f"invalid net_win from driver: {net_win!r}")
                    obj["net_win"] = max(-64.0, min(64.0, net_val))
                    obj["timestamp"] = int(time.time())                 # 程序把当前 Unix 秒写入 obj["timestamp"]
                    updated += 1
                except Exception as e:
                    # 失败：保持原样（仍待补）
                    print(f"[ERR] solve failed at line {idx}: {e}", file=sys.stderr)
            
            out_lines.append(json.dumps(obj, separators=(',', ':')))    # 把当前记录 obj 序列化成紧凑的 JSON 文本，并把这段文本追加到 out_lines 列表中。

    # 轻量备份
    try:
        shutil.copyfile(INPUT_PATH, BACKUP_PATH)
        print(f"[BACKUP] wrote {BACKUP_PATH}")
    except Exception as e:
        print(f"[WARN] backup failed: {e}", file=sys.stderr)

    # 然后覆写原文件
    with open(INPUT_PATH, "w", encoding="utf-8") as fout:
        fout.write("\n".join(out_lines) + "\n")

    # 最后总结下
    elapsed = time.time() - start
    print(f"[DONE] processed={processed} updated={updated} elapsed={elapsed:.2f}s")
   

def main():
    ap = argparse.ArgumentParser(description="Fill best_moves and net_win into full_book.jsonl (simple overwrite)")
    ap.add_argument("--limit", type=int, default=None, help="max records to update in this run")
    ap.add_argument("--time-budget", type=float, default=None, help="seconds budget for this run")
    ap.add_argument("--mock", action="store_true", help="use mock driver (no real software)")
    args = ap.parse_args()
    process(limit=args.limit, time_budget=args.time_budget, mock=args.mock)

if __name__ == "__main__":
    main()