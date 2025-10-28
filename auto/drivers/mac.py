# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py

from typing import List, Tuple, Optional
import subprocess
import time
import re
import os
import json
from .base import SoftwareDriverBase

APP_NAME = "Othello Sensei"
CALIB_PATH = os.path.expanduser("~/.sensei_calib.json")  # 保存标定数据


def _save_calib(data: dict) -> None:
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

def _load_calib() -> dict:
    # 如果标定文件不存在，给出清晰提示
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(
            f"calibration not found: {CALIB_PATH}. Run with --probe-calibrate first."
        )
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
'''
文件里存了什么
JSON 字段：
    app: "Othello Sensei"
    board_rel: [bx, by, bw, bh]
        含义：棋盘在应用窗口内的相对矩形，单位像素
        bx,by 为棋盘左上角相对“窗口左上角”的偏移
        bw,bh 为棋盘区域的宽高
    运行时：实际点击坐标 = 窗口坐标 + board_rel 里的相对坐标
'''

# 把如 "d3" 的棋盘坐标转换为屏幕像素点(x,y)，使用 window 矩形(win=wx,wy,ww,wh)与棋盘相对矩形(board_rel=bx,by,bw,bh)，按 a1 左下、h8 右上的约定计算中心点
def _coord_to_xy(coord: str, win: Tuple[int,int,int,int], board_rel: Tuple[int,int,int,int]) -> Tuple[int,int]:
    # coord: "d3" 等；a1 左下，h8 右上
    wx, wy, ww, wh = win
    bx, by, bw, bh = board_rel
    if len(coord) != 2:
        raise ValueError(f"bad coord: {coord!r}")
    col = coord[0].lower()
    row = coord[1]
    if not ('a' <= col <= 'h') or not ('1' <= row <= '8'):
        raise ValueError(f"coord out of range: {coord!r}")
    s = bw / 8.0  # 单格边长
    c = ord(col) - ord('a')          # 列 0..7
    r = int(row) - 1                 # 行 0..7（自上而下）
    x = int(wx + bx + (c + 0.5) * s)
    y = int(wy + by + (r + 0.5) * s)          # 屏幕 y 向下增大
    return x, y
    

# 把那段 script 交给系统命令 osascript 执行。
def _run_osa(script: str) -> str:
    """Run an AppleScript snippet and return stdout (stripped)."""
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:                    # 如果执行失败（返回码非 0），抛出 RuntimeError，并把错误信息带出来。
        raise RuntimeError(f"osascript failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()                  # 如果成功，返回这段 AppleScript 的标准输出（去掉首尾空白）。

# 判断某个应用是否在运行。
def _app_is_running(name: str) -> bool:
    # Fast path: AppleScript 'application "X" is running'
    out = _run_osa(f'application "{name}" is running as boolean')   # 这句是 AppleScript（系统脚本语言），不是“人话”。
    return out.lower() == "true"    

# activate_app: 发送 AppleScript 的 activate 把已运行的应用置前台，通常不负责启动应用。
def _activate_app(name: str) -> None:
    _run_osa(f'tell application "{name}" to activate')

# _launch_app: 用 open -a 启动应用以确保进程存在，但不保证前置到前台。
def _launch_app(name: str) -> None:
    # Use open -a to launch by name; avoids needing full path
    subprocess.run(["open", "-a", name], check=True)

# _window_count(name): 询问 System Events“这个应用现在有多少个窗口”，返回整数。需要为你的终端/VS Code授予“辅助功能”权限。
def _window_count(name: str) -> int:
    # Query via System Events; requires Accessibility permission for your terminal/Python
    script = (
        'tell application "System Events"\n'
        f'  if exists process "{name}" then\n'
        f'    return (count of windows of process "{name}")\n'
        '  else\n'
        '    return 0\n'
        '  end if\n'
        'end tell'
    )
    out = _run_osa(script)
    try:
        return int(out)
    except ValueError:
        return 0
    
def _get_window_bounds(name: str) -> Tuple[int, int, int, int]:
    """
    返回窗口矩形 (x, y, w, h)。对 osascript 的输出做鲁棒解析。
    """
    # 方案A：position + size
    script_a = (
        'tell application "System Events"\n'
        f'  if not (exists process "{name}") then return "NA"\n'
        f'  tell process "{name}"\n'
        '    if not (exists window 1) then return "NA"\n'
        '    set p to position of window 1\n'
        '    set s to size of window 1\n'
        '    return (item 1 of p) & "," & (item 2 of p) & "," & (item 1 of s) & "," & (item 2 of s)\n'
        '  end tell\n'
        'end tell'
    )
    out = _run_osa(script_a)
    nums = re.findall(r'-?\d+', out)
    if len(nums) < 4:
        # 方案B：直接读 bounds（有的系统更稳定）
        script_b = (
            'tell application "System Events"\n'
            f'  if not (exists process "{name}") then return "NA"\n'
            f'  tell process "{name}"\n'
            '    if not (exists window 1) then return "NA"\n'
            '    set b to bounds of window 1 -- {x,y,w,h}\n'
            '    return (item 1 of b) & "," & (item 2 of b) & "," & (item 3 of b) & "," & (item 4 of b)\n'
            '  end tell\n'
            'end tell'
        )
        out = _run_osa(script_b)
        nums = re.findall(r'-?\d+', out)

    if len(nums) < 4:
        raise RuntimeError(f"window bounds parse failed: {out!r}")
    x, y, w, h = map(int, nums[:4])
    return x, y, w, h
    
def _snap_rect_png(x: int, y: int, w: int, h: int, out_path: str) -> None:
    # 用系统截图工具截取矩形区域（需要屏幕录制权限）
    subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", out_path], check=True)
    
# ---------------------------------------- Mac Driver

class MacDriver(SoftwareDriverBase):
    def ensure_running(self) -> None:
        # 1) 启动（如未运行）
        if not _app_is_running(APP_NAME):
            _launch_app(APP_NAME)

        # 2) 激活到前台
        _activate_app(APP_NAME)

        # 3) 等待至少出现一个窗口（最多等待 ~10 秒）
        deadline = time.time() + 10.0
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                if _window_count(APP_NAME) > 0:
                    return
            except Exception as e:
                last_err = str(e)
            time.sleep(0.2)

        hint = (
            "If this is the first run or permissions changed, grant Accessibility: "
            "System Settings > Privacy & Security > Accessibility, add your Terminal/VS Code."
        )
        if last_err:
            raise RuntimeError(f"Failed to detect window for '{APP_NAME}': {last_err}. {hint}")
        raise RuntimeError(f"Timed out waiting for '{APP_NAME}' window. {hint}")

    # 探针：窗口截图探针，供 --probe-snap 使用
    def snap_window(self, out_path: str) -> str:
        self.ensure_running()
        x, y, w, h = _get_window_bounds(APP_NAME)
        _snap_rect_png(x, y, w, h, out_path)
        return out_path
    
    # 标定：让用户依次指向“棋盘左上角”和“棋盘右下角”，按回车确认
    # 引导你把鼠标依次放到“棋盘左上角、棋盘右下角”，按回车采样两点，计算出 board_rel 并写入 /Users/juanjuan1/.sensei_calib.json
    def probe_calibrate(self) -> str:
        import pyautogui  # 延迟导入
        pyautogui.FAILSAFE = False
        self.ensure_running()
        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        print("[CAL] 请把 Othello Sensei 置前台。")
        input("[CAL] 把鼠标移动到 棋盘左上角 的边界上，按回车确认...")
        tl = pyautogui.position()
        input("[CAL] 把鼠标移动到 棋盘右下角 的边界上，按回车确认...")
        br = pyautogui.position()
        tlx, tly = tl.x, tl.y
        brx, bry = br.x, br.y
        bx, by = tlx - wx, tly - wy
        bw, bh = brx - tlx, bry - tly
        if bw <= 0 or bh <= 0:
            raise ValueError(f"bad board rectangle: bw={bw}, bh={bh}")
        data = {"app": APP_NAME, "board_rel": [int(bx), int(by), int(bw), int(bh)]}
        _save_calib(data)
        print(f"[CAL] saved to {CALIB_PATH}: board_rel={data['board_rel']}")
        return CALIB_PATH
    
    # 试点点击：按棋盘坐标点击（例如 ["d3","e6"]）
    # 读取标定文件与当前窗口位置，依次把 "d3","e6" 等转为像素并点击；"--" 跳过
    def probe_click(self, coords: List[str], delay: float = 0.12) -> None:
        import pyautogui
        pyautogui.FAILSAFE = False
        self.ensure_running()
        cfg = _load_calib()
        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        board_rel = cfg.get("board_rel")
        if not (isinstance(board_rel, list) and len(board_rel) == 4):
            raise ValueError(f"bad calibration in {CALIB_PATH}: {board_rel!r}")
        bx, by, bw, bh = board_rel
        for c in coords:
            c = c.strip()
            if not c or c == "--":
                time.sleep(delay)
                continue
            x, y = _coord_to_xy(c, (wx,wy,ww,wh), (bx,by,bw,bh))
            pyautogui.click(x, y)
            time.sleep(delay)
    
    def reset_board(self) -> None:
        # TODO: 回到初始局面（下一步实现）
        return

    def replay_moves(self, moves_played: List[str]) -> None:
        # TODO: 点击/快捷键复盘；"--" 表示 PASS
        return

    def wait_and_read(self) -> Tuple[List[str], float]:
        # TODO: 等 self.engine_time 并解析 UI
        raise NotImplementedError("Implement macOS UI reading")