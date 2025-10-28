# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py

from typing import List, Tuple, Optional
import subprocess
import time
import re
from .base import SoftwareDriverBase

APP_NAME = "Othello Sensei"

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

    # 新增：窗口截图探针，供 --probe-snap 使用
    def snap_window(self, out_path: str) -> str:
        self.ensure_running()
        x, y, w, h = _get_window_bounds(APP_NAME)
        _snap_rect_png(x, y, w, h, out_path)
        return out_path
    
    def reset_board(self) -> None:
        # TODO: 回到初始局面（下一步实现）
        return

    def replay_moves(self, moves_played: List[str]) -> None:
        # TODO: 点击/快捷键复盘；"--" 表示 PASS
        return

    def wait_and_read(self) -> Tuple[List[str], float]:
        # TODO: 等 self.engine_time 并解析 UI
        raise NotImplementedError("Implement macOS UI reading")