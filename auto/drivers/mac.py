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
    
# 合并保存（不覆盖已有字段）
def _save_calib_merged(update: dict) -> dict:
    '''
    先读现有/Users/juanjuan1/.sensei_calib.json，再用新键值更新并写回，保留未更新的旧键。
    '''
    cur = {}
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                cur = json.load(f) or {}
        except Exception:
            cur = {}
    cur.update(update)
    _save_calib(cur)
    return cur
    
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

def _dismiss_new_game_prompt(name: str) -> bool:
    """
    尝试关闭“New game?”提示。返回 True 表示做了动作（点了按钮或发回车）。
    需要“辅助功能”权限。
    """
    script = (
        'tell application "System Events"\n'
        f'  if not (exists process "{name}") then return false\n'
        f'  tell process "{name}"\n'
        '    set acted to false\n'
        '    -- sheet（贴在窗口上的对话）\n'
        '    try\n'
        '      if exists sheet 1 of front window then\n'
        '        set s to sheet 1 of front window\n'
        '        set btns to {"Yes","OK","Ok","确定","是"}\n'
        '        repeat with t in btns\n'
        '          try\n'
        '            if exists button (t as string) of s then\n'
        '              click button (t as string) of s\n'
        '              set acted to true\n'
        '              exit repeat\n'
        '            end if\n'
        '          end try\n'
        '        end repeat\n'
        '        if acted is false then key code 36 -- Return\n'
        '        set acted to true\n'
        '      end if\n'
        '    end try\n'
        '    -- AXDialog（独立对话框）\n'
        '    try\n'
        '      if exists (window 1 whose subrole is "AXDialog") then\n'
        '        set dlg to (window 1 whose subrole is "AXDialog")\n'
        '        set btns2 to {"Yes","OK","Ok","确定","是"}\n'
        '        repeat with t in btns2\n'
        '          try\n'
        '            if exists button (t as string) of dlg then\n'
        '              click button (t as string) of dlg\n'
        '              set acted to true\n'
        '              exit repeat\n'
        '            end if\n'
        '          end try\n'
        '        end repeat\n'
        '        if acted is false then key code 36 -- Return\n'
        '        set acted to true\n'
        '      end if\n'
        '    end try\n'
        '    return acted\n'
        '  end tell\n'
        'end tell'
    )
    out = _run_osa(script).lower()
    return out == "true"

# 把如 "d3" 的棋盘坐标转换为屏幕像素点(x,y)，使用窗口矩形与棋盘相对矩形，按 a1 左上、h8 右下 的约定计算格子中心点
def _coord_to_xy(coord: str, win: Tuple[int,int,int,int], board_rel: Tuple[int,int,int,int]) -> Tuple[int,int]:
    # coord: "d3"；a1 左上，h8 右下
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

# ---------------------------------------- OCR phase below

def _get_board_rect_px() -> Tuple[int,int,int,int]:
    """
    读取标定并计算棋盘的屏幕绝对矩形 (x,y,w,h)。
    """
    cfg = _load_calib()
    wx, wy, ww, wh = _get_window_bounds(APP_NAME)
    board_rel = cfg.get("board_rel")
    if not (isinstance(board_rel, list) and len(board_rel) == 4):
        raise FileNotFoundError(f"board_rel missing in {CALIB_PATH}. Run --probe-calibrate first.")
    bx, by, bw, bh = board_rel
    return wx + int(bx), wy + int(by), int(bw), int(bh)

def _screenshot_board() -> "Image.Image":
    """
    截取棋盘区域为 PIL.Image（需要屏幕录制权限）。
    """
    from PIL import ImageGrab  # 延迟导入
    x, y, w, h = _get_board_rect_px()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    return img

def _xy_to_coord_local(x: float, y: float, w: int, h: int) -> str:
    """
    将棋盘内相对坐标(x,y)映射为坐标字符串（a1 左上，h8 右下）。
    """
    cell_w = w / 8.0
    cell_h = h / 8.0
    c = int(max(0, min(7, x // cell_w)))   # 0..7 左→右
    r = int(max(0, min(7, y // cell_h)))   # 0..7 上→下
    return f"{chr(ord('a') + c)}{r + 1}"

def _analyze_board_best(img_pil: "Image.Image", debug_dir: Optional[str] = None) -> Tuple[List[str], float]:
    """
    基于棋盘截图，定位黄色最佳点，OCR 读数值。
    返回 ([best_move], net_win)。a1 左上，h8 右下。
    """
    import numpy as np
    import cv2
    import pytesseract
    from PIL import Image

    # 1) 转 OpenCV 图像并做黄色分割
    img_rgb = np.array(img_pil.convert("RGB"))  # HxWx3, RGB
    H, W = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 黄色阈值（适度放宽）
    lower = (12, 80, 150)   # H,S,V
    upper = (60, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学：先闭运算连通，再轻微膨胀
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    # 2) 连通域候选
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 12:  # 放低面积门槛
            continue
        cx, cy = centroids[i]
        pad = 3  # 适度放大候选框
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        candidates.append((x0, y0, x1, y1, cx, cy, area))

    if not candidates:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(img_rgb).save(os.path.join(debug_dir, "sensei_board.png"))
            cv2.imwrite(os.path.join(debug_dir, "sensei_mask.png"), mask)
        raise RuntimeError("no yellow candidates found; adjust color threshold or ensure highlights are visible")

    # 3) 对候选框做 OCR（多种预处理与配置尝试）
    results = []
    try_cfgs = [
        "--psm 7 -c tessedit_char_whitelist=+-0123456789.",
        "--psm 6 -c tessedit_char_whitelist=+-0123456789.",
        "--psm 8 -c tessedit_char_whitelist=+-0123456789.",
    ]
    roi_idx = 0
    for x0, y0, x1, y1, cx, cy, area in candidates:
        # 仅保留黄色前景，去掉背景
        roi_rgb = img_rgb[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]
        fg = cv2.bitwise_and(roi_rgb, roi_rgb, mask=roi_mask)

        # 灰度 + OTSU 二值（白底黑字）
        gray = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)
        bin1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        bin_inv = 255 - bin1
        bin_inv = cv2.dilate(bin_inv, np.ones((2, 2), np.uint8), iterations=1)
        up = cv2.resize(bin_inv, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

        txt = ""
        num_match = None
        for cfg in try_cfgs:
            s = pytesseract.image_to_string(up, config=cfg).strip()
            m = re.search(r'[+-]?\d+(?:\.\d+)?', s)
            if m:
                txt = s
                num_match = m
                break

        # 若失败，换自适应阈值再试
        if num_match is None:
            bin2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
            )
            bin2 = cv2.dilate(bin2, np.ones((2, 2), np.uint8), iterations=1)
            up2 = cv2.resize(bin2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            for cfg in try_cfgs:
                s = pytesseract.image_to_string(up2, config=cfg).strip()
                m = re.search(r'[+-]?\d+(?:\.\d+)?', s)
                if m:
                    txt = s
                    num_match = m
                    up = up2
                    break

        if num_match is None:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"roi_{roi_idx:02d}_gray.png"), gray)
                cv2.imwrite(os.path.join(debug_dir, f"roi_{roi_idx:02d}_bin.png"), bin1)
                cv2.imwrite(os.path.join(debug_dir, f"roi_{roi_idx:02d}_up.png"), up)
            roi_idx += 1
            continue

        try:
            val = float(num_match.group(0))
        except ValueError:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"roi_{roi_idx:02d}_raw.txt"), "w", encoding="utf-8") as f:
                    f.write(txt)
            roi_idx += 1
            continue

        coord = _xy_to_coord_local(cx, cy, W, H)
        results.append((coord, val, (x0, y0, x1, y1), (cx, cy)))

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"roi_{roi_idx:02d}_ok.png"), up)
        roi_idx += 1

    if not results:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(img_rgb).save(os.path.join(debug_dir, "sensei_board.png"))
            cv2.imwrite(os.path.join(debug_dir, "sensei_mask.png"), mask)
        raise RuntimeError("OCR got no usable numbers from yellow candidates")

    # 4) 合并同格，选取最佳
    merged: dict = {}
    for coord, val, box, ctr in results:
        if coord not in merged or val > merged[coord][0]:
            merged[coord] = (val, box, ctr)

    pairs = [(c, v[0]) for c, v in merged.items()]
    pairs.sort(key=lambda t: t[1], reverse=True)
    best_val = pairs[0][1]
    tied = [p for p in pairs if abs(p[1] - best_val) <= 0.1 + 1e-9]
    tied.sort(key=lambda t: t[0])
    best_move = tied[0][0]
    net_win = float(best_val)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        base = os.path.join(debug_dir, "sensei")
        Image.fromarray(img_rgb).save(base + "_board.png")
        cv2.imwrite(base + "_mask.png", mask)
        vis = img_bgr.copy()
        for coord, (val, box, ctr) in merged.items():
            x0, y0, x1, y1 = box
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.putText(
                vis, f"{coord}:{val:.2f}", (x0, max(10, y0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA
            )
        cv2.imwrite(base + "_vis.png", vis)

    return [best_move], net_win
    
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
        data = _save_calib_merged({"app": APP_NAME, "board_rel": [int(bx), int(by), int(bw), int(bh)]})
        print(f"[CAL] saved to {CALIB_PATH}: board_rel={data['board_rel']}")
        return CALIB_PATH
    
    # 标定“<<”复位按钮（相对窗口坐标）
    def probe_calibrate_nav(self) -> str:
        import pyautogui
        pyautogui.FAILSAFE = False
        self.ensure_running()
        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        print("[CAL] 把鼠标移动到 顶部工具栏的 <<（回到开局） 图标上，按回车确认...")
        input()
        p = pyautogui.position()
        nx, ny = p.x - wx, p.y - wy
        data = _save_calib_merged({"nav_reset_rel": [int(nx), int(ny)]})
        print(f"[CAL] saved to {CALIB_PATH}: nav_reset_rel={data['nav_reset_rel']}")
        return CALIB_PATH
    
    # 标定“New game?”弹窗里的 Yes 按钮（相对窗口坐标）
    def probe_calibrate_yes(self) -> str:
        import pyautogui
        pyautogui.FAILSAFE = False
        self.ensure_running()
        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        print("[CAL] 请先让弹窗出现（比如点一次 <<），把鼠标移动到弹窗的 Yes 按钮上，按回车确认...")
        input()
        p = pyautogui.position()
        yx, yy = p.x - wx, p.y - wy
        data = _save_calib_merged({"nav_yes_rel": [int(yx), int(yy)]})
        print(f"[CAL] saved to {CALIB_PATH}: nav_yes_rel={data['nav_yes_rel']}")
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

    def probe_read(self) -> Tuple[List[str], float]:
        """
        截取棋盘 -> 识别黄色最佳点 -> 打印并保存调试图到 /tmp/sensei_read_<ts>/
        """
        self.ensure_running()
        # 等待一点时间让评估稳定（用 engine_time；没有就默认 4s）
        time.sleep(float(getattr(self, "engine_time", 4.0)))
        img = _screenshot_board()
        tsdir = f"/tmp/sensei_read_{int(time.time())}"
        best_moves, net_win = _analyze_board_best(img, debug_dir=tsdir)
        print(f"[READ] best_moves={best_moves} net_win={net_win:.2f}")
        print(f"[READ] debug saved to {tsdir}")
        return best_moves, net_win
    
    def reset_board(self) -> None:
        import pyautogui
        pyautogui.FAILSAFE = False
        self.ensure_running()
        cfg = _load_calib()
        nav = cfg.get("nav_reset_rel")
        if not (isinstance(nav, list) and len(nav) == 2):
            raise FileNotFoundError(f"nav_reset_rel missing in {CALIB_PATH}. Run --probe-calibrate-nav first.")

        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        # 1) 点“<<”
        nx, ny = nav
        pyautogui.click(wx + int(nx), wy + int(ny))
        time.sleep(0.20)

        # 2) 先用键盘兜底（很多弹窗 Enter = 默认 Yes）
        pyautogui.press("enter")
        time.sleep(0.12)

        # 3) 如果你标定了 Yes，就直接点它（即使 AX 不可见也有效）
        yes = cfg.get("nav_yes_rel")
        if isinstance(yes, list) and len(yes) == 2:
            yx, yy = yes
            pyautogui.click(wx + int(yx), wy + int(yy))
            time.sleep(0.12)

        # 4) 再补一次 Enter，确保关闭
        pyautogui.press("enter")
        time.sleep(0.10)

        # 5) 最后再尝试一次 AX 方式（若可见就点掉）
        _dismiss_new_game_prompt(APP_NAME)

    def replay_moves(self, moves_played: List[str]) -> None:
        # 依次点子；"--" 跳过
        import pyautogui
        pyautogui.FAILSAFE = False
        self.ensure_running()
        cfg = _load_calib()
        board_rel = cfg.get("board_rel")
        if not (isinstance(board_rel, list) and len(board_rel) == 4):
            raise FileNotFoundError(f"board_rel missing in {CALIB_PATH}. Run --probe-calibrate first.")
        wx, wy, ww, wh = _get_window_bounds(APP_NAME)
        bx, by, bw, bh = board_rel
        for mv in moves_played:
            if mv == "--": 
                time.sleep(0.12); continue
            x, y = _coord_to_xy(mv, (wx,wy,ww,wh), (bx,by,bw,bh))
            pyautogui.click(x, y)
            time.sleep(0.12)

    def wait_and_read(self) -> Tuple[List[str], float]:
        """
        正式读取：等待 engine_time 秒，然后返回 ([best_move], net_win)
        """
        time.sleep(float(getattr(self, "engine_time", 4.0)))
        img = _screenshot_board()
        best_moves, net_win = _analyze_board_best(img, debug_dir=None)
        return best_moves, net_win