# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/mac.py
from typing import List, Tuple
from .base import SoftwareDriverBase

class MacDriver(SoftwareDriverBase):
    def ensure_running(self) -> None:
        # TODO: osascript/pyobjc 启动或附着软件
        return

    def reset_board(self) -> None:
        # TODO: 复位到初始局面
        return

    def replay_moves(self, moves_played: List[str]) -> None:
        # TODO: 点击/快捷键复盘；"--" 表示 PASS
        return

    def wait_and_read(self) -> Tuple[List[str], float]:
        # TODO: 等 self.engine_time 并解析 UI
        raise NotImplementedError("Implement macOS UI reading")