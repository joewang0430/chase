# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/windows.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/windows.py
from typing import List, Tuple
from .base import SoftwareDriverBase

class WindowsDriver(SoftwareDriverBase):
    def ensure_running(self) -> None:
        return
    def reset_board(self) -> None:
        return
    def replay_moves(self, moves_played: List[str]) -> None:
        return
    def wait_and_read(self) -> Tuple[List[str], float]:
        raise NotImplementedError("Implement Windows UI reading")