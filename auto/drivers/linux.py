# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/linux.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/linux.py
from typing import List, Tuple
from .base import SoftwareDriverBase

class LinuxDriver(SoftwareDriverBase):
    def ensure_running(self) -> None:
        return
    def reset_board(self) -> None:
        return
    def replay_moves(self, moves_played: List[str]) -> None:
        return
    def wait_and_read(self) -> Tuple[List[str], float]:
        raise NotImplementedError("Implement Linux UI reading")
    def click_move(self, coord: str, delay: float = 0.12) -> None:
        raise NotImplementedError("Implement Linux UI clicking")