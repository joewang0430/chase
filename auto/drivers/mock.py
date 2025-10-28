# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/mock.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/mock.py
from typing import List, Tuple
import time
from .base import SoftwareDriverBase

class MockDriver(SoftwareDriverBase):
    def ensure_running(self) -> None:
        return
    def reset_board(self) -> None:
        return
    def replay_moves(self, moves_played: List[str]) -> None:
        return
    def wait_and_read(self) -> Tuple[List[str], float]:
        time.sleep(min(0.02, self.engine_time))
        return ["a1"], 0.0