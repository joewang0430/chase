# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/base.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

class SoftwareDriverBase(ABC):
    def __init__(self, engine_time: float = 4.0):
        self.engine_time = engine_time

    @abstractmethod
    def ensure_running(self) -> None: ...
    @abstractmethod
    def reset_board(self) -> None: ...
    @abstractmethod
    def replay_moves(self, moves_played: List[str]) -> None: ...
    @abstractmethod
    def wait_and_read(self) -> Tuple[List[str], float]: ...

    def solve(self, moves_played: List[str]) -> Tuple[List[str], float]:
        self.ensure_running()
        self.reset_board()
        self.replay_moves(moves_played)
        return self.wait_and_read()