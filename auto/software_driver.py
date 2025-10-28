from typing import List, Tuple
import time

class SoftwareDriver:
    """
    Placeholder for external software automation.
    实际实现时替换 ensure_running/reset_board/replay_moves/wait_and_read。

    外部软件的适配层：
    - ensure_running: 启动/附着软件
    - reset_board: 回到初始棋盘
    - replay_moves: 回放 moves_played（"--" 为 PASS）
    - wait_and_read: 等待 engine_time 后读取 (best_moves, net_win)
    - solve: 组合调用以上步骤
    备注：mock=True 时走快速占位返回，便于联调。
    """
    def __init__(self, engine_time: float, mock: bool = False):
        self.engine_time = engine_time
        self.mock = mock
    
    def ensure_running(self):
        return
    
    def reset_board(self):
        return
    
    def replay_moves(self, moves_played: List[str]):
        return
    
    # 识别最好解 (move 和 net_win)
    def wait_and_read(self) -> Tuple[List[str], float]:
        if self.mock:   # test: 这是“假驱动”模式，用于快速联调整条流水线。
            time.sleep(min(0.02, self.engine_time))
            return ["a1"], 0.0
        raise NotImplementedError("Implement UI reading for non-mock mode") # TODO: 非 mock 情况。
    
    def solve(self, moves_played: List[str]) -> Tuple[List[str], float]:
        self.ensure_running()
        self.reset_board()
        self.replay_moves(moves_played)
        return self.wait_and_read()