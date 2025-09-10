# 绝对贪心基线：在所有合法着法中，选择“立即翻子数”最多的一手。
# 接口设计与项目一致：move 索引 0..63，PASS = 64。
# 复用 botzone.play.OthelloAI 的位运算合法生成/落子逻辑，避免重复实现。

from typing import Tuple, List
from botzone.play import OthelloAI, popcount

PASS_INDEX = 64

def enumerate_moves(bb: int) -> List[int]:
    res = []
    m = bb
    while m:
        l = m & -m
        res.append(l.bit_length() - 1)
        m ^= l
    return res

class GreedyBaseline:
    def __init__(self):
        # 复用 play.py 的位运算实现（合法着/落子）
        self._ai = OthelloAI()

    def legal_moves(self, my: int, opp: int) -> int:
        return self._ai.get_legal_moves(my, opp)

    def count_flips(self, my: int, opp: int, pos: int) -> int:
        # 使用 make_move 计算落子后我方子数增加量（= 翻子数）
        new_my, new_opp = self._ai.make_move(my, opp, pos)
        # 非法着（没有翻子）会返回原局面，此时翻子数为 0
        return popcount(new_my) - popcount(my) - 1 if ((my | opp) & (1 << pos)) == 0 else max(0, popcount(new_my) - popcount(my) - 1)

    def choose_move(self, my: int, opp: int) -> int:
        """返回最佳着法索引 0..63；无合法着法返回 PASS_INDEX=64。"""
        legal = self.legal_moves(my, opp)
        if legal == 0:
            return PASS_INDEX
        best_pos = None
        best_flips = -1
        for pos in enumerate_moves(legal):
            flips = self.count_flips(my, opp, pos)
            if flips > best_flips or (flips == best_flips and best_pos is not None and pos < best_pos):
                best_flips = flips
                best_pos = pos
        return best_pos if best_pos is not None else PASS_INDEX

    def apply_best(self, my: int, opp: int) -> Tuple[int, int, int]:
        """选择并应用最佳着法，返回 (new_my, new_opp, move_idx)。若 PASS 则返回 (opp, my, 64)。"""
        mv = self.choose_move(my, opp)
        if mv == PASS_INDEX:
            # 与项目约定一致：PASS 后换边
            return opp, my, PASS_INDEX
        new_my, new_opp = self._ai.make_move(my, opp, mv)
        return new_my, new_opp, mv

# 便捷函数：与类方法等价
_greedy_singleton = GreedyBaseline()

def choose_move(my_pieces: int, opp_pieces: int) -> int:
    """顶层接口：返回 0..63；无着返回 64。"""
    return _greedy_singleton.choose_move(my_pieces, opp_pieces)

def apply_best(my_pieces: int, opp_pieces: int) -> Tuple[int, int, int]:
    """选择并应用：返回 (new_my, new_opp, move_idx)。"""
    return _greedy_singleton.apply_best(my_pieces, opp_pieces)