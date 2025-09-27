# python3 -m utils.rotators

class Rotators:
    @staticmethod
    def up_diag_mirror(me: int, opp: int) -> tuple[int, int]:
        """
        Mirror both bitboards along the up-diagonal (left-bottom ↗ right-top).
        (r,c) -> (7-c, 7-r), index = r*8 + c.
        """
        def _mirror(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = (7 - c) * 8 + (7 - r)
                res |= (1 << j)
                x ^= lsb
            return res
        return _mirror(me), _mirror(opp)
    
    @staticmethod
    def down_diag_mirror(me: int, opp: int) -> tuple[int, int]:
        """
        Mirror both bitboards along the down-diagonal (left-top ↘ right-bottom).
        (r,c) -> (c, r), index = r*8 + c.
        """
        def _mirror(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = c * 8 + r
                res |= (1 << j)
                x ^= lsb
            return res
        return _mirror(me), _mirror(opp)

    @staticmethod
    def rotation_180(me: int, opp: int) -> tuple[int, int]:
        """
        Rotate both bitboards 180 degrees.
        (r,c) -> (7-r, 7-c), index = r*8 + c.
        """
        def _rot(bb: int) -> int:
            if bb == 0:
                return 0
            res = 0
            x = bb
            while x:
                lsb = x & -x
                i = lsb.bit_length() - 1
                r, c = divmod(i, 8)
                j = (7 - r) * 8 + (7 - c)
                res |= (1 << j)
                x ^= lsb
            return res
        return _rot(me), _rot(opp)
    


### Test above functions ###
import os
try:
    from converters import Converters
except ImportError:
    # 支持作为包运行 (python -m utils.rotators)
    from .converters import Converters

def _print_section(title: str, me: int, opp: int):
    print(f"=== {title} ===")
    print(Converters.bb_to_visual(me, opp))
    print()

if __name__ == "__main__":
    # 直接从同目录 tested_board.txt 读取棋盘
    here = os.path.dirname(os.path.abspath(__file__))
    board_path = os.path.join(here, "tested_board.txt")

    if not os.path.exists(board_path):
        print("tested_board.txt not found in", here)
        exit(1)

    with open(board_path, "r", encoding="utf-8") as f:
        board_text = f.read()

    if not board_text.strip():
        print("tested_board.txt is empty.")
        exit(1)

    me, opp = Converters.visual_to_bb(board_text)

    _print_section("ORIGINAL", me, opp)

    me_u, opp_u = Rotators.up_diag_mirror(me, opp)
    _print_section("UP_DIAG_MIRROR (↗ 反对角线翻转)", me_u, opp_u)

    me_d, opp_d = Rotators.down_diag_mirror(me, opp)
    _print_section("DOWN_DIAG_MIRROR (↘ 主对角线翻转)", me_d, opp_d)

    me_r, opp_r = Rotators.rotation_180(me, opp)
    _print_section("ROTATION_180 (180° 旋转)", me_r, opp_r)