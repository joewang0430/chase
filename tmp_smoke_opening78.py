'''
生成 1–6 步；
强制 7–8 走“噪声”分支（避免 mock oracle 给出非法点）；
打印 moves 和 logs。
'''

# python3 tmp_smoke_opening78.py

import random
from auto.generate_dataset import opening_moves_1_to_6, opening_moves_7_to_8

# Seed RNG for reproducibility
rng = random.Random(42)

# Generate 1-6 locally
moves_1_6, b, w, side = opening_moves_1_to_6(rng=rng)
print("moves_1_6:", moves_1_6, "side:", side)

# Force noise branch for 7-8 to avoid mock oracle proposing illegal moves
class NoOracleRng(random.Random):
    def random(self):
        return 0.6  # >0.5 => use_oracle=False

norng = NoOracleRng(123)

# Run 7-8 with mock driver (no UI), ensure it returns two moves and logs
new_moves, b2, w2, side2, logs = opening_moves_7_to_8(
    moves_so_far=list(moves_1_6),
    black_bb=b,
    white_bb=w,
    side_to_move=side,
    driver_name="mac",
    mock=True,
    rng=norng,
)
print("moves_7_8:", new_moves, "side:", side2)
print("logs:", logs)
